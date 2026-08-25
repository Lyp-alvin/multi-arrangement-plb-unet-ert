from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from matplotlib.colors import Colormap
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from lnn_unet import build_lnn_unet
from train_wa_closed_loop_lnn import WASingleArrayDataset, seed_worker


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "wa_single_closed_loop"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.target_sum = 0.0
        self.target_square_sum = 0.0
        self.square_error_sum = 0.0
        self.absolute_error_sum = 0.0

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target64 = np.asarray(target, dtype=np.float64).ravel()
        prediction64 = np.asarray(prediction, dtype=np.float64).ravel()
        error = prediction64 - target64
        self.count += target64.size
        self.target_sum += float(target64.sum())
        self.target_square_sum += float(np.square(target64).sum())
        self.square_error_sum += float(np.square(error).sum())
        self.absolute_error_sum += float(np.abs(error).sum())

    def pooled_metrics(self) -> dict[str, float]:
        mse = self.square_error_sum / max(self.count, 1)
        target_ss = self.target_square_sum - self.target_sum**2 / max(self.count, 1)
        return {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.absolute_error_sum / max(self.count, 1),
            "r2": 1.0 - self.square_error_sum / max(target_ss, np.finfo(float).eps),
        }


def build_loader(dataset: WASingleArrayDataset, batch_size: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload.get("config", {})
    variant = config.get("variant", "base")
    model = build_lnn_unet(variant, input_channels=1, output_channels=1).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def safe_data_range(target: np.ndarray) -> float:
    data_range = float(np.max(target) - np.min(target))
    return max(data_range, np.finfo(np.float32).eps)


def basic_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    error = prediction64 - target64
    mse = float(np.mean(np.square(error)))
    target_ss = float(np.sum(np.square(target64 - np.mean(target64))))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0 - float(np.sum(np.square(error))) / max(target_ss, np.finfo(float).eps),
        "psnr": 20.0 * math.log10(safe_data_range(target64) / max(math.sqrt(mse), np.finfo(float).eps)),
    }


def inverse_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    metrics = basic_metrics(target, prediction)
    metrics["ssim"] = float(
        structural_similarity(target, prediction, data_range=safe_data_range(target))
    )
    return metrics


def masked_forward_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    valid = np.asarray(mask) > 0.5
    target_valid = target[valid]
    prediction_valid = prediction[valid]
    metrics = basic_metrics(target_valid, prediction_valid)

    # Average the SSIM map only where the complete 7x7 window is inside the mask.
    fill_value = float(np.mean(target_valid))
    target_filled = np.where(valid, target, fill_value)
    prediction_filled = np.where(valid, prediction, fill_value)
    _, ssim_map = structural_similarity(
        target_filled,
        prediction_filled,
        data_range=safe_data_range(target_valid),
        full=True,
    )
    interior = binary_erosion(valid, iterations=3, border_value=0)
    metrics["ssim"] = float(np.mean(ssim_map[interior if interior.any() else valid]))
    metrics["valid_fraction"] = float(np.mean(valid))
    return metrics


def summarize_rows(rows: list[dict[str, float | int]], accumulator: MetricAccumulator) -> dict:
    metric_names = ("mse", "rmse", "mae", "r2", "psnr", "ssim")
    per_sample: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        per_sample[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {
        "samples": len(rows),
        "value_domain": "MAT array values used during training (log10 resistivity)",
        "pooled": accumulator.pooled_metrics(),
        "per_sample": per_sample,
    }


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def white_bad_cmap(name: str = "turbo") -> Colormap:
    cmap = matplotlib.colormaps[name].copy()
    cmap.set_bad("white")
    return cmap


def image_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(array)[np.isfinite(array)].ravel() for array in arrays])
    return float(np.min(finite)), float(np.max(finite))


def plot_forward_sample(
    output_path: Path,
    file_id: int,
    rho: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    metrics: dict[str, float],
) -> None:
    valid = mask > 0.5
    target_masked = np.ma.masked_where(~valid, target)
    prediction_masked = np.ma.masked_where(~valid, prediction)
    error_masked = np.ma.masked_where(~valid, np.abs(prediction - target))
    value_min, value_max = image_limits(target_masked.filled(np.nan), prediction_masked.filled(np.nan))
    error_max = float(np.percentile(np.abs(prediction[valid] - target[valid]), 99))
    cmap = white_bad_cmap()

    fig, axes = plt.subplots(2, 2, figsize=(15, 7.8), constrained_layout=True)
    panels = (
        (rho, "Input: true resistivity model", None, None),
        (target_masked, "Target: WA forward response", value_min, value_max),
        (prediction_masked, "Prediction: WA forward response", value_min, value_max),
        (error_masked, "Absolute error (valid mask only)", 0.0, max(error_max, 1e-8)),
    )
    for axis, (array, title, vmin, vmax) in zip(axes.flat, panels):
        image = axis.imshow(array, cmap=cmap, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Horizontal sample")
        axis.set_ylabel("Vertical sample")
        axis.set_facecolor("white")
        fig.colorbar(image, ax=axis, shrink=0.88, label="log10 resistivity")
    fig.suptitle(
        f"WA forward prediction | ID {file_id} | RMSE={metrics['rmse']:.5f}, "
        f"MAE={metrics['mae']:.5f}, R2={metrics['r2']:.5f}, SSIM={metrics['ssim']:.5f}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_inverse_sample(
    output_path: Path,
    file_id: int,
    traditional_inverse: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, float],
) -> None:
    value_min, value_max = image_limits(traditional_inverse, target, prediction)
    error = np.abs(prediction - target)
    error_max = float(np.percentile(error, 99))
    cmap = white_bad_cmap()

    fig, axes = plt.subplots(2, 2, figsize=(15, 7.8), constrained_layout=True)
    panels = (
        (traditional_inverse, "Input: traditional WA inversion", value_min, value_max),
        (target, "Target: true resistivity model", value_min, value_max),
        (prediction, "Prediction: neural inversion", value_min, value_max),
        (error, "Absolute error", 0.0, max(error_max, 1e-8)),
    )
    for axis, (array, title, vmin, vmax) in zip(axes.flat, panels):
        image = axis.imshow(array, cmap=cmap, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Horizontal sample")
        axis.set_ylabel("Vertical sample")
        fig.colorbar(image, ax=axis, shrink=0.88, label="log10 resistivity")
    fig.suptitle(
        f"WA inverse prediction | ID {file_id} | RMSE={metrics['rmse']:.5f}, "
        f"MAE={metrics['mae']:.5f}, R2={metrics['r2']:.5f}, SSIM={metrics['ssim']:.5f}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_loss_curve(csv_path: Path, output_path: Path, stage: str) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    if stage == "forward":
        train = np.asarray([float(row["train_Lfwd_raw"]) for row in rows])
        val = np.asarray([float(row["val_Lfwd_raw"]) for row in rows])
        title = "WA forward training loss"
        ylabel = "Masked forward MSE"
    else:
        train = np.asarray([float(row["train_Linv_raw"]) for row in rows])
        val = np.asarray([float(row["val_Linv_raw"]) for row in rows])
        title = "WA inverse closed-loop inversion loss"
        ylabel = "Inverse MSE"

    best_index = int(np.argmin(val))
    fig, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    axis.semilogy(epochs, train, color="#167d9a", linewidth=2.0, label="Training loss")
    axis.semilogy(epochs, val, color="#d1495b", linewidth=2.0, label="Validation loss")
    axis.scatter(
        epochs[best_index],
        val[best_index],
        color="#d1495b",
        edgecolor="black",
        linewidth=0.6,
        s=55,
        zorder=3,
        label=f"Best validation (epoch {epochs[best_index]})",
    )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def test_forward(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    mat_output_dir: Path,
    plot_ids: set[int],
) -> tuple[list[dict[str, float | int]], dict]:
    rows: list[dict[str, float | int]] = []
    accumulator = MetricAccumulator()
    output_dir.mkdir(parents=True, exist_ok=True)
    mat_output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch in loader:
            rho_tensor = batch["rho"].to(device, non_blocking=True)
            predictions = model(rho_tensor).cpu().numpy()[:, 0]
            rho_batch = batch["rho"].numpy()[:, 0]
            target_batch = batch["wa_img"].numpy()[:, 0]
            mask_batch = batch["wa_mask"].numpy()[:, 0]
            ids = batch["id"].numpy().tolist()
            for file_id, rho, target, prediction, mask in zip(
                ids, rho_batch, target_batch, predictions, mask_batch
            ):
                metrics = masked_forward_metrics(target, prediction, mask)
                row: dict[str, float | int] = {"id": int(file_id), **metrics}
                rows.append(row)
                valid = mask > 0.5
                accumulator.update(target[valid], prediction[valid])
                sio.savemat(
                    mat_output_dir / f"forward_prediction_id_{file_id}.mat",
                    {
                        "WA_pred": np.asarray(prediction, dtype=np.float32),
                        "mask": np.asarray(valid, dtype=np.uint8),
                    },
                    do_compression=True,
                )
                if int(file_id) in plot_ids:
                    plot_forward_sample(
                        output_dir / f"forward_prediction_id_{file_id}.png",
                        int(file_id), rho, target, prediction, mask, metrics,
                    )
    return rows, summarize_rows(rows, accumulator)


def test_inverse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    mat_output_dir: Path,
    plot_ids: set[int],
) -> tuple[list[dict[str, float | int]], dict]:
    rows: list[dict[str, float | int]] = []
    accumulator = MetricAccumulator()
    output_dir.mkdir(parents=True, exist_ok=True)
    mat_output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch in loader:
            inverse_tensor = batch["wa_inv"].to(device, non_blocking=True)
            predictions = model(inverse_tensor).cpu().numpy()[:, 0]
            inverse_batch = batch["wa_inv"].numpy()[:, 0]
            target_batch = batch["rho"].numpy()[:, 0]
            ids = batch["id"].numpy().tolist()
            for file_id, traditional_inverse, target, prediction in zip(
                ids, inverse_batch, target_batch, predictions
            ):
                metrics = inverse_metrics(target, prediction)
                row: dict[str, float | int] = {"id": int(file_id), **metrics}
                rows.append(row)
                accumulator.update(target, prediction)
                sio.savemat(
                    mat_output_dir / f"inverse_prediction_id_{file_id}.mat",
                    {"rho_pred": np.asarray(prediction, dtype=np.float32)},
                    do_compression=True,
                )
                if int(file_id) in plot_ids:
                    plot_inverse_sample(
                        output_dir / f"inverse_prediction_id_{file_id}.png",
                        int(file_id), traditional_inverse, target, prediction, metrics,
                    )
    return rows, summarize_rows(rows, accumulator)


def write_summary_text(path: Path, summary: dict) -> None:
    lines = [
        "WA closed-loop validation evaluation",
        "value_domain = MAT array values used during training (log10 resistivity)",
        "",
    ]
    for stage in ("forward", "inverse"):
        stage_summary = summary[stage]
        lines.append(f"[{stage}]")
        lines.append(f"samples = {stage_summary['samples']}")
        if stage == "forward":
            lines.append("metric_region = mask > 0.5 only")
        else:
            lines.append("metric_region = full 256 x 1024 image")
        for name, value in stage_summary["pooled"].items():
            lines.append(f"pooled_{name} = {value:.8f}")
        for name, values in stage_summary["per_sample"].items():
            lines.append(
                f"{name}: mean={values['mean']:.8f}, std={values['std']:.8f}, "
                f"median={values['median']:.8f}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained WA forward and inverse LNN-U-Nets")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--plot-count",
        type=int,
        default=0,
        help="Number of validation IDs to plot; 0 or less plots the complete validation set",
    )
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Skip PNG rendering while still exporting MAT predictions and metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or args.run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    forward_checkpoint = args.run_dir / "forward" / "forward_lnn_unet_best.pth"
    inverse_checkpoint = args.run_dir / "inverse" / "inverse_closed_loop_best.pth"
    if not forward_checkpoint.exists() or not inverse_checkpoint.exists():
        raise FileNotFoundError("Best forward and inverse checkpoints are required")

    forward_dataset = WASingleArrayDataset(args.data_root, "val", "forward")
    inverse_dataset = WASingleArrayDataset(args.data_root, "val", "inverse")
    if args.skip_png:
        plot_ids: set[int] = set()
    elif args.plot_count <= 0:
        plot_ids = set(forward_dataset.ids)
    else:
        plot_ids = set(forward_dataset.ids[: args.plot_count])
    print(
        f"device={device} validation_samples={len(forward_dataset)} plot_count={len(plot_ids)}",
        flush=True,
    )

    forward_model, forward_payload = load_model(forward_checkpoint, device)
    forward_rows, forward_summary = test_forward(
        forward_model,
        build_loader(forward_dataset, args.batch_size, args.num_workers),
        device,
        output_dir / "forward_predictions",
        output_dir / "forward_predictions_mat",
        plot_ids,
    )
    del forward_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_csv(output_dir / "forward_metrics_per_sample.csv", forward_rows)
    print(f"forward pooled metrics: {forward_summary['pooled']}", flush=True)

    inverse_model, inverse_payload = load_model(inverse_checkpoint, device)
    inverse_rows, inverse_summary = test_inverse(
        inverse_model,
        build_loader(inverse_dataset, args.batch_size, args.num_workers),
        device,
        output_dir / "inverse_predictions",
        output_dir / "inverse_predictions_mat",
        plot_ids,
    )
    write_csv(output_dir / "inverse_metrics_per_sample.csv", inverse_rows)
    print(f"inverse pooled metrics: {inverse_summary['pooled']}", flush=True)

    summary = {
        "split": "val",
        "validation_ids": str(args.data_root / "val_ids.txt"),
        "plotted_ids": sorted(plot_ids),
        "forward_checkpoint": {
            "path": str(forward_checkpoint),
            "epoch": int(forward_payload["epoch"]),
            "best_val_loss": float(forward_payload["best_val_loss"]),
        },
        "inverse_checkpoint": {
            "path": str(inverse_checkpoint),
            "epoch": int(inverse_payload["epoch"]),
            "best_val_loss": float(inverse_payload["best_val_loss"]),
        },
        "forward": forward_summary,
        "inverse": inverse_summary,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    write_summary_text(output_dir / "metrics_summary.txt", summary)
    plot_loss_curve(
        args.run_dir / "forward" / "losses.csv",
        output_dir / "forward_loss_curve.png",
        "forward",
    )
    plot_loss_curve(
        args.run_dir / "inverse" / "losses.csv",
        output_dir / "inverse_loss_curve.png",
        "inverse",
    )
    print(f"outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
