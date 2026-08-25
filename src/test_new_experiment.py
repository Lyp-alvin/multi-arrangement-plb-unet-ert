from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from test_wa_closed_loop import (
    MetricAccumulator,
    inverse_metrics,
    masked_forward_metrics,
    summarize_rows,
    white_bad_cmap,
    write_csv,
)
from multi_lnn_unet import (
    ARRANGEMENTS,
    build_multi_forward_lnn_unet,
    build_multi_inverse_lnn_unet,
)
from plain_unet import build_plain_unet
from train_multi_closed_loop_lnn import MultiArrangementDataset
from train_wa_closed_loop_lnn import WASingleArrayDataset, build_loader


STAGES = ("unet_open", "unet_forward", "unet_closed", "multi_forward", "multi_inverse")


def load_model(model, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array)[np.isfinite(array)].ravel() for array in arrays])
    return float(values.min()), float(values.max())


def plot_forward(path, file_id, arrangement, rho, target, prediction, mask, metrics):
    valid = mask > 0.5
    target_masked = np.ma.masked_where(~valid, target)
    prediction_masked = np.ma.masked_where(~valid, prediction)
    error_masked = np.ma.masked_where(~valid, np.abs(prediction - target))
    vmin, vmax = limits(target[valid], prediction[valid])
    error_max = max(float(np.percentile(np.abs(prediction[valid] - target[valid]), 99)), 1e-8)
    cmap = white_bad_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.8), constrained_layout=True)
    panels = (
        (rho, "Input: true resistivity model", None, None),
        (target_masked, f"Target: {arrangement.upper()} forward response", vmin, vmax),
        (prediction_masked, f"Prediction: {arrangement.upper()} forward response", vmin, vmax),
        (error_masked, "Absolute error (valid mask only)", 0.0, error_max),
    )
    for axis, (array, title, lower, upper) in zip(axes.flat, panels):
        image = axis.imshow(array, cmap=cmap, origin="upper", aspect="auto", vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_xlabel("Horizontal sample")
        axis.set_ylabel("Vertical sample")
        axis.set_facecolor("white")
        fig.colorbar(image, ax=axis, shrink=0.88, label="log10 resistivity")
    fig.suptitle(
        f"{arrangement.upper()} forward | ID {file_id} | RMSE={metrics['rmse']:.5f}, "
        f"R2={metrics['r2']:.5f}, SSIM={metrics['ssim']:.5f}", fontsize=13,
    )
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_single_inverse(path, file_id, input_image, target, prediction, metrics, title):
    vmin, vmax = limits(input_image, target, prediction)
    error = np.abs(prediction - target)
    error_max = max(float(np.percentile(error, 99)), 1e-8)
    cmap = white_bad_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.8), constrained_layout=True)
    panels = (
        (input_image, "Input: traditional WA inversion", vmin, vmax),
        (target, "Target: true resistivity model", vmin, vmax),
        (prediction, "Prediction: neural inversion", vmin, vmax),
        (error, "Absolute error", 0.0, error_max),
    )
    for axis, (array, panel_title, lower, upper) in zip(axes.flat, panels):
        image = axis.imshow(array, cmap=cmap, origin="upper", aspect="auto", vmin=lower, vmax=upper)
        axis.set_title(panel_title)
        axis.set_xlabel("Horizontal sample")
        axis.set_ylabel("Vertical sample")
        fig.colorbar(image, ax=axis, shrink=0.88, label="log10 resistivity")
    fig.suptitle(
        f"{title} | ID {file_id} | RMSE={metrics['rmse']:.5f}, "
        f"R2={metrics['r2']:.5f}, SSIM={metrics['ssim']:.5f}", fontsize=13,
    )
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_multi_inverse(path, file_id, inputs, target, prediction, metrics):
    vmin, vmax = limits(*inputs.values(), target, prediction)
    error = np.abs(prediction - target)
    error_max = max(float(np.percentile(error, 99)), 1e-8)
    cmap = white_bad_cmap()
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    panels = [
        (inputs[name], f"Input: traditional {name.upper()} inversion", vmin, vmax)
        for name in ARRANGEMENTS
    ] + [
        (target, "Target: true resistivity model", vmin, vmax),
        (prediction, "Prediction: multi-array inversion", vmin, vmax),
        (error, "Absolute error", 0.0, error_max),
    ]
    for axis, (array, panel_title, lower, upper) in zip(axes.flat, panels):
        image = axis.imshow(array, cmap=cmap, origin="upper", aspect="auto", vmin=lower, vmax=upper)
        axis.set_title(panel_title)
        axis.set_xlabel("Horizontal sample")
        axis.set_ylabel("Vertical sample")
        fig.colorbar(image, ax=axis, shrink=0.82, label="log10 resistivity")
    fig.suptitle(
        f"WA/WB/SLM multi-array inversion | ID {file_id} | RMSE={metrics['rmse']:.5f}, "
        f"R2={metrics['r2']:.5f}, SSIM={metrics['ssim']:.5f}", fontsize=13,
    )
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_loss(csv_path: Path, output_path: Path, stage: str):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    columns = {
        "unet_open": ("train_Linv_raw", "val_Linv_raw", "Plain U-Net WA open-loop loss", "Inversion MSE"),
        "unet_forward": ("train_Lfwd_raw", "val_Lfwd_raw", "Plain U-Net WA forward loss", "Masked forward MSE"),
        "unet_closed": ("train_loss_total", "val_loss_total", "Plain U-Net WA closed-loop loss", "Total loss"),
        "multi_forward": ("train_total", "val_total", "LNN-U-Net multi-forward loss", "Mean masked MSE"),
        "multi_inverse": ("train_total", "val_total", "LNN-U-Net multi-array closed-loop loss", "Total loss"),
    }
    train_key, val_key, title, ylabel = columns[stage]
    train = np.asarray([float(row[train_key]) for row in rows])
    val = np.asarray([float(row[val_key]) for row in rows])
    best = int(np.argmin(val))
    fig, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    axis.semilogy(epochs, train, color="#167d9a", linewidth=2, label="Training loss")
    axis.semilogy(epochs, val, color="#d1495b", linewidth=2, label="Validation loss")
    axis.scatter(epochs[best], val[best], color="#d1495b", edgecolor="black", s=55, zorder=3, label=f"Best validation (epoch {epochs[best]})")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def test_single_inverse(stage, model, dataset, loader, device, output_dir):
    png_dir = output_dir / "predictions_png"
    mat_dir = output_dir / "predictions_mat"
    png_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accumulator = MetricAccumulator()
    with torch.inference_mode():
        for batch in loader:
            predictions = model(batch["wa_inv"].to(device, non_blocking=True)).cpu().numpy()[:, 0]
            inputs = batch["wa_inv"].numpy()[:, 0]
            targets = batch["rho"].numpy()[:, 0]
            ids = batch["id"].numpy().tolist()
            for file_id, input_image, target, prediction in zip(ids, inputs, targets, predictions):
                metrics = inverse_metrics(target, prediction)
                rows.append({"id": int(file_id), **metrics})
                accumulator.update(target, prediction)
                sio.savemat(mat_dir / f"inverse_prediction_id_{file_id}.mat", {"rho_pred": prediction.astype(np.float32)}, do_compression=True)
                plot_single_inverse(
                    png_dir / f"inverse_prediction_id_{file_id}.png", int(file_id), input_image,
                    target, prediction, metrics,
                    "Plain U-Net WA open-loop inversion" if stage == "unet_open" else "Plain U-Net WA closed-loop inversion",
                )
    return rows, summarize_rows(rows, accumulator)


def test_plain_forward(model, dataset, loader, device, output_dir):
    png_dir = output_dir / "predictions_png"
    mat_dir = output_dir / "predictions_mat"
    png_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accumulator = MetricAccumulator()
    with torch.inference_mode():
        for batch in loader:
            predictions = model(batch["rho"].to(device, non_blocking=True)).cpu().numpy()[:, 0]
            rhos = batch["rho"].numpy()[:, 0]
            targets = batch["wa_img"].numpy()[:, 0]
            masks = batch["wa_mask"].numpy()[:, 0]
            ids = batch["id"].numpy().tolist()
            for file_id, rho, target, prediction, mask in zip(ids, rhos, targets, predictions, masks):
                metrics = masked_forward_metrics(target, prediction, mask)
                rows.append({"id": int(file_id), **metrics})
                valid = mask > 0.5
                accumulator.update(target[valid], prediction[valid])
                sio.savemat(
                    mat_dir / f"forward_prediction_id_{file_id}.mat",
                    {"WA_pred": prediction.astype(np.float32), "mask": valid.astype(np.uint8)},
                    do_compression=True,
                )
                plot_forward(png_dir / f"forward_prediction_id_{file_id}.png", int(file_id), "wa", rho, target, prediction, mask, metrics)
    return rows, summarize_rows(rows, accumulator)


def test_multi_forward(model, loader, device, output_dir):
    rows_by_name = {name: [] for name in ARRANGEMENTS}
    accumulators = {name: MetricAccumulator() for name in ARRANGEMENTS}
    for name in ARRANGEMENTS:
        (output_dir / name / "predictions_png").mkdir(parents=True, exist_ok=True)
        (output_dir / name / "predictions_mat").mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch in loader:
            predictions = {name: value.cpu().numpy()[:, 0] for name, value in model(batch["rho"].to(device, non_blocking=True)).items()}
            rhos = batch["rho"].numpy()[:, 0]
            ids = batch["id"].numpy().tolist()
            for batch_index, file_id in enumerate(ids):
                for name in ARRANGEMENTS:
                    target = batch["forward"][name].numpy()[batch_index, 0]
                    mask = batch["mask"][name].numpy()[batch_index, 0]
                    prediction = predictions[name][batch_index]
                    metrics = masked_forward_metrics(target, prediction, mask)
                    rows_by_name[name].append({"id": int(file_id), **metrics})
                    valid = mask > 0.5
                    accumulators[name].update(target[valid], prediction[valid])
                    stage_dir = output_dir / name
                    sio.savemat(
                        stage_dir / "predictions_mat" / f"forward_prediction_id_{file_id}.mat",
                        {f"{name.upper()}_pred": prediction.astype(np.float32), "mask": valid.astype(np.uint8)},
                        do_compression=True,
                    )
                    plot_forward(
                        stage_dir / "predictions_png" / f"forward_prediction_id_{file_id}.png",
                        int(file_id), name, rhos[batch_index], target, prediction, mask, metrics,
                    )
    summaries = {name: summarize_rows(rows_by_name[name], accumulators[name]) for name in ARRANGEMENTS}
    return rows_by_name, summaries


def test_multi_inverse(model, loader, device, output_dir):
    png_dir = output_dir / "predictions_png"
    mat_dir = output_dir / "predictions_mat"
    png_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accumulator = MetricAccumulator()
    with torch.inference_mode():
        for batch in loader:
            input_tensors = {name: batch["inverse"][name].to(device, non_blocking=True) for name in ARRANGEMENTS}
            predictions = model(input_tensors).cpu().numpy()[:, 0]
            targets = batch["rho"].numpy()[:, 0]
            ids = batch["id"].numpy().tolist()
            input_arrays = {name: batch["inverse"][name].numpy()[:, 0] for name in ARRANGEMENTS}
            for batch_index, file_id in enumerate(ids):
                target = targets[batch_index]
                prediction = predictions[batch_index]
                metrics = inverse_metrics(target, prediction)
                rows.append({"id": int(file_id), **metrics})
                accumulator.update(target, prediction)
                sio.savemat(mat_dir / f"inverse_prediction_id_{file_id}.mat", {"rho_pred": prediction.astype(np.float32)}, do_compression=True)
                plot_multi_inverse(
                    png_dir / f"inverse_prediction_id_{file_id}.png", int(file_id),
                    {name: input_arrays[name][batch_index] for name in ARRANGEMENTS},
                    target, prediction, metrics,
                )
    return rows, summarize_rows(rows, accumulator)


def main():
    parser = argparse.ArgumentParser(description="Evaluate each new WA/multi-array experiment stage")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_directories = {
        "unet_open": ("open_inverse", "inverse_open_loop_best.pth"),
        "unet_forward": ("forward", "forward_unet_best.pth"),
        "unet_closed": ("closed_inverse", "inverse_closed_loop_best.pth"),
        "multi_forward": ("forward", "forward_multi_lnn_best.pth"),
        "multi_inverse": ("inverse", "inverse_multi_closed_loop_best.pth"),
    }
    directory_name, checkpoint_name = stage_directories[args.stage]
    stage_dir = args.run_dir / directory_name
    output_dir = stage_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = stage_dir / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if args.stage in ("unet_open", "unet_closed"):
        dataset = WASingleArrayDataset(args.data_root, "val", "inverse")
        if args.max_samples > 0:
            dataset.ids = dataset.ids[: args.max_samples]
        model, payload = load_model(build_plain_unet(), checkpoint, device)
        rows, summary = test_single_inverse(args.stage, model, dataset, build_loader(dataset, args.batch_size, False, args.num_workers, 43), device, output_dir)
        write_csv(output_dir / "metrics_per_sample.csv", rows)
    elif args.stage == "unet_forward":
        dataset = WASingleArrayDataset(args.data_root, "val", "forward")
        if args.max_samples > 0:
            dataset.ids = dataset.ids[: args.max_samples]
        model, payload = load_model(build_plain_unet(), checkpoint, device)
        rows, summary = test_plain_forward(model, dataset, build_loader(dataset, args.batch_size, False, args.num_workers, 43), device, output_dir)
        write_csv(output_dir / "metrics_per_sample.csv", rows)
    elif args.stage == "multi_forward":
        dataset = MultiArrangementDataset(args.data_root, "val", include_inverse=False)
        if args.max_samples > 0:
            dataset.ids = dataset.ids[: args.max_samples]
        model, payload = load_model(build_multi_forward_lnn_unet(), checkpoint, device)
        rows_by_name, summary = test_multi_forward(model, build_loader(dataset, args.batch_size, False, args.num_workers, 43), device, output_dir)
        for name, rows in rows_by_name.items():
            write_csv(output_dir / name / "metrics_per_sample.csv", rows)
    else:
        dataset = MultiArrangementDataset(args.data_root, "val", include_inverse=True)
        if args.max_samples > 0:
            dataset.ids = dataset.ids[: args.max_samples]
        model, payload = load_model(build_multi_inverse_lnn_unet(), checkpoint, device)
        rows, summary = test_multi_inverse(model, build_loader(dataset, args.batch_size, False, args.num_workers, 43), device, output_dir)
        write_csv(output_dir / "metrics_per_sample.csv", rows)

    result = {
        "stage": args.stage,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "best_val_loss": float(payload["best_val_loss"]),
        "validation_samples": len(dataset),
        "metrics": summary,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_loss(stage_dir / "losses.csv", output_dir / "loss_curve.png", args.stage)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
