from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from multi_lnn_unet import (
    ARRANGEMENTS,
    build_multi_forward_lnn_unet,
    build_multi_inverse_lnn_unet,
)
from train_wa_closed_loop_lnn import (
    IMAGE_SHAPE,
    adaptive_loss_weights,
    append_csv,
    build_loader,
    create_logger,
    load_mat_array,
    masked_mse,
    read_ids,
    set_seed,
)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_WORK_DIR = Path(__file__).resolve().parent / "runs" / "lnn_multi_closed_loop"
INVERSE_INFO = {
    "wa": ("inv_input_wa", "wainv", "WA"),
    "wb": ("inv_input_wb", "wbinv", "WB"),
    "slm": ("inv_input_slm", "slminv", "SLM"),
}


@dataclass(frozen=True)
class Config:
    stage: str
    data_root: str
    work_dir: str
    batch_size: int
    accumulation_steps: int
    effective_batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    scheduler_patience: int
    scheduler_factor: float
    scheduler_threshold: float
    early_stopping_patience: int
    num_workers: int
    seed: int
    adaptive_weight_eps: float
    arrangements: tuple[str, ...]
    use_amp: bool = False


class MultiArrangementDataset(Dataset):
    def __init__(self, data_root: Path, split: str, include_inverse: bool):
        self.data_root = data_root
        self.ids = read_ids(data_root / f"{split}_ids.txt")
        self.include_inverse = include_inverse
        self._validate_files()

    def _validate_files(self) -> None:
        missing = []
        for file_id in self.ids:
            paths = [self.data_root / "rho" / f"rho_{file_id}.mat"]
            for name in ARRANGEMENTS:
                paths.append(self.data_root / f"{name}_256_layered" / f"rhoa_2d_{file_id}.mat")
                if self.include_inverse:
                    directory, prefix, _ = INVERSE_INFO[name]
                    paths.append(self.data_root / directory / f"{prefix}_{file_id}.mat")
            missing.extend(str(path) for path in paths if not path.exists())
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} files: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        file_id = self.ids[index]
        rho = load_mat_array(self.data_root / "rho" / f"rho_{file_id}.mat", "rho")
        forward_images = {}
        masks = {}
        inverse_images = {}
        for name in ARRANGEMENTS:
            payload = sio.loadmat(
                self.data_root / f"{name}_256_layered" / f"rhoa_2d_{file_id}.mat"
            )
            image = np.asarray(payload["img"], dtype=np.float32).squeeze()
            mask = np.asarray(payload["mask"], dtype=np.float32).squeeze()
            if image.shape != IMAGE_SHAPE or mask.shape != IMAGE_SHAPE:
                raise ValueError(f"Invalid {name} shape for ID {file_id}")
            if not np.isfinite(image).all() or not np.isfinite(mask).all() or mask.sum() <= 0:
                raise ValueError(f"Invalid {name} values for ID {file_id}")
            forward_images[name] = torch.from_numpy(image[None])
            masks[name] = torch.from_numpy(mask[None])
            if self.include_inverse:
                directory, prefix, key = INVERSE_INFO[name]
                inverse = load_mat_array(
                    self.data_root / directory / f"{prefix}_{file_id}.mat", key
                )
                inverse_images[name] = torch.from_numpy(inverse[None])
        return {
            "id": file_id,
            "rho": torch.from_numpy(rho[None]),
            "forward": forward_images,
            "mask": masks,
            "inverse": inverse_images,
        }


def save_checkpoint(path, stage, epoch, model, optimizer, scheduler, best_val, no_improvement, config):
    torch.save(
        {
            "stage": stage,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val,
            "epochs_without_improvement": no_improvement,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload["epoch"]) + 1, float(payload["best_val_loss"]), int(payload["epochs_without_improvement"])


def prepare(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_dir = Path(args.work_dir) / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        stage=args.stage,
        data_root=str(Path(args.data_root).resolve()),
        work_dir=str(Path(args.work_dir).resolve()),
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        effective_batch_size=args.batch_size * args.accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_threshold=args.scheduler_threshold,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        adaptive_weight_eps=args.adaptive_weight_eps,
        arrangements=ARRANGEMENTS,
    )
    (stage_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    logger = create_logger(stage_dir / "train.log")
    logger.info("Stage=%s device=%s config=%s", args.stage, device, asdict(config))
    return stage_dir, config, logger, device


def make_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        threshold=args.scheduler_threshold,
    )


def _batch_limit(loader, maximum: int) -> int:
    return min(len(loader), maximum) if maximum > 0 else len(loader)


def multi_forward_epoch(model, loader, device, optimizer, accumulation_steps: int, max_batches: int):
    training = optimizer is not None
    model.train(training)
    keys = ["total", *ARRANGEMENTS]
    sums = {key: 0.0 for key in keys}
    count = 0
    limit = _batch_limit(loader, max_batches)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        if batch_index >= limit:
            break
        rho = batch["rho"].to(device, non_blocking=True)
        predictions = model(rho)
        losses = {
            name: masked_mse(
                predictions[name],
                batch["forward"][name].to(device, non_blocking=True),
                batch["mask"][name].to(device, non_blocking=True),
            )
            for name in ARRANGEMENTS
        }
        total_loss = torch.stack(list(losses.values())).mean()
        if training:
            (total_loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == limit:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        batch_size = rho.shape[0]
        sums["total"] += float(total_loss.detach()) * batch_size
        for name, loss in losses.items():
            sums[name] += float(loss.detach()) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def multi_inverse_epoch(
    inverse_model,
    forward_model,
    loader,
    device,
    optimizer,
    accumulation_steps: int,
    adaptive_weight_eps: float,
    max_batches: int,
):
    training = optimizer is not None
    inverse_model.train(training)
    forward_model.eval()
    keys = [
        "total",
        "inv_raw",
        "fwd_raw",
        "inv_weighted",
        "fwd_weighted",
        "w_inv",
        "w_fwd",
        "g_inv",
        "g_fwd",
        *[f"fwd_{name}" for name in ARRANGEMENTS],
    ]
    sums = {key: 0.0 for key in keys}
    count = 0
    limit = _batch_limit(loader, max_batches)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        if batch_index >= limit:
            break
        inverse_inputs = {
            name: batch["inverse"][name].to(device, non_blocking=True)
            for name in ARRANGEMENTS
        }
        rho_true = batch["rho"].to(device, non_blocking=True)
        with torch.enable_grad():
            rho_prediction = inverse_model(inverse_inputs)
            forward_predictions = forward_model(rho_prediction)
            inv_raw = nn.functional.mse_loss(rho_prediction, rho_true)
            forward_losses = {
                name: masked_mse(
                    forward_predictions[name],
                    batch["forward"][name].to(device, non_blocking=True),
                    batch["mask"][name].to(device, non_blocking=True),
                )
                for name in ARRANGEMENTS
            }
            fwd_raw = torch.stack(list(forward_losses.values())).mean()
            inverse_weight, forward_weight, inverse_grad_norm, forward_grad_norm = adaptive_loss_weights(
                inv_raw,
                fwd_raw,
                inverse_model.parameters(),
                adaptive_weight_eps,
            )
            inv_weighted = inverse_weight * inv_raw
            fwd_weighted = forward_weight * fwd_raw
            total_loss = inv_weighted + fwd_weighted
        if training:
            (total_loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == limit:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        batch_size = rho_true.shape[0]
        values = {
            "total": total_loss,
            "inv_raw": inv_raw,
            "fwd_raw": fwd_raw,
            "inv_weighted": inv_weighted,
            "fwd_weighted": fwd_weighted,
            "w_inv": inverse_weight,
            "w_fwd": forward_weight,
            "g_inv": inverse_grad_norm,
            "g_fwd": forward_grad_norm,
            **{f"fwd_{name}": value for name, value in forward_losses.items()},
        }
        for key, value in values.items():
            sums[key] += float(value.detach()) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_forward(args):
    stage_dir, config, logger, device = prepare(args)
    root = Path(args.data_root)
    train_set = MultiArrangementDataset(root, "train", include_inverse=False)
    val_set = MultiArrangementDataset(root, "val", include_inverse=False)
    train_loader = build_loader(train_set, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_set, args.batch_size, False, args.num_workers, args.seed + 1)
    model = build_multi_forward_lnn_unet().to(device)
    model.initialize_output_bias(args.output_bias)
    logger.info("Samples train=%d val=%d parameters=%d", len(train_set), len(val_set), sum(p.numel() for p in model.parameters()))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    best_path = stage_dir / "forward_multi_lnn_best.pth"
    last_path = stage_dir / "forward_multi_lnn_last.pth"
    start_epoch, best_val, no_improvement = 1, float("inf"), 0
    if args.resume and last_path.exists():
        start_epoch, best_val, no_improvement = load_checkpoint(last_path, model, optimizer, scheduler, device)
        logger.info("Resumed from %s at epoch %d", last_path, start_epoch)
    fields = ["epoch", "train_total", *[f"train_{name}" for name in ARRANGEMENTS], "val_total", *[f"val_{name}" for name in ARRANGEMENTS], "lr", "epoch_time_sec"]
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = multi_forward_epoch(model, train_loader, device, optimizer, args.accumulation_steps, args.max_train_batches)
        with torch.no_grad():
            val_metrics = multi_forward_epoch(model, val_loader, device, None, 1, args.max_val_batches)
        scheduler.step(val_metrics["total"])
        elapsed = time.perf_counter() - started
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %03d/%03d | Train total=%.8f WA=%.8f WB=%.8f SLM=%.8f | Val total=%.8f WA=%.8f WB=%.8f SLM=%.8f | LR=%.3e time=%.1fs",
            epoch, args.epochs, train_metrics["total"], train_metrics["wa"], train_metrics["wb"], train_metrics["slm"],
            val_metrics["total"], val_metrics["wa"], val_metrics["wb"], val_metrics["slm"], lr, elapsed,
        )
        row = {"epoch": epoch, "train_total": train_metrics["total"], "val_total": val_metrics["total"], "lr": lr, "epoch_time_sec": elapsed}
        for name in ARRANGEMENTS:
            row[f"train_{name}"] = train_metrics[name]
            row[f"val_{name}"] = val_metrics[name]
        append_csv(stage_dir / "losses.csv", fields, row)
        improved = val_metrics["total"] < best_val
        best_val, no_improvement = (val_metrics["total"], 0) if improved else (best_val, no_improvement + 1)
        save_checkpoint(last_path, args.stage, epoch, model, optimizer, scheduler, best_val, no_improvement, config)
        if improved:
            save_checkpoint(best_path, args.stage, epoch, model, optimizer, scheduler, best_val, no_improvement, config)
            logger.info("Saved best multi-forward model: val_total=%.8f", best_val)
        if no_improvement >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs without improvement", no_improvement)
            break


def train_inverse(args):
    stage_dir, config, logger, device = prepare(args)
    root = Path(args.data_root)
    train_set = MultiArrangementDataset(root, "train", include_inverse=True)
    val_set = MultiArrangementDataset(root, "val", include_inverse=True)
    train_loader = build_loader(train_set, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_set, args.batch_size, False, args.num_workers, args.seed + 1)
    forward_model = build_multi_forward_lnn_unet().to(device)
    forward_payload = torch.load(args.forward_checkpoint, map_location=device)
    forward_model.load_state_dict(forward_payload["model_state_dict"])
    forward_model.eval()
    for parameter in forward_model.parameters():
        parameter.requires_grad_(False)
    inverse_model = build_multi_inverse_lnn_unet().to(device)
    inverse_model.initialize_output_bias(args.output_bias)
    logger.info("Loaded and froze multi-forward model from %s", args.forward_checkpoint)
    logger.info("Samples train=%d val=%d inverse_parameters=%d", len(train_set), len(val_set), sum(p.numel() for p in inverse_model.parameters()))
    optimizer = optim.AdamW(inverse_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    best_path = stage_dir / "inverse_multi_closed_loop_best.pth"
    last_path = stage_dir / "inverse_multi_closed_loop_last.pth"
    start_epoch, best_val, no_improvement = 1, float("inf"), 0
    if args.resume and last_path.exists():
        start_epoch, best_val, no_improvement = load_checkpoint(last_path, inverse_model, optimizer, scheduler, device)
        logger.info("Resumed from %s at epoch %d", last_path, start_epoch)
    metric_keys = ["total", "inv_raw", "fwd_raw", "inv_weighted", "fwd_weighted", *[f"fwd_{name}" for name in ARRANGEMENTS]]
    fields = ["epoch", *[f"train_{key}" for key in metric_keys], *[f"val_{key}" for key in metric_keys], "lr", "epoch_time_sec"]
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = multi_inverse_epoch(
            inverse_model, forward_model, train_loader, device, optimizer, args.accumulation_steps,
            args.adaptive_weight_eps, args.max_train_batches,
        )
        val_metrics = multi_inverse_epoch(
            inverse_model, forward_model, val_loader, device, None, 1,
            args.adaptive_weight_eps, args.max_val_batches,
        )
        scheduler.step(val_metrics["total"])
        elapsed = time.perf_counter() - started
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %03d/%03d | Train total=%.8f Linv=%.8f (w=%.4f weighted=%.8f g=%.4e) Lfwd=%.8f (w=%.4f weighted=%.8f g=%.4e; WA=%.8f WB=%.8f SLM=%.8f) | Val total=%.8f Linv=%.8f (w=%.4f weighted=%.8f g=%.4e) Lfwd=%.8f (w=%.4f weighted=%.8f g=%.4e; WA=%.8f WB=%.8f SLM=%.8f) | LR=%.3e time=%.1fs",
            epoch, args.epochs,
            train_metrics["total"], train_metrics["inv_raw"], train_metrics["w_inv"], train_metrics["inv_weighted"], train_metrics["g_inv"],
            train_metrics["fwd_raw"], train_metrics["w_fwd"], train_metrics["fwd_weighted"], train_metrics["g_fwd"], train_metrics["fwd_wa"], train_metrics["fwd_wb"], train_metrics["fwd_slm"],
            val_metrics["total"], val_metrics["inv_raw"], val_metrics["w_inv"], val_metrics["inv_weighted"], val_metrics["g_inv"],
            val_metrics["fwd_raw"], val_metrics["w_fwd"], val_metrics["fwd_weighted"], val_metrics["g_fwd"], val_metrics["fwd_wa"], val_metrics["fwd_wb"], val_metrics["fwd_slm"], lr, elapsed,
        )
        row = {"epoch": epoch, "lr": lr, "epoch_time_sec": elapsed}
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key in metric_keys:
                row[f"{prefix}_{key}"] = metrics[key]
        append_csv(stage_dir / "losses.csv", fields, row)
        improved = val_metrics["total"] < best_val
        best_val, no_improvement = (val_metrics["total"], 0) if improved else (best_val, no_improvement + 1)
        save_checkpoint(last_path, args.stage, epoch, inverse_model, optimizer, scheduler, best_val, no_improvement, config)
        if improved:
            save_checkpoint(best_path, args.stage, epoch, inverse_model, optimizer, scheduler, best_val, no_improvement, config)
            logger.info("Saved best multi-inverse model: val_total=%.8f", best_val)
        if no_improvement >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs without improvement", no_improvement)
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Train WA/WB/SLM closed-loop LNN-U-Nets")
    parser.add_argument("stage", choices=("forward", "inverse"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive-weight-eps", type=float, default=1e-8)
    parser.add_argument("--output-bias", type=float, default=2.5)
    parser.add_argument("--forward-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("batch size and accumulation steps must be positive")
    if args.stage == "forward":
        train_forward(args)
    else:
        if args.forward_checkpoint is None or not args.forward_checkpoint.exists():
            raise FileNotFoundError("--forward-checkpoint is required for inverse training")
        train_inverse(args)


if __name__ == "__main__":
    main()
