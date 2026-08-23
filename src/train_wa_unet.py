from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn, optim

from plain_unet import build_plain_unet
from train_wa_closed_loop_lnn import (
    WASingleArrayDataset,
    append_csv,
    build_loader,
    create_logger,
    masked_mse,
    set_seed,
)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_WORK_DIR = Path(__file__).resolve().parent / "runs" / "unet_wa"


@dataclass(frozen=True)
class Config:
    stage: str
    data_root: str
    work_dir: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    scheduler_patience: int
    scheduler_factor: float
    scheduler_threshold: float
    early_stopping_patience: int
    num_workers: int
    seed: int
    inverse_weight: float
    forward_weight: float
    use_amp: bool = False


def save_checkpoint(
    path: Path,
    stage: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    best_val_loss: float,
    epochs_without_improvement: int,
    config: Config,
) -> None:
    torch.save(
        {
            "stage": stage,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float, int]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return (
        int(payload["epoch"]) + 1,
        float(payload.get("best_val_loss", float("inf"))),
        int(payload.get("epochs_without_improvement", 0)),
    )


def make_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        threshold=args.scheduler_threshold,
    )


def prepare(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_dir = Path(args.work_dir) / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        stage=args.stage,
        data_root=str(Path(args.data_root).resolve()),
        work_dir=str(Path(args.work_dir).resolve()),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_threshold=args.scheduler_threshold,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        inverse_weight=args.inverse_weight,
        forward_weight=args.forward_weight,
    )
    (stage_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    logger = create_logger(stage_dir / "train.log")
    logger.info("Stage=%s device=%s config=%s", args.stage, device, asdict(config))
    return stage_dir, config, logger, device


def open_epoch(model, loader, device, optimizer, max_batches: int) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        input_image = batch["wa_inv"].to(device, non_blocking=True)
        target = batch["rho"].to(device, non_blocking=True)
        prediction = model(input_image)
        loss = nn.functional.mse_loss(prediction, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = input_image.shape[0]
        total += float(loss.detach()) * batch_size
        count += batch_size
    return total / max(count, 1)


def forward_epoch(model, loader, device, optimizer, max_batches: int) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        rho = batch["rho"].to(device, non_blocking=True)
        target = batch["wa_img"].to(device, non_blocking=True)
        mask = batch["wa_mask"].to(device, non_blocking=True)
        prediction = model(rho)
        loss = masked_mse(prediction, target, mask)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = rho.shape[0]
        total += float(loss.detach()) * batch_size
        count += batch_size
    return total / max(count, 1)


def closed_epoch(
    inverse_model,
    forward_model,
    loader,
    device,
    optimizer,
    inverse_weight: float,
    forward_weight: float,
    max_batches: int,
) -> dict[str, float]:
    training = optimizer is not None
    inverse_model.train(training)
    forward_model.eval()
    sums = {"total": 0.0, "inv_raw": 0.0, "fwd_raw": 0.0, "inv_weighted": 0.0, "fwd_weighted": 0.0}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        input_image = batch["wa_inv"].to(device, non_blocking=True)
        rho_true = batch["rho"].to(device, non_blocking=True)
        forward_true = batch["wa_img"].to(device, non_blocking=True)
        mask = batch["wa_mask"].to(device, non_blocking=True)
        rho_prediction = inverse_model(input_image)
        forward_prediction = forward_model(rho_prediction)
        inv_raw = nn.functional.mse_loss(rho_prediction, rho_true)
        fwd_raw = masked_mse(forward_prediction, forward_true, mask)
        inv_weighted = inverse_weight * inv_raw
        fwd_weighted = forward_weight * fwd_raw
        total_loss = inv_weighted + fwd_weighted
        if training:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
        batch_size = input_image.shape[0]
        values = {
            "total": total_loss,
            "inv_raw": inv_raw,
            "fwd_raw": fwd_raw,
            "inv_weighted": inv_weighted,
            "fwd_weighted": fwd_weighted,
        }
        for key, value in values.items():
            sums[key] += float(value.detach()) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def fit_scalar_stage(args, epoch_function, fields, checkpoint_prefix, train_set, val_set) -> None:
    stage_dir, config, logger, device = prepare(args)
    train_loader = build_loader(train_set, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_set, args.batch_size, False, args.num_workers, args.seed + 1)
    model = build_plain_unet().to(device)
    model.initialize_output_bias(args.output_bias)
    logger.info("Samples train=%d val=%d parameters=%d", len(train_set), len(val_set), sum(p.numel() for p in model.parameters()))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    best_path = stage_dir / f"{checkpoint_prefix}_best.pth"
    last_path = stage_dir / f"{checkpoint_prefix}_last.pth"
    start_epoch, best_val, no_improvement = 1, float("inf"), 0
    if args.resume and last_path.exists():
        start_epoch, best_val, no_improvement = load_checkpoint(last_path, model, optimizer, scheduler, device)
        logger.info("Resumed from %s at epoch %d", last_path, start_epoch)
    csv_path = stage_dir / "losses.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_loss = epoch_function(model, train_loader, device, optimizer, args.max_train_batches)
        with torch.no_grad():
            val_loss = epoch_function(model, val_loader, device, None, args.max_val_batches)
        scheduler.step(val_loss)
        elapsed = time.perf_counter() - started
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %03d/%03d | Train=%.8f Val=%.8f LR=%.3e time=%.1fs", epoch, args.epochs, train_loss, val_loss, lr, elapsed)
        append_csv(csv_path, fields, {fields[0]: epoch, fields[1]: train_loss, fields[2]: val_loss, "lr": lr, "epoch_time_sec": elapsed})
        improved = val_loss < best_val
        if improved:
            best_val, no_improvement = val_loss, 0
        else:
            no_improvement += 1
        save_checkpoint(last_path, args.stage, epoch, model, optimizer, scheduler, best_val, no_improvement, config)
        if improved:
            save_checkpoint(best_path, args.stage, epoch, model, optimizer, scheduler, best_val, no_improvement, config)
            logger.info("Saved best model: val=%.8f", best_val)
        if no_improvement >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs without improvement", no_improvement)
            break


def train_open(args) -> None:
    root = Path(args.data_root)
    fit_scalar_stage(
        args,
        open_epoch,
        ["epoch", "train_Linv_raw", "val_Linv_raw", "lr", "epoch_time_sec"],
        "inverse_open_loop",
        WASingleArrayDataset(root, "train", "inverse"),
        WASingleArrayDataset(root, "val", "inverse"),
    )


def train_forward(args) -> None:
    root = Path(args.data_root)
    fit_scalar_stage(
        args,
        forward_epoch,
        ["epoch", "train_Lfwd_raw", "val_Lfwd_raw", "lr", "epoch_time_sec"],
        "forward_unet",
        WASingleArrayDataset(root, "train", "forward"),
        WASingleArrayDataset(root, "val", "forward"),
    )


def train_closed(args) -> None:
    stage_dir, config, logger, device = prepare(args)
    root = Path(args.data_root)
    train_set = WASingleArrayDataset(root, "train", "inverse")
    val_set = WASingleArrayDataset(root, "val", "inverse")
    train_loader = build_loader(train_set, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_set, args.batch_size, False, args.num_workers, args.seed + 1)
    forward_model = build_plain_unet().to(device)
    forward_payload = torch.load(args.forward_checkpoint, map_location=device)
    forward_model.load_state_dict(forward_payload["model_state_dict"])
    forward_model.eval()
    for parameter in forward_model.parameters():
        parameter.requires_grad_(False)
    inverse_model = build_plain_unet().to(device)
    inverse_model.initialize_output_bias(args.output_bias)
    optimizer = optim.AdamW(inverse_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    logger.info("Loaded and froze forward model from %s", args.forward_checkpoint)
    logger.info("Samples train=%d val=%d inverse_parameters=%d", len(train_set), len(val_set), sum(p.numel() for p in inverse_model.parameters()))
    best_path = stage_dir / "inverse_closed_loop_best.pth"
    last_path = stage_dir / "inverse_closed_loop_last.pth"
    start_epoch, best_val, no_improvement = 1, float("inf"), 0
    if args.resume and last_path.exists():
        start_epoch, best_val, no_improvement = load_checkpoint(last_path, inverse_model, optimizer, scheduler, device)
        logger.info("Resumed from %s at epoch %d", last_path, start_epoch)
    fields = [
        "epoch", "train_loss_total", "train_Linv_raw", "train_Lfwd_raw", "train_Linv_weighted", "train_Lfwd_weighted",
        "val_loss_total", "val_Linv_raw", "val_Lfwd_raw", "val_Linv_weighted", "val_Lfwd_weighted", "lr", "epoch_time_sec",
    ]
    csv_path = stage_dir / "losses.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = closed_epoch(inverse_model, forward_model, train_loader, device, optimizer, args.inverse_weight, args.forward_weight, args.max_train_batches)
        with torch.no_grad():
            val_metrics = closed_epoch(inverse_model, forward_model, val_loader, device, None, args.inverse_weight, args.forward_weight, args.max_val_batches)
        scheduler.step(val_metrics["total"])
        elapsed = time.perf_counter() - started
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %03d/%03d | Train total=%.8f Linv=%.8f (weighted=%.8f) Lfwd=%.8f (weighted=%.8f) | Val total=%.8f Linv=%.8f (weighted=%.8f) Lfwd=%.8f (weighted=%.8f) | LR=%.3e time=%.1fs",
            epoch, args.epochs, train_metrics["total"], train_metrics["inv_raw"], train_metrics["inv_weighted"], train_metrics["fwd_raw"], train_metrics["fwd_weighted"],
            val_metrics["total"], val_metrics["inv_raw"], val_metrics["inv_weighted"], val_metrics["fwd_raw"], val_metrics["fwd_weighted"], lr, elapsed,
        )
        row = {"epoch": epoch, "lr": lr, "epoch_time_sec": elapsed}
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            row.update({
                f"{prefix}_loss_total": metrics["total"], f"{prefix}_Linv_raw": metrics["inv_raw"], f"{prefix}_Lfwd_raw": metrics["fwd_raw"],
                f"{prefix}_Linv_weighted": metrics["inv_weighted"], f"{prefix}_Lfwd_weighted": metrics["fwd_weighted"],
            })
        append_csv(csv_path, fields, row)
        improved = val_metrics["total"] < best_val
        if improved:
            best_val, no_improvement = val_metrics["total"], 0
        else:
            no_improvement += 1
        save_checkpoint(last_path, args.stage, epoch, inverse_model, optimizer, scheduler, best_val, no_improvement, config)
        if improved:
            save_checkpoint(best_path, args.stage, epoch, inverse_model, optimizer, scheduler, best_val, no_improvement, config)
            logger.info("Saved best inverse model: val_total=%.8f", best_val)
        if no_improvement >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs without improvement", no_improvement)
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train plain U-Net WA baselines")
    parser.add_argument("stage", choices=("open_inverse", "forward", "closed_inverse"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inverse-weight", type=float, default=0.8)
    parser.add_argument("--forward-weight", type=float, default=0.2)
    parser.add_argument("--output-bias", type=float, default=2.5)
    parser.add_argument("--forward-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "open_inverse":
        train_open(args)
    elif args.stage == "forward":
        train_forward(args)
    else:
        if args.forward_checkpoint is None or not args.forward_checkpoint.exists():
            raise FileNotFoundError("--forward-checkpoint is required for closed_inverse")
        train_closed(args)


if __name__ == "__main__":
    main()
