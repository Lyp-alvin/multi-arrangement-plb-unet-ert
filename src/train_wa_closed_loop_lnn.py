from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from lnn_unet import build_lnn_unet

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_DIR / "data"
DEFAULT_WORK_DIR = Path(__file__).resolve().parent / "runs" / "wa_single_closed_loop"
IMAGE_SHAPE = (256, 1024)


@dataclass(frozen=True)
class TrainConfig:
    stage: str
    data_root: str
    work_dir: str
    variant: str
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
    adaptive_weight_eps: float
    use_amp: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_ids(path: Path) -> list[int]:
    ids = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def load_mat_array(path: Path, key: str) -> np.ndarray:
    payload = sio.loadmat(path)
    if key not in payload:
        raise KeyError(f"Missing key {key!r} in {path}")
    array = np.asarray(payload[key], dtype=np.float32).squeeze()
    if array.shape != IMAGE_SHAPE:
        raise ValueError(f"Expected {IMAGE_SHAPE} in {path}, found {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"NaN or Inf found in {path}")
    return array


class WASingleArrayDataset(Dataset):
    def __init__(self, data_root: Path, split: str, stage: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        if stage not in {"forward", "inverse"}:
            raise ValueError(f"Unsupported stage: {stage}")

        self.data_root = data_root
        self.split = split
        self.stage = stage
        self.ids = read_ids(data_root / f"{split}_ids.txt")
        self.rho_dir = data_root / "rho"
        self.wa_forward_dir = data_root / "wa_256_layered"
        self.wa_inverse_dir = data_root / "inv_input_wa"
        self._validate_files()

    def _validate_files(self) -> None:
        missing: list[str] = []
        for file_id in self.ids:
            required = [
                self.rho_dir / f"rho_{file_id}.mat",
                self.wa_forward_dir / f"rhoa_2d_{file_id}.mat",
            ]
            if self.stage == "inverse":
                required.append(self.wa_inverse_dir / f"wainv_{file_id}.mat")
            for path in required:
                if not path.exists():
                    missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} required files for {self.split}/{self.stage}: "
                f"{missing[:10]}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        file_id = self.ids[index]
        rho = load_mat_array(self.rho_dir / f"rho_{file_id}.mat", "rho")
        forward_payload = sio.loadmat(self.wa_forward_dir / f"rhoa_2d_{file_id}.mat")
        if "img" not in forward_payload or "mask" not in forward_payload:
            raise KeyError(f"Missing img/mask for WA forward ID {file_id}")
        wa_img = np.asarray(forward_payload["img"], dtype=np.float32).squeeze()
        wa_mask = np.asarray(forward_payload["mask"], dtype=np.float32).squeeze()
        if wa_img.shape != IMAGE_SHAPE or wa_mask.shape != IMAGE_SHAPE:
            raise ValueError(f"Invalid WA forward shape for ID {file_id}")
        if not np.isfinite(wa_img).all() or not np.isfinite(wa_mask).all():
            raise ValueError(f"Invalid WA forward values for ID {file_id}")
        if wa_mask.sum() <= 0:
            raise ValueError(f"Empty WA mask for ID {file_id}")

        sample: dict[str, torch.Tensor | int] = {
            "id": file_id,
            "rho": torch.from_numpy(rho[None, ...]),
            "wa_img": torch.from_numpy(wa_img[None, ...]),
            "wa_mask": torch.from_numpy(wa_mask[None, ...]),
        }
        if self.stage == "inverse":
            wa_inv = load_mat_array(self.wa_inverse_dir / f"wainv_{file_id}.mat", "WA")
            sample["wa_inv"] = torch.from_numpy(wa_inv[None, ...])
        return sample


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (prediction - target).square() * mask
    numerator = squared_error.flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).mean()


def _gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    squared_norm = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().pow(2).sum()
    return squared_norm.sqrt()


def adaptive_loss_weights(
    inverse_loss: torch.Tensor,
    forward_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balance inverse/forward losses by inverse gradient norms.

    The returned weights are detached scalars, so optimization does not involve
    second-order derivatives through the weighting rule.
    """
    trainable_parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters available for adaptive closed-loop weighting.")

    inverse_grad_norm = _gradient_norm(inverse_loss, trainable_parameters)
    forward_grad_norm = _gradient_norm(forward_loss, trainable_parameters)
    inverse_score = torch.reciprocal(inverse_grad_norm + float(eps))
    forward_score = torch.reciprocal(forward_grad_norm + float(eps))
    normalizer = inverse_score + forward_score
    inverse_weight = (inverse_score / normalizer).detach()
    forward_weight = (forward_score / normalizer).detach()
    return inverse_weight, forward_weight, inverse_grad_norm.detach(), forward_grad_norm.detach()


def create_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def append_csv(path: Path, fieldnames: list[str], row: dict[str, float | int]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    stage: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    best_val_loss: float,
    epochs_without_improvement: int,
    config: TrainConfig,
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
            "seed": config.seed,
            "adaptive_weight_eps": config.adaptive_weight_eps,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_val_loss", float("inf"))),
        int(checkpoint.get("epochs_without_improvement", 0)),
    )


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def batch_iterator(loader: DataLoader, description: str, enabled: bool) -> Iterable:
    if enabled and tqdm is not None:
        return tqdm(loader, total=len(loader), desc=description, dynamic_ncols=True, leave=False)
    return loader


def forward_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    description: str,
    show_progress: bool,
    max_batches: int,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    iterator = batch_iterator(loader, description, show_progress)

    for batch_index, batch in enumerate(iterator):
        if max_batches > 0 and batch_index >= max_batches:
            break
        rho = batch["rho"].to(device, non_blocking=True)
        wa_img = batch["wa_img"].to(device, non_blocking=True)
        wa_mask = batch["wa_mask"].to(device, non_blocking=True)
        prediction = model(rho)
        loss = masked_mse(prediction, wa_img, wa_mask)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = rho.shape[0]
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(Lfwd=f"{loss_sum / sample_count:.6f}")

    return loss_sum / max(sample_count, 1)


def inverse_epoch(
    inverse_model: nn.Module,
    forward_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    adaptive_weight_eps: float,
    description: str,
    show_progress: bool,
    max_batches: int,
) -> dict[str, float]:
    training = optimizer is not None
    inverse_model.train(training)
    forward_model.eval()
    sums = {
        "inv_raw": 0.0,
        "fwd_raw": 0.0,
        "inv_weighted": 0.0,
        "fwd_weighted": 0.0,
        "w_inv": 0.0,
        "w_fwd": 0.0,
        "g_inv": 0.0,
        "g_fwd": 0.0,
        "total": 0.0,
    }
    sample_count = 0
    iterator = batch_iterator(loader, description, show_progress)

    for batch_index, batch in enumerate(iterator):
        if max_batches > 0 and batch_index >= max_batches:
            break
        wa_inv = batch["wa_inv"].to(device, non_blocking=True)
        rho_true = batch["rho"].to(device, non_blocking=True)

        if training:
            wa_true = batch["wa_img"].to(device, non_blocking=True)
            wa_mask = batch["wa_mask"].to(device, non_blocking=True)
            rho_pred = inverse_model(wa_inv)
            wa_pred = forward_model(rho_pred)
            inverse_loss = nn.functional.mse_loss(rho_pred, rho_true)
            forward_loss = masked_mse(wa_pred, wa_true, wa_mask)
            inverse_weight, forward_weight, inverse_grad_norm, forward_grad_norm = adaptive_loss_weights(
                inverse_loss,
                forward_loss,
                inverse_model.parameters(),
                adaptive_weight_eps,
            )
            inverse_weighted = inverse_weight * inverse_loss
            forward_weighted = forward_weight * forward_loss
            total_loss = inverse_weighted + forward_weighted
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                rho_pred = inverse_model(wa_inv)
                inverse_loss = nn.functional.mse_loss(rho_pred, rho_true)
            nan = torch.tensor(float("nan"), device=device)
            forward_loss = nan
            inverse_weight = nan
            forward_weight = nan
            inverse_grad_norm = nan
            forward_grad_norm = nan
            inverse_weighted = inverse_loss
            forward_weighted = nan
            total_loss = inverse_loss

        batch_size = wa_inv.shape[0]
        values = {
            "inv_raw": inverse_loss,
            "fwd_raw": forward_loss,
            "inv_weighted": inverse_weighted,
            "fwd_weighted": forward_weighted,
            "w_inv": inverse_weight,
            "w_fwd": forward_weight,
            "g_inv": inverse_grad_norm,
            "g_fwd": forward_grad_norm,
            "total": total_loss,
        }
        for key, value in values.items():
            sums[key] += float(value.detach().item()) * batch_size
        sample_count += batch_size
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                total=f"{sums['total'] / sample_count:.6f}",
                Linv=f"{sums['inv_raw'] / sample_count:.6f}",
                Lfwd=f"{sums['fwd_raw'] / sample_count:.6f}",
                w_inv=f"{sums['w_inv'] / sample_count:.3f}",
                w_fwd=f"{sums['w_fwd'] / sample_count:.3f}",
            )

    return {key: value / max(sample_count, 1) for key, value in sums.items()}


def prepare_stage(args: argparse.Namespace) -> tuple[TrainConfig, Path, logging.Logger, torch.device]:
    work_root = Path(args.work_dir)
    stage_dir = work_root / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        stage=args.stage,
        data_root=str(Path(args.data_root).resolve()),
        work_dir=str(work_root.resolve()),
        variant=args.variant,
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
        adaptive_weight_eps=args.adaptive_weight_eps,
        use_amp=False,
    )
    (stage_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    logger = create_logger(stage_dir / "train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Stage=%s device=%s", args.stage, device)
    if device.type == "cuda":
        logger.info("GPU=%s memory=%.2f GB", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30)
    logger.info("Config=%s", asdict(config))
    return config, stage_dir, logger, device


def train_forward(args: argparse.Namespace) -> None:
    config, stage_dir, logger, device = prepare_stage(args)
    train_set = WASingleArrayDataset(Path(args.data_root), "train", "forward")
    val_set = WASingleArrayDataset(Path(args.data_root), "val", "forward")
    train_loader = build_loader(train_set, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_set, args.batch_size, False, args.num_workers, args.seed + 1)
    logger.info("Samples train=%d val=%d", len(train_set), len(val_set))

    model = build_lnn_unet(args.variant, input_channels=1, output_channels=1).to(device)
    model.initialize_output_bias(args.output_bias)
    logger.info("Parameters=%d", sum(parameter.numel() for parameter in model.parameters()))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        threshold=args.scheduler_threshold,
    )
    start_epoch = 1
    best_val = float("inf")
    no_improvement = 0
    last_path = stage_dir / "forward_lnn_unet_last.pth"
    best_path = stage_dir / "forward_lnn_unet_best.pth"
    if args.resume and last_path.exists():
        start_epoch, best_val, no_improvement = load_training_checkpoint(last_path, model, optimizer, scheduler, device)
        logger.info("Resumed from %s at epoch %d", last_path, start_epoch)

    csv_path = stage_dir / "losses.csv"
    fields = ["epoch", "train_Lfwd_raw", "val_Lfwd_raw", "lr", "epoch_time_sec"]
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = forward_epoch(model, train_loader, device, optimizer, f"Forward train {epoch}/{args.epochs}", not args.no_progress, args.max_train_batches)
        with torch.no_grad():
            val_loss = forward_epoch(model, val_loader, device, None, f"Forward val {epoch}/{args.epochs}", not args.no_progress, args.max_val_batches)
        scheduler.step(val_loss)
        elapsed = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %03d/%03d | Train Lfwd=%.8f | Val Lfwd=%.8f | LR=%.3e | time=%.1fs", epoch, args.epochs, train_loss, val_loss, lr, elapsed)
        append_csv(csv_path, fields, {"epoch": epoch, "train_Lfwd_raw": train_loss, "val_Lfwd_raw": val_loss, "lr": lr, "epoch_time_sec": elapsed})

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
        save_checkpoint(last_path, "forward", epoch, model, optimizer, scheduler, best_val, no_improvement, config)
        if improved:
            save_checkpoint(best_path, "forward", epoch, model, optimizer, scheduler, best_val, no_improvement, config)
            logger.info("Saved best forward model: val_Lfwd=%.8f", best_val)
        if no_improvement >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs without improvement", no_improvement)
            break


def train_inverse(args: argparse.Namespace) -> None:
    config, stage_dir, logger, device = prepare_stage(args)
    train_set = WASingleArrayDataset(Path(args.data_root), "train", "inverse")
    val_set = WASingleArrayDataset(Path(args.data_root), "val", "inverse")
    train_loader = build_loader(train_set, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_set, args.batch_size, False, args.num_workers, args.seed + 1)
    logger.info("Samples train=%d val=%d", len(train_set), len(val_set))

    forward_checkpoint = Path(args.forward_checkpoint)
    if not forward_checkpoint.exists():
        raise FileNotFoundError(f"Forward checkpoint not found: {forward_checkpoint}")
    forward_model = build_lnn_unet(args.variant, input_channels=1, output_channels=1).to(device)
    forward_payload = torch.load(forward_checkpoint, map_location=device)
    forward_model.load_state_dict(forward_payload["model_state_dict"])
    forward_model.eval()
    for parameter in forward_model.parameters():
        parameter.requires_grad_(False)
    logger.info("Loaded and froze forward model from %s", forward_checkpoint)

    inverse_model = build_lnn_unet(args.variant, input_channels=1, output_channels=1).to(device)
    inverse_model.initialize_output_bias(args.output_bias)
    logger.info("Inverse parameters=%d", sum(parameter.numel() for parameter in inverse_model.parameters()))
    optimizer = optim.AdamW(inverse_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        threshold=args.scheduler_threshold,
    )
    start_epoch = 1
    best_val = float("inf")
    no_improvement = 0
    last_path = stage_dir / "inverse_closed_loop_last.pth"
    best_path = stage_dir / "inverse_closed_loop_best.pth"
    if args.resume and last_path.exists():
        start_epoch, best_val, no_improvement = load_training_checkpoint(last_path, inverse_model, optimizer, scheduler, device)
        logger.info("Resumed from %s at epoch %d", last_path, start_epoch)

    csv_path = stage_dir / "losses.csv"
    fields = [
        "epoch",
        "train_loss_total",
        "train_Linv_raw",
        "train_Lfwd_raw",
        "train_Linv_weighted",
        "train_Lfwd_weighted",
        "train_w_inv",
        "train_w_fwd",
        "train_g_inv",
        "train_g_fwd",
        "val_loss_total",
        "val_Linv_raw",
        "val_Lfwd_raw",
        "val_Linv_weighted",
        "val_Lfwd_weighted",
        "val_w_inv",
        "val_w_fwd",
        "val_g_inv",
        "val_g_fwd",
        "lr",
        "epoch_time_sec",
    ]
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = inverse_epoch(
            inverse_model, forward_model, train_loader, device, optimizer,
            args.adaptive_weight_eps,
            f"Inverse train {epoch}/{args.epochs}", not args.no_progress, args.max_train_batches,
        )
        val_metrics = inverse_epoch(
            inverse_model, forward_model, val_loader, device, None,
            args.adaptive_weight_eps,
            f"Inverse val {epoch}/{args.epochs}", not args.no_progress, args.max_val_batches,
        )
        selection_loss = val_metrics["inv_raw"]
        scheduler.step(selection_loss)
        elapsed = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %03d/%03d | Train total=%.8f Linv=%.8f (w=%.4f weighted=%.8f g=%.4e) Lfwd=%.8f (w=%.4f weighted=%.8f g=%.4e) | "
            "Val total=%.8f Linv=%.8f (w=%.4f weighted=%.8f g=%.4e) Lfwd=%.8f (w=%.4f weighted=%.8f g=%.4e) | LR=%.3e | time=%.1fs",
            epoch, args.epochs,
            train_metrics["total"], train_metrics["inv_raw"], train_metrics["w_inv"], train_metrics["inv_weighted"], train_metrics["g_inv"],
            train_metrics["fwd_raw"], train_metrics["w_fwd"], train_metrics["fwd_weighted"], train_metrics["g_fwd"],
            val_metrics["total"], val_metrics["inv_raw"], val_metrics["w_inv"], val_metrics["inv_weighted"], val_metrics["g_inv"],
            val_metrics["fwd_raw"], val_metrics["w_fwd"], val_metrics["fwd_weighted"], val_metrics["g_fwd"],
            lr, elapsed,
        )
        row = {
            "epoch": epoch,
            "train_loss_total": train_metrics["total"],
            "train_Linv_raw": train_metrics["inv_raw"],
            "train_Lfwd_raw": train_metrics["fwd_raw"],
            "train_Linv_weighted": train_metrics["inv_weighted"],
            "train_Lfwd_weighted": train_metrics["fwd_weighted"],
            "train_w_inv": train_metrics["w_inv"],
            "train_w_fwd": train_metrics["w_fwd"],
            "train_g_inv": train_metrics["g_inv"],
            "train_g_fwd": train_metrics["g_fwd"],
            "val_loss_total": val_metrics["total"],
            "val_Linv_raw": val_metrics["inv_raw"],
            "val_Lfwd_raw": val_metrics["fwd_raw"],
            "val_Linv_weighted": val_metrics["inv_weighted"],
            "val_Lfwd_weighted": val_metrics["fwd_weighted"],
            "val_w_inv": val_metrics["w_inv"],
            "val_w_fwd": val_metrics["w_fwd"],
            "val_g_inv": val_metrics["g_inv"],
            "val_g_fwd": val_metrics["g_fwd"],
            "lr": lr,
            "epoch_time_sec": elapsed,
        }
        append_csv(csv_path, fields, row)

        improved = selection_loss < best_val
        if improved:
            best_val = selection_loss
            no_improvement = 0
        else:
            no_improvement += 1
        save_checkpoint(last_path, "inverse", epoch, inverse_model, optimizer, scheduler, best_val, no_improvement, config)
        if improved:
            save_checkpoint(best_path, "inverse", epoch, inverse_model, optimizer, scheduler, best_val, no_improvement, config)
            logger.info("Saved best inverse model: val_Linv=%.8f", best_val)
        if no_improvement >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs without improvement", no_improvement)
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train WA forward and closed-loop inverse LNN-U-Nets")
    parser.add_argument("stage", choices=("forward", "inverse"))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--variant", default="base", choices=("tiny", "base", "large"))
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
    parser.add_argument("--adaptive-weight-eps", type=float, default=1e-8)
    parser.add_argument("--output-bias", type=float, default=2.5)
    parser.add_argument("--forward-checkpoint", default=str(DEFAULT_WORK_DIR / "forward" / "forward_lnn_unet_best.pth"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    if args.stage == "forward":
        train_forward(args)
    else:
        train_inverse(args)


if __name__ == "__main__":
    main()
