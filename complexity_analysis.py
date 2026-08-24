from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lnn_unet import build_lnn_unet
from multi_lnn_unet import build_multi_inverse_lnn_unet
from plain_unet import build_plain_unet


@dataclass(frozen=True)
class MethodSpec:
    name: str
    builder: str
    checkpoint: Path
    input_description: str


METHODS = (
    MethodSpec(
        name="SA-U-Net-OL",
        builder="plain_single",
        checkpoint=ROOT / "checkpoints" / "sa_unet_ol" / "open_inverse" / "inverse_open_loop_best.pth",
        input_description="single WA tensor: (1, 1, {height}, {width})",
    ),
    MethodSpec(
        name="SA-U-Net-CL",
        builder="plain_single",
        checkpoint=ROOT / "checkpoints" / "sa_unet_cl" / "closed_inverse" / "inverse_closed_loop_best.pth",
        input_description="single WA tensor: (1, 1, {height}, {width})",
    ),
    MethodSpec(
        name="SA-LIB-U-Net-CL",
        builder="lib_single",
        checkpoint=ROOT / "checkpoints" / "sa_plb_unet_cl" / "inverse" / "inverse_lnn_unet_best.pth",
        input_description="single WA tensor: (1, 1, {height}, {width})",
    ),
    MethodSpec(
        name="MA-LIB-U-Net-CL",
        builder="lib_multi",
        checkpoint=ROOT / "checkpoints" / "ma_plb_unet_cl" / "inverse" / "inverse_multi_closed_loop_best.pth",
        input_description="three tensors: WA/WB/SLM, each (1, 1, {height}, {width})",
    ),
)


def build_model(builder: str) -> torch.nn.Module:
    if builder == "plain_single":
        return build_plain_unet(input_channels=1, output_channels=1)
    if builder == "lib_single":
        return build_lnn_unet("base", input_channels=1, output_channels=1)
    if builder == "lib_multi":
        return build_multi_inverse_lnn_unet()
    raise ValueError(f"Unknown builder: {builder}")


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> int | None:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    if isinstance(payload, dict) and "epoch" in payload:
        return int(payload["epoch"])
    return None


def make_inputs(builder: str, batch_size: int, height: int, width: int, device: torch.device) -> Any:
    shape = (batch_size, 1, height, width)
    if builder == "lib_multi":
        return {
            "wa": torch.randn(shape, device=device),
            "wb": torch.randn(shape, device=device),
            "slm": torch.randn(shape, device=device),
        }
    return torch.randn(shape, device=device)


def forward_once(model: torch.nn.Module, inputs: Any) -> torch.Tensor:
    output = model(inputs)
    if isinstance(output, dict):
        output = tuple(output.values())[0]
    return output


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def profile_flops(model: torch.nn.Module, inputs: Any, device: torch.device) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            _ = forward_once(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
    return int(sum(event.flops for event in profiler.key_averages() if event.flops is not None))


def measure_latency_ms(
    model: torch.nn.Module,
    inputs: Any,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = forward_once(model, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            _ = forward_once(model, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / float(repeats)


def measure_peak_memory_gb(model: torch.nn.Module, inputs: Any, device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _ = forward_once(model, inputs)
        torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)


def write_results(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "complexity_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Method",
                "Input shape",
                "Checkpoint epoch",
                "Params (M)",
                "Trainable params (M)",
                "FLOPs (G)",
                "Latency (ms/sample)",
                "Peak GPU Memory (GB)",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    txt_path = output_dir / "complexity_analysis.txt"
    widths = {
        "Method": 18,
        "Params (M)": 12,
        "FLOPs (G)": 10,
        "Latency (ms/sample)": 20,
        "Peak GPU Memory (GB)": 20,
    }
    lines = [
        "Computational complexity and inference efficiency",
        "",
        json.dumps(metadata, indent=2),
        "",
        f"{'Method':<{widths['Method']}} | {'Params (M)':>{widths['Params (M)']}} | "
        f"{'FLOPs (G)':>{widths['FLOPs (G)']}} | {'Latency (ms/sample)':>{widths['Latency (ms/sample)']}} | "
        f"{'Peak GPU Memory (GB)':>{widths['Peak GPU Memory (GB)']}}",
        "-" * 92,
    ]
    for row in rows:
        lines.append(
            f"{row['Method']:<{widths['Method']}} | {row['Params (M)']:>{widths['Params (M)']}} | "
            f"{row['FLOPs (G)']:>{widths['FLOPs (G)']}} | "
            f"{row['Latency (ms/sample)']:>{widths['Latency (ms/sample)']}} | "
            f"{row['Peak GPU Memory (GB)']:>{widths['Peak GPU Memory (GB)']}}"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Closed-loop forward surrogate networks are used only during training and are not included in inference complexity.",
            "- FLOPs are measured with torch.profiler(with_flops=True) on the actual model forward path.",
            "- PyTorch profiler reports FLOPs for convolution/matrix-multiplication style operators; element-wise nonlinearities, interpolation, and some normalization operations may be omitted.",
            "- Convolution FLOPs are reported as floating-point operations, not raw MAC counts.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure inversion-network parameters, FLOPs, latency, and peak GPU memory."
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.backends.cudnn.benchmark = True
    rows: list[dict[str, Any]] = []
    metadata = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "batch_size": args.batch_size,
        "input_height": args.height,
        "input_width": args.width,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.repeats,
        "inference_scope": "inversion network only; closed-loop forward surrogate excluded",
        "flops_definition": "torch.profiler with_flops=True; convolution FLOPs are floating-point operations rather than unreported MACs",
    }

    print("Inference complexity scope: inversion network only.")
    print("Closed-loop forward surrogate is excluded because deployment prediction uses only G_phi -> rho_hat.")
    print(f"Device: {metadata['device']}")

    for spec in METHODS:
        print(f"\nMeasuring {spec.name} ...", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model = build_model(spec.builder).to(device)
        epoch = None
        if not args.skip_checkpoints:
            epoch = load_checkpoint(model, spec.checkpoint, device)
        model.eval()
        inputs = make_inputs(spec.builder, args.batch_size, args.height, args.width, device)

        with torch.inference_mode():
            output = forward_once(model, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if tuple(output.shape) != (args.batch_size, 1, args.height, args.width):
            raise RuntimeError(f"{spec.name} returned unexpected shape {tuple(output.shape)}")

        total_params, trainable_params = count_parameters(model)
        flops = profile_flops(model, inputs, device)
        latency = measure_latency_ms(model, inputs, device, args.warmup, args.repeats)
        peak_memory = measure_peak_memory_gb(model, inputs, device)

        row = {
            "Method": spec.name,
            "Input shape": spec.input_description.format(height=args.height, width=args.width),
            "Checkpoint epoch": "" if epoch is None else epoch,
            "Params (M)": f"{total_params / 1e6:.3f}",
            "Trainable params (M)": f"{trainable_params / 1e6:.3f}",
            "FLOPs (G)": f"{flops / 1e9:.3f}",
            "Latency (ms/sample)": f"{latency:.3f}",
            "Peak GPU Memory (GB)": "N/A" if peak_memory is None else f"{peak_memory:.3f}",
        }
        rows.append(row)
        print(row, flush=True)

        del model, inputs, output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_results(args.output_dir, rows, metadata)
    print(f"\nSaved: {args.output_dir / 'complexity_analysis.csv'}")
    print(f"Saved: {args.output_dir / 'complexity_analysis.txt'}")


if __name__ == "__main__":
    main()
