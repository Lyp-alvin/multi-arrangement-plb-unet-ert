from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from lnn_unet import build_lnn_unet
from multi_lnn_unet import build_multi_inverse_lnn_unet
from plain_unet import build_plain_unet


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "checkpoints"


def load_array(path: Path, key: str) -> np.ndarray:
    data = sio.loadmat(path)
    if key not in data:
        raise KeyError(f"Missing key {key!r} in {path}")
    array = np.asarray(data[key], dtype=np.float32).squeeze()
    if array.shape != (256, 1024) or not np.isfinite(array).all():
        raise ValueError(f"Invalid {key} array in {path}: {array.shape}")
    return array


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def infer_single(model: torch.nn.Module, array: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(tensor)
    output = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
    if output.shape != (256, 1024) or not np.isfinite(output).all():
        raise ValueError(f"Invalid model output: {output.shape}")
    return output


def save_prediction(name: str, prediction: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.mat"
    sio.savemat(path, {"rho_pred": prediction}, do_compression=True)
    print(
        f"{name}: shape={prediction.shape} "
        f"range=[{prediction.min():.6f}, {prediction.max():.6f}] -> {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="1")
    args = parser.parse_args()
    field_dir = ROOT / "shice" / args.case_id
    input_dir = field_dir / "processed" / "network_inputs"
    output_dir = field_dir / "results" / "predictions_mat"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing field inputs: {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"case={args.case_id} device={device}")
    wa = load_array(input_dir / "wa.mat", "WA")
    wb = load_array(input_dir / "wb.mat", "WB")
    slm = load_array(input_dir / "slm.mat", "SLM")

    plain_experiments = (
        (
            "unet_wa_open_loop",
            RUNS / "unet_wa_open_loop" / "open_inverse" / "inverse_open_loop_best.pth",
        ),
        (
            "unet_wa_closed_loop",
            RUNS / "unet_wa_closed_loop" / "closed_inverse" / "inverse_closed_loop_best.pth",
        ),
    )
    for name, checkpoint in plain_experiments:
        model, _ = load_checkpoint(build_plain_unet(), checkpoint, device)
        save_prediction(name, infer_single(model, wa, device), output_dir)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    single_checkpoint = (
        RUNS
        / "lnn_wa_single_closed_loop"
        / "inverse"
        / "inverse_closed_loop_best.pth"
    )
    single_payload = torch.load(single_checkpoint, map_location="cpu")
    variant = single_payload.get("config", {}).get("variant", "base")
    single_model = build_lnn_unet(variant, input_channels=1, output_channels=1)
    single_model, _ = load_checkpoint(single_model, single_checkpoint, device)
    save_prediction(
        "lnn_wa_single_closed_loop", infer_single(single_model, wa, device),
        output_dir,
    )
    del single_model, single_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()

    multi_checkpoint = (
        RUNS
        / "lnn_multi_closed_loop"
        / "inverse"
        / "inverse_multi_closed_loop_best.pth"
    )
    multi_model, _ = load_checkpoint(
        build_multi_inverse_lnn_unet(), multi_checkpoint, device
    )
    inputs = {
        "wa": torch.from_numpy(wa).unsqueeze(0).unsqueeze(0).to(device),
        "wb": torch.from_numpy(wb).unsqueeze(0).unsqueeze(0).to(device),
        "slm": torch.from_numpy(slm).unsqueeze(0).unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        multi_prediction = multi_model(inputs)[0, 0].detach().cpu().numpy()
    multi_prediction = np.asarray(multi_prediction, dtype=np.float32)
    if multi_prediction.shape != (256, 1024) or not np.isfinite(multi_prediction).all():
        raise ValueError(f"Invalid multi-model output: {multi_prediction.shape}")
    save_prediction("lnn_multi_closed_loop", multi_prediction, output_dir)


if __name__ == "__main__":
    main()
