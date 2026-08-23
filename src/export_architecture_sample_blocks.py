from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from multi_lnn_unet import ARRANGEMENTS, build_multi_forward_lnn_unet, build_multi_inverse_lnn_unet
from train_multi_closed_loop_lnn import INVERSE_INFO
from train_wa_closed_loop_lnn import load_mat_array, read_ids, set_seed


def _load_forward_payload(data_root: Path, arrangement: str, file_id: int) -> tuple[np.ndarray, np.ndarray]:
    payload = sio.loadmat(data_root / f"{arrangement}_256_layered" / f"rhoa_2d_{file_id}.mat")
    image = np.asarray(payload["img"], dtype=np.float32).squeeze()
    mask = np.asarray(payload["mask"], dtype=np.float32).squeeze()
    return image, mask


def _load_inverse_input(data_root: Path, arrangement: str, file_id: int) -> np.ndarray:
    directory, prefix, key = INVERSE_INFO[arrangement]
    return load_mat_array(data_root / directory / f"{prefix}_{file_id}.mat", key)


def _channel_mean(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        array = array.mean(axis=0)
    return np.asarray(array, dtype=np.float32)


def _split_liquid_parameter_maps(tensor: torch.Tensor) -> dict[str, np.ndarray]:
    raw_rate, raw_bias, candidate_a, candidate_b = torch.chunk(tensor.detach(), chunks=4, dim=1)
    return {
        "plb_r_raw": _channel_mean(raw_rate),
        "plb_beta": _channel_mean(raw_bias),
        "plb_candidate_a": _channel_mean(candidate_a),
        "plb_candidate_b": _channel_mean(candidate_b),
    }


def export_blocks(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    data_root = Path(args.data_root)
    run_root = Path(args.run_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_ids = read_ids(data_root / "train_ids.txt")
    if args.file_id is None:
        file_id = int(random.Random(args.seed).choice(train_ids))
    else:
        file_id = int(args.file_id)
        if file_id not in train_ids:
            raise ValueError(f"ID {file_id} is not in train_ids.txt.")
    sample_dir = output_root / f"id_{file_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    inverse_model = build_multi_inverse_lnn_unet().to(device)
    inverse_ckpt = torch.load(
        run_root / "inverse" / "inverse_multi_closed_loop_best.pth",
        map_location=device,
    )
    inverse_model.load_state_dict(inverse_ckpt["model_state_dict"])
    inverse_model.eval()

    forward_model = build_multi_forward_lnn_unet().to(device)
    forward_ckpt = torch.load(
        run_root / "forward" / "forward_multi_lnn_best.pth",
        map_location=device,
    )
    forward_model.load_state_dict(forward_ckpt["model_state_dict"])
    forward_model.eval()

    inverse_inputs_np = {name: _load_inverse_input(data_root, name, file_id) for name in ARRANGEMENTS}
    inverse_inputs = {
        name: torch.from_numpy(value[None, None]).to(device=device, dtype=torch.float32)
        for name, value in inverse_inputs_np.items()
    }
    rho_true = load_mat_array(data_root / "rho" / f"rho_{file_id}.mat", "rho")
    forward_labels = {}
    forward_masks = {}
    for name in ARRANGEMENTS:
        forward_labels[name], forward_masks[name] = _load_forward_payload(data_root, name, file_id)

    captured: dict[str, torch.Tensor] = {}
    plb = inverse_model.encoders["wa"].stages[0].blocks[0]

    def block_hook(_module, inputs, output):
        captured["plb_X"] = inputs[0].detach()
        captured["plb_Y"] = output.detach()

    def z_hook(_module, _inputs, output):
        captured["plb_Z"] = output.detach()

    def parameter_hook(_module, _inputs, output):
        captured["plb_parameters"] = output.detach()

    handle_block = plb.register_forward_hook(block_hook)
    handle_z = plb.spatial_mix.register_forward_hook(z_hook)
    handle_parameters = plb.liquid_projection.register_forward_hook(parameter_hook)
    with torch.no_grad():
        rho_pred = inverse_model(inverse_inputs)
        forward_pred = forward_model(rho_pred)
    handle_block.remove()
    handle_z.remove()
    handle_parameters.remove()

    arrays: dict[str, np.ndarray | int] = {
        "file_id": np.array([[file_id]], dtype=np.int32),
        "random_seed": np.array([[args.seed]], dtype=np.int32),
        "split": np.array(["train"], dtype="U5"),
        "rho_true": rho_true.astype(np.float32),
        "rho_pred": rho_pred.detach().cpu().numpy()[0, 0].astype(np.float32),
        "plb_X": _channel_mean(captured["plb_X"]),
        "plb_Z": _channel_mean(captured["plb_Z"]),
        "plb_Y": _channel_mean(captured["plb_Y"]),
    }
    arrays.update(_split_liquid_parameter_maps(captured["plb_parameters"]))
    for name in ARRANGEMENTS:
        upper = name.upper()
        arrays[f"{upper}_input"] = inverse_inputs_np[name].astype(np.float32)
        arrays[f"{upper}_forward_label"] = forward_labels[name].astype(np.float32)
        arrays[f"{upper}_forward_pred"] = forward_pred[name].detach().cpu().numpy()[0, 0].astype(np.float32)
        arrays[f"{upper}_mask"] = forward_masks[name].astype(np.float32)

    sio.savemat(sample_dir / "architecture_sample_blocks.mat", arrays, do_compression=True)
    print(sample_dir)
    return sample_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", default=str(repo_root / "examples" / "demo_data"))
    parser.add_argument("--run-root", default=str(repo_root / "checkpoints" / "ma_plb_unet_cl"))
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "outputs" / "architecture_sample_blocks"),
    )
    parser.add_argument("--file-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    export_blocks(args)


if __name__ == "__main__":
    main()
