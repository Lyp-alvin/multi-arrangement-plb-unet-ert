from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygimli as pg
import scipy.io as sio
from pygimli.physics import ert
from scipy.interpolate import RegularGridInterpolator


IDS = (4, 52, 126, 347, 434, 484, 487, 512)
ELEC_COUNT = 50
SPACING = 2.5
ARRAY_LENGTH = (ELEC_COUNT - 1) * SPACING
DEPTH = max(int(ARRAY_LENGTH * 0.15), 1)


def create_mesh_for_forward() -> pg.Mesh:
    inner_half = ARRAY_LENGTH / 2.0
    outer_grid_size = SPACING * 2.0
    x_inner_grid_size = 1.25
    y_inner_grid_size = 1.0
    x_nodes = np.unique(
        list(pg.utils.grange(-inner_half - ARRAY_LENGTH, -inner_half, dx=outer_grid_size))
        + list(pg.utils.grange(-inner_half, inner_half, dx=x_inner_grid_size))
        + list(pg.utils.grange(inner_half, inner_half + ARRAY_LENGTH, dx=outer_grid_size))
    )
    z_nodes = np.unique(
        list(pg.utils.grange(0, -DEPTH - 2 * DEPTH, dx=outer_grid_size))
        + list(pg.utils.grange(-DEPTH, 0, dx=y_inner_grid_size))
    )
    return pg.createGrid(x=x_nodes, y=z_nodes)


def map_log_rho_to_mesh(rho_log: np.ndarray, mesh: pg.Mesh) -> pg.Vector:
    rho_log = np.asarray(rho_log, dtype=float).squeeze()
    if rho_log.shape != (256, 1024):
        raise ValueError(f"Expected 256x1024 rho_pred, got {rho_log.shape}")
    if not np.isfinite(rho_log).all():
        raise ValueError("rho_pred contains non-finite values")

    rho_linear = np.power(10.0, rho_log)
    z_old = np.linspace(0.0, -float(DEPTH), rho_log.shape[0])
    x_old = np.linspace(-ARRAY_LENGTH / 2.0, ARRAY_LENGTH / 2.0, rho_log.shape[1])
    interpolator = RegularGridInterpolator(
        (z_old, x_old),
        rho_linear,
        bounds_error=False,
        fill_value=100.0,
    )
    values = np.empty(mesh.cellCount(), dtype=float)
    for index, cell in enumerate(mesh.cells()):
        center = cell.center()
        values[index] = interpolator((center[1], center[0]))
    return pg.Vector(values)


def compute_points_and_levels(data, electrode_x: np.ndarray):
    a = np.asarray(data["a"], dtype=int)
    b = np.asarray(data["b"], dtype=int)
    m = np.asarray(data["m"], dtype=int)
    n = np.asarray(data["n"], dtype=int)
    try:
        elec_pos = np.asarray(
            [data.sensorPosition(i)[0] for i in range(data.sensorCount())],
            dtype=float,
        )
    except Exception:
        elec_pos = electrode_x
    x_mid = (elec_pos[a] + elec_pos[b] + elec_pos[m] + elec_pos[n]) / 4.0
    am_distance = np.abs(elec_pos[a] - elec_pos[m])
    levels = np.round(am_distance / SPACING).astype(np.int32)
    levels[levels < 1] = 1
    return a, b, m, n, x_mid, levels


def run(args: argparse.Namespace) -> None:
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    wa_dir = output_dir / "wa"
    wa_dir.mkdir(parents=True, exist_ok=True)
    ids = tuple(int(value) for value in args.ids)

    mesh = create_mesh_for_forward()
    electrode_x = np.linspace(-ARRAY_LENGTH / 2.0, ARRAY_LENGTH / 2.0, ELEC_COUNT)
    scheme = ert.createData(elecs=electrode_x, schemeName="wa")

    for file_id in ids:
        prediction_path = prediction_dir / f"inverse_prediction_id_{file_id}.mat"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction: {prediction_path}")
        payload = sio.loadmat(prediction_path)
        if "rho_pred" not in payload:
            raise KeyError(f"Missing rho_pred in {prediction_path}")

        rho_log = np.asarray(payload["rho_pred"], dtype=float).squeeze()
        resistivity = map_log_rho_to_mesh(rho_log, mesh)
        data = ert.simulate(
            mesh,
            scheme=scheme,
            res=resistivity,
            noiseLevel=0.0,
            noiseAbs=0.0,
            verbose=False,
        )
        data.remove(data["rhoa"] <= 0)
        dat_path = wa_dir / f"forward_unet_open_loop_{file_id}.dat"
        data.save(str(dat_path))

        rhoa = np.asarray(data["rhoa"], dtype=float)
        a, b, m, n, x_mid, levels = compute_points_and_levels(data, electrode_x)
        sio.savemat(
            wa_dir / f"raw_scatter_unet_open_loop_{file_id}.mat",
            {
                "rhoa": rhoa.astype(np.float64),
                "a": a.astype(np.int32),
                "b": b.astype(np.int32),
                "m": m.astype(np.int32),
                "n": n.astype(np.int32),
                "x_mid": x_mid.astype(np.float64),
                "levels": levels.astype(np.int32),
                "a_unit": np.float64(SPACING),
                "elec_x": electrode_x.astype(np.float64),
                "source_prediction": str(prediction_path),
                "noise_level": np.float64(0.0),
            },
            do_compression=True,
        )
        print(f"saved id={file_id} dat={dat_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir",
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "sa_unet_ol" / "open_inverse" / "evaluation" / "predictions_mat"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "unet_open_loop_pygimli_forward_wa"),
    )
    parser.add_argument("--ids", type=int, nargs="+", default=list(IDS))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
