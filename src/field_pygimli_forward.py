from __future__ import annotations

import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPDATA_DIR = ROOT / ".appdata"
APPDATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("APPDATA", str(APPDATA_DIR))

import numpy as np
import pygimli as pg
import scipy.io as sio
from pygimli.physics import ert
from scipy.interpolate import RegularGridInterpolator


ARRANGEMENTS = ("wa", "wb", "slm")
ARRAY_TYPES = {"wa": 1, "wb": 4, "slm": 7}
ELECTRODE_SPACING = 2.0
MODEL_DEPTH = 18.0
BACKGROUND_RESISTIVITY = 100.0


def _numbers(line: str) -> list[float]:
    return [
        float(token)
        for token in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)
    ]


def _read_res2dinv(path: Path) -> tuple[dict, np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    if len(lines) < 7:
        raise ValueError(f"Too few lines in {path}")
    header = {
        "source_file": path.name,
        "name": lines[0].strip(),
        "spacing": float(lines[1].strip()),
        "array_type": int(float(lines[2].strip())),
        "count": int(float(lines[3].strip())),
    }
    rows = [_numbers(line) for line in lines[6 : 6 + header["count"]]]
    if len(rows) != header["count"] or any(len(row) < 3 for row in rows):
        raise ValueError(f"Invalid data block in {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Inconsistent row widths in {path}")
    values = np.asarray(rows, dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, -1] <= 0):
        raise ValueError(f"Invalid apparent resistivity in {path}")
    return header, values


def find_survey_file(case_dir: Path, arrangement: str) -> Path:
    expected_type = ARRAY_TYPES[arrangement]
    matches: list[Path] = []
    for path in case_dir.glob("*.dat"):
        if "modres" in path.name.lower():
            continue
        try:
            header, _ = _read_res2dinv(path)
        except (ValueError, OSError):
            continue
        if header["array_type"] == expected_type:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one type-{expected_type} DAT for {arrangement}, found {matches}"
        )
    return matches[0]


def _electrode_positions(
    arrangement: str, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = rows[:, 0]
    spacing = rows[:, 1]
    if arrangement == "wa":
        a = center - 1.5 * spacing
        b = center + 1.5 * spacing
        m = center - 0.5 * spacing
        n = center + 0.5 * spacing
    elif arrangement == "wb":
        a = center - 1.5 * spacing
        b = center - 0.5 * spacing
        m = center + 0.5 * spacing
        n = center + 1.5 * spacing
    elif arrangement == "slm":
        if rows.shape[1] < 4:
            raise ValueError("SLM rows must contain center, spacing, factor, rhoa")
        factor = rows[:, 2]
        a = center - (factor + 0.5) * spacing
        b = center + (factor + 0.5) * spacing
        m = center - 0.5 * spacing
        n = center + 0.5 * spacing
    else:
        raise ValueError(f"Unknown arrangement: {arrangement}")
    return a, b, m, n


def load_field_survey(
    case_dir: Path, arrangement: str
) -> tuple[pg.DataContainerERT, dict]:
    source = find_survey_file(case_dir, arrangement)
    header, rows = _read_res2dinv(source)
    if header["array_type"] != ARRAY_TYPES[arrangement]:
        raise ValueError(f"Unexpected array type in {source}: {header['array_type']}")
    if not np.isclose(header["spacing"], ELECTRODE_SPACING):
        raise ValueError(f"Unexpected electrode spacing in {source}: {header['spacing']}")

    physical = _electrode_positions(arrangement, rows)
    all_positions = np.concatenate(physical)
    origin = float(
        np.rint(np.min(all_positions) / ELECTRODE_SPACING) * ELECTRODE_SPACING
    )
    line_end = float(
        np.rint(np.max(all_positions) / ELECTRODE_SPACING) * ELECTRODE_SPACING
    )
    line_length = line_end - origin
    sensor_count_float = line_length / ELECTRODE_SPACING + 1.0
    sensor_count = int(round(sensor_count_float))
    if not np.isclose(sensor_count, sensor_count_float, atol=1e-7):
        raise ValueError(f"Nonuniform survey extent in {source}")
    sensor_x = np.linspace(0.0, line_length, sensor_count)

    indices: list[np.ndarray] = []
    for values in physical:
        index_float = (values - origin) / ELECTRODE_SPACING
        index = np.rint(index_float).astype(np.int32)
        # Res2DInv stores some SLM factors with only five decimals.
        if not np.allclose(index_float, index, atol=1e-3):
            raise ValueError(f"Electrodes do not lie on the 2 m grid in {source}")
        if np.any(index < 0) or np.any(index >= sensor_count):
            raise ValueError(f"Electrode index outside survey line in {source}")
        indices.append(index)

    data = pg.DataContainerERT()
    for x in sensor_x:
        data.createSensor(pg.Pos(float(x), 0.0))
    data.resize(header["count"])
    for token, index in zip(("a", "b", "m", "n"), indices):
        data[token] = index
    data["rhoa"] = rows[:, -1]
    data["valid"] = np.ones(header["count"], dtype=float)
    data["err"] = np.full(header["count"], 0.03, dtype=float)
    data["k"] = ert.createGeometricFactors(data, numerical=False)

    reconstructed_center = sum(sensor_x[index] for index in indices) / 4.0 + origin
    if not np.allclose(reconstructed_center, rows[:, 0], atol=1e-3):
        raise ValueError(f"ABMN center validation failed for {source}")
    if np.any(np.asarray(data["k"], dtype=float) == 0):
        raise ValueError(f"Zero geometric factor in {source}")

    metadata = {
        **header,
        "arrangement": arrangement,
        "sensor_count": sensor_count,
        "line_origin_global": origin,
        "line_end_global": line_end,
        "line_length": line_length,
        "observed_rhoa_range": [float(rows[:, -1].min()), float(rows[:, -1].max())],
    }
    return data, metadata


def create_forward_mesh(line_length: float) -> pg.Mesh:
    x_start = -line_length / 2.0
    x_end = line_length / 2.0
    outer_step = ELECTRODE_SPACING * 2.0
    x_nodes = np.unique(
        list(pg.utils.grange(x_start - line_length, x_start, dx=outer_step))
        + list(pg.utils.grange(x_start, x_end, dx=1.0))
        + list(pg.utils.grange(x_end, x_end + line_length, dx=outer_step))
    )
    z_nodes = np.unique(
        list(pg.utils.grange(0.0, -3.0 * MODEL_DEPTH, dx=outer_step))
        + list(pg.utils.grange(-MODEL_DEPTH, 0.0, dx=1.0))
    )
    return pg.createGrid(x=x_nodes, y=z_nodes)


def map_log_model_to_mesh(
    mesh: pg.Mesh, rho_log10: np.ndarray, line_length: float
) -> pg.Vector:
    rho_log10 = np.asarray(rho_log10, dtype=float).squeeze()
    if rho_log10.shape != (256, 1024) or not np.isfinite(rho_log10).all():
        raise ValueError(f"Invalid predicted model: {rho_log10.shape}")
    rho_linear = np.power(10.0, rho_log10)
    x_old = np.linspace(-line_length / 2.0, line_length / 2.0, rho_log10.shape[1])
    z_old = np.linspace(0.0, -MODEL_DEPTH, rho_log10.shape[0])
    interpolator = RegularGridInterpolator(
        (z_old, x_old),
        rho_linear,
        bounds_error=False,
        fill_value=BACKGROUND_RESISTIVITY,
    )
    values = np.asarray(
        [float(interpolator((cell.center().y(), cell.center().x()))) for cell in mesh.cells()],
        dtype=float,
    )
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Invalid mesh resistivity")
    return pg.Vector(values)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    if np.any(observed <= 0) or np.any(predicted <= 0):
        raise ValueError("Physical forward response must be positive")
    observed_log = np.log10(observed)
    predicted_log = np.log10(predicted)
    residual_log = predicted_log - observed_log
    relative = (predicted - observed) / observed
    denominator = float(np.sum((observed_log - observed_log.mean()) ** 2))
    return {
        "data_count": int(observed.size),
        "log10_mse": float(np.mean(residual_log**2)),
        "log10_rmse": float(np.sqrt(np.mean(residual_log**2))),
        "log10_mae": float(np.mean(np.abs(residual_log))),
        "log10_r2": float(1.0 - np.sum(residual_log**2) / denominator),
        "relative_rmse_percent": float(100.0 * np.sqrt(np.mean(relative**2))),
        "median_absolute_percentage_error_percent": float(
            100.0 * np.median(np.abs(relative))
        ),
    }


def evaluate_model_physical(
    rho_log10: np.ndarray,
    case_dir: Path,
    output_dir: Path,
    model_name: str,
) -> dict:
    surveys = {
        arrangement: load_field_survey(case_dir, arrangement)
        for arrangement in ARRANGEMENTS
    }
    lengths = [metadata["line_length"] for _, metadata in surveys.values()]
    if not np.allclose(lengths, lengths[0], atol=1e-7):
        raise ValueError(f"Arrangement line lengths disagree: {lengths}")
    line_length = float(lengths[0])
    mesh = create_forward_mesh(line_length)
    resistivity = map_log_model_to_mesh(mesh, rho_log10, line_length)

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    arrangement_metrics: dict[str, dict] = {}
    pooled_observed: list[np.ndarray] = []
    pooled_predicted: list[np.ndarray] = []
    for arrangement in ARRANGEMENTS:
        survey, metadata = surveys[arrangement]
        centered = pg.DataContainerERT(survey)
        centered.translate(pg.Pos(-line_length / 2.0, 0.0))
        calibration_dir = case_dir / "processed" / "pygimli_physical_calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        calibration_path = calibration_dir / f"{arrangement}_homogeneous_100.mat"
        if calibration_path.exists():
            reference = np.asarray(
                sio.loadmat(calibration_path)["reference_rhoa"], dtype=float
            ).squeeze()
        else:
            homogeneous = ert.simulate(
                mesh,
                scheme=centered,
                res=pg.Vector(mesh.cellCount(), BACKGROUND_RESISTIVITY),
                noiseLevel=0.0,
                verbose=False,
            )
            reference = np.asarray(homogeneous["rhoa"], dtype=float)
            sio.savemat(
                calibration_path,
                {
                    "reference_rhoa": reference,
                    "reference_resistivity": BACKGROUND_RESISTIVITY,
                    "mesh_cell_count": mesh.cellCount(),
                    "line_length": line_length,
                },
                do_compression=True,
            )
        simulated = ert.simulate(
            mesh,
            scheme=centered,
            res=resistivity,
            noiseLevel=0.0,
            verbose=False,
        )
        predicted_uncalibrated = np.asarray(simulated["rhoa"], dtype=float)
        if reference.shape != predicted_uncalibrated.shape or np.any(reference <= 0):
            raise RuntimeError(f"Invalid homogeneous calibration for {arrangement}")
        predicted = (
            predicted_uncalibrated * BACKGROUND_RESISTIVITY / reference
        )
        simulated["rhoa"] = predicted
        observed = np.asarray(survey["rhoa"], dtype=float)
        if predicted.shape != observed.shape:
            raise RuntimeError(f"Response size mismatch for {arrangement}")
        metrics = _metrics(observed, predicted)
        metrics["source_file"] = metadata["source_file"]
        arrangement_metrics[arrangement] = metrics
        pooled_observed.append(observed)
        pooled_predicted.append(predicted)
        simulated.save(str(model_dir / f"{arrangement}_physical_forward.dat"))
        sio.savemat(
            model_dir / f"{arrangement}_physical_forward.mat",
            {
                "observed_rhoa": observed,
                "predicted_rhoa": predicted,
                "predicted_rhoa_uncalibrated": predicted_uncalibrated,
                "homogeneous_reference_rhoa": reference,
                "log10_residual": np.log10(predicted) - np.log10(observed),
                "a": np.asarray(survey["a"], dtype=np.int32),
                "b": np.asarray(survey["b"], dtype=np.int32),
                "m": np.asarray(survey["m"], dtype=np.int32),
                "n": np.asarray(survey["n"], dtype=np.int32),
                "sensor_x": np.asarray(
                    [position.x() for position in survey.sensorPositions()], dtype=float
                ),
            },
            do_compression=True,
        )

    pooled = _metrics(np.concatenate(pooled_observed), np.concatenate(pooled_predicted))
    macro_log10_mse = float(
        np.mean([arrangement_metrics[name]["log10_mse"] for name in ARRANGEMENTS])
    )
    report = {
        "method": "pyGIMLi physical forward using the exact field ABMN protocols",
        "model_name": model_name,
        "model_domain": "input and prediction are log10 resistivity; pyGIMLi uses linear resistivity",
        "mesh": {
            "cells": int(mesh.cellCount()),
            "line_length": line_length,
            "model_depth": MODEL_DEPTH,
            "outside_and_below_fill_ohm_m": BACKGROUND_RESISTIVITY,
            "fem_calibration": "pointwise homogeneous 100 Ohm m half-space response",
        },
        "arrangements": arrangement_metrics,
        "macro_mean_log10_mse": macro_log10_mse,
        "pooled": pooled,
    }
    (model_dir / "physical_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
