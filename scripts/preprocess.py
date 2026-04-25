from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm

from hmno.utils import load_config, resolve_config_paths


def normalize_coords(coords: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    out = coords.astype(np.float32).copy()
    out[:, 0] = 2.0 * (out[:, 0] - x_min) / (x_max - x_min) - 1.0
    out[:, 1] = 2.0 * (out[:, 1] - y_min) / (y_max - y_min) - 1.0
    out[:, 2] = 2.0 * (out[:, 2] - z_min) / (z_max - z_min) - 1.0
    return out


def normalize_design(params: np.ndarray, norm_cfg: dict) -> np.ndarray:
    out = params.astype(np.float32).copy()
    rack_min, rack_max = norm_cfg["rack_power"]
    temp_min, temp_max = norm_cfg["crac_temp"]
    fan_min, fan_max = norm_cfg["crac_fan"]
    out[:, 0:10] = (out[:, 0:10] - rack_min) / (rack_max - rack_min)
    out[:, 10:12] = (out[:, 10:12] - temp_min) / (temp_max - temp_min)
    out[:, 12:14] = (out[:, 12:14] - fan_min) / (fan_max - fan_min)
    return np.clip(out, 0.0, 1.0)


def vtu_path(root: Path, scenario_id: int) -> Path:
    return root / str(scenario_id) / "VTM" / "region_0.vtu"


def compute_global_stats(vtu_root: Path, scenario_ids: np.ndarray) -> dict:
    t_min = np.inf
    t_max = -np.inf
    v_abs_max = 0.0
    for sid in tqdm(scenario_ids, desc="Computing global stats"):
        mesh = pv.read(vtu_path(vtu_root, int(sid)))
        temp = np.asarray(mesh.cell_data["Temperature (C)"]).reshape(-1)
        vel = np.asarray(mesh.cell_data["Velocity (m/s)"])
        t_min = min(t_min, float(temp.min()))
        t_max = max(t_max, float(temp.max()))
        v_abs_max = max(v_abs_max, float(np.abs(vel).max()))
    return {"t_min": t_min, "t_max": t_max, "v_abs_max": v_abs_max}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hmno.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config_paths(load_config(args.config), args.config)
    data_cfg = cfg["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    fields_dir = Path(data_cfg["fields_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_cfg["raw_csv_path"])
    scenario_ids = df.iloc[:, 0].astype(int).to_numpy()
    design = normalize_design(df.iloc[:, 1:].to_numpy(), data_cfg["norm"])
    np.save(data_cfg["design_path"], design.astype(np.float32))
    np.save(processed_dir / "scenario_ids.npy", scenario_ids)

    coords_path = Path(data_cfg["coords_path"])
    if args.force or not coords_path.exists():
        mesh = pv.read(vtu_path(Path(data_cfg["vtu_root_path"]), int(scenario_ids[0])))
        coords = normalize_coords(np.asarray(mesh.cell_centers().points), np.asarray(data_cfg["input_bounds"], dtype=np.float32))
        np.save(coords_path, coords.astype(np.float32))

    stats_path = Path(data_cfg["norm_stats_path"])
    if args.force or not stats_path.exists():
        stats = compute_global_stats(Path(data_cfg["vtu_root_path"]), scenario_ids)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    else:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    t_range = float(stats["t_max"]) - float(stats["t_min"])
    v_abs_max = float(stats["v_abs_max"])
    compress = bool(data_cfg.get("compress_fields", True))

    for sid in tqdm(scenario_ids, desc="Saving normalized fields"):
        out_path = fields_dir / f"scenario_{int(sid):04d}.npz"
        if out_path.exists() and not args.force:
            continue
        mesh = pv.read(vtu_path(Path(data_cfg["vtu_root_path"]), int(sid)))
        temp = np.asarray(mesh.cell_data["Temperature (C)"]).reshape(-1).astype(np.float32)
        vel = np.asarray(mesh.cell_data["Velocity (m/s)"]).astype(np.float32)
        payload = {"temp": (temp - float(stats["t_min"])) / t_range, "vel": vel / v_abs_max}
        if compress:
            np.savez_compressed(out_path, **payload)
        else:
            np.savez(out_path, **payload)


if __name__ == "__main__":
    main()
