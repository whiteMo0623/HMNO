from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def resolve_config_paths(cfg: Dict[str, Any], config_path: str | Path) -> Dict[str, Any]:
    base = Path(config_path).resolve().parent.parent
    data_cfg = cfg.get("data", {})
    for key in [
        "raw_csv_path",
        "vtu_root_path",
        "processed_dir",
        "coords_path",
        "design_path",
        "fields_dir",
        "norm_stats_path",
        "split_scenarios_path",
        "val_indices_path",
    ]:
        if key in data_cfg:
            data_cfg[key] = resolve_path(base, data_cfg[key])
    if "train" in cfg and "save_dir" in cfg["train"]:
        cfg["train"]["save_dir"] = resolve_path(base, cfg["train"]["save_dir"])
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
