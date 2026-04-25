from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_or_create_split(data_cfg: Dict, scenario_ids: np.ndarray, seed: int, train_ratio: float):
    split_path = Path(data_cfg.get("split_scenarios_path", Path(data_cfg["processed_dir"]) / "split_scenarios.json"))
    if split_path.exists():
        import json

        payload = json.loads(split_path.read_text(encoding="utf-8"))
        return np.asarray(payload["train_indices"]), np.asarray(payload["val_indices"])

    rng = np.random.RandomState(seed)
    indices = np.arange(len(scenario_ids))
    rng.shuffle(indices)
    n_train = int(round(len(indices) * train_ratio))
    train_indices = np.sort(indices[:n_train])
    val_indices = np.sort(indices[n_train:])

    payload = {
        "seed": seed,
        "train_ratio": train_ratio,
        "n_total": int(len(indices)),
        "train_indices": train_indices.astype(int).tolist(),
        "val_indices": val_indices.astype(int).tolist(),
        "train_scenario_ids": scenario_ids[train_indices].astype(int).tolist(),
        "val_scenario_ids": scenario_ids[val_indices].astype(int).tolist(),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return train_indices, val_indices


class HNOFieldDataset(Dataset):
    def __init__(self, cfg: Dict, split: str = "train") -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.cfg = cfg
        self.split = split
        data_cfg = cfg["data"]
        processed_dir = Path(data_cfg["processed_dir"])

        self.coords = np.load(data_cfg["coords_path"]).astype(np.float32)
        self.design = np.load(data_cfg["design_path"]).astype(np.float32)
        scenario_path = processed_dir / "scenario_ids.npy"
        self.scenario_ids = np.load(scenario_path).astype(int) if scenario_path.exists() else np.arange(1, len(self.design) + 1)
        self.fields_dir = Path(data_cfg["fields_dir"])
        self.n_sample = int(data_cfg["n_sample_per_batch"] if split == "train" else data_cfg["n_sample_val"])

        seed = int(cfg.get("project", {}).get("seed", 2024))
        train_ratio = float(data_cfg.get("train_ratio", 0.9))
        train_idx, val_idx = load_or_create_split(data_cfg, self.scenario_ids, seed, train_ratio)
        self.indices = train_idx if split == "train" else val_idx

        if split == "train":
            sampling_cfg = data_cfg.get("sampling", {})
            self.weights = self._build_sampling_weights(
                self.coords,
                float(sampling_cfg.get("eps", 1.0e-3)),
                float(sampling_cfg.get("gamma", 2.0)),
                float(sampling_cfg.get("mix", 0.2)),
            )
        else:
            self.weights = None
            self.fixed_indices = self._load_or_create_val_indices(data_cfg, len(self.coords))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        scenario_idx = int(self.indices[idx])
        scenario_id = int(self.scenario_ids[scenario_idx])
        sample_idx = self._sample_train_indices() if self.split == "train" else self.fixed_indices

        with np.load(self.fields_dir / f"scenario_{scenario_id:04d}.npz") as data:
            temp = data["temp"].astype(np.float32)
            vel = data["vel"].astype(np.float32)

        return (
            torch.from_numpy(self.design[scenario_idx]),
            torch.from_numpy(self.coords[sample_idx]),
            torch.from_numpy(temp[sample_idx]),
            torch.from_numpy(vel[sample_idx]),
        )

    @staticmethod
    def _build_sampling_weights(coords: np.ndarray, eps: float, gamma: float, mix: float) -> torch.Tensor:
        d = np.minimum.reduce([1.0 - np.abs(coords[:, 0]), 1.0 - np.abs(coords[:, 1]), 1.0 - np.abs(coords[:, 2])])
        d = np.clip(d, 0.0, None)
        weights = mix * np.power(d + eps, gamma).astype(np.float64) + (1.0 - mix)
        weights = weights / weights.sum()
        return torch.from_numpy(weights.astype(np.float32))

    @staticmethod
    def _load_or_create_val_indices(data_cfg: Dict, n_coords: int) -> np.ndarray:
        path = Path(data_cfg.get("val_indices_path", Path(data_cfg["processed_dir"]) / "val_indices.npy"))
        if path.exists():
            return np.load(path)
        rng = np.random.RandomState(int(data_cfg.get("sampling", {}).get("seed", 2024)))
        idx = rng.choice(n_coords, size=int(data_cfg["n_sample_val"]), replace=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, idx)
        return idx

    def _sample_train_indices(self) -> np.ndarray:
        assert self.weights is not None
        return torch.multinomial(self.weights, self.n_sample, replacement=False).numpy()
