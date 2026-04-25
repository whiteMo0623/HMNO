from __future__ import annotations

from typing import Dict

import torch


def denorm_temperature(t_norm: torch.Tensor, stats: Dict) -> torch.Tensor:
    t_min = float(stats["t_min"])
    t_max = float(stats["t_max"])
    return t_norm * (t_max - t_min) + t_min


def compute_metrics(pred_norm: torch.Tensor, target_norm: torch.Tensor, stats: Dict) -> Dict[str, float]:
    pred = denorm_temperature(pred_norm, stats)
    target = denorm_temperature(target_norm, stats)
    diff = pred - target
    abs_diff = diff.abs()

    ss_res = (diff**2).sum()
    target_mean = target.mean()
    ss_tot = ((target - target_mean) ** 2).sum()

    return {
        "MAE_C": float(abs_diff.mean().item()),
        "RMSE_C": float(torch.sqrt((diff**2).mean()).item()),
        "MaxAbs_C": float(abs_diff.max().item()),
        "P95Abs_C": float(torch.quantile(abs_diff, 0.95).item()),
        "HotspotErr_C": float((pred.max() - target.max()).item()),
        "R2": float((1.0 - ss_res / (ss_tot + 1.0e-12)).item()),
    }


def scene_hotspot_error(pred_norm: torch.Tensor, target_norm: torch.Tensor, stats: Dict) -> float:
    pred = denorm_temperature(pred_norm, stats)
    target = denorm_temperature(target_norm, stats)
    return float((pred.max(dim=1).values - target.max(dim=1).values).mean().item())
