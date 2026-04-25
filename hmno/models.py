from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
from torch import nn


def build_mlp(layers: Iterable[int], activation: type[nn.Module]) -> nn.Sequential:
    layers = list(layers)
    modules: list[nn.Module] = []
    for i in range(len(layers) - 1):
        modules.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            modules.append(activation())
    return nn.Sequential(*modules)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_features: int, scale: float = 1.0) -> None:
        super().__init__()
        b = torch.randn(in_dim, num_features) * scale
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.b)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class HNOModel(nn.Module):
    """Physics-embedded Hyper-Modulated Neural Operator.

    The trunk network learns reusable geometry-aware spatial bases from
    coordinates. The branch network maps the 14-dimensional operating condition
    to low-rank modulation parameters that deform those bases.
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]

        self.concat_xyz = bool(model_cfg.get("concat_xyz", True))
        self.fourier_features = int(model_cfg["fourier_features"])
        expected_in = 2 * self.fourier_features + (3 if self.concat_xyz else 0)

        trunk_in_dim = int(model_cfg["trunk_in_dim"])
        if trunk_in_dim != expected_in:
            raise ValueError(f"trunk_in_dim={trunk_in_dim} does not match expected {expected_in}")

        trunk_layers = list(model_cfg["trunk_layers"])
        if trunk_layers[0] != trunk_in_dim:
            raise ValueError("trunk_layers[0] must match trunk_in_dim")

        self.fourier = FourierFeatures(in_dim=3, num_features=self.fourier_features)
        self.trunk = build_mlp(trunk_layers, nn.SiLU)
        self.k = int(trunk_layers[-1])
        self.rank = int(model_cfg["mixer_rank"])

        branch_layers = list(model_cfg["branch_layers"])
        out_dim = self.k * self.rank + self.rank + 1
        self.branch = build_mlp(branch_layers + [out_dim], nn.ReLU)
        self.mixer_activation = nn.SiLU()

        alpha_init = float(model_cfg.get("alpha_init", 0.01))
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, design: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(design.shape[0], -1, -1)

        feats = self.fourier(coords)
        if self.concat_xyz:
            feats = torch.cat([feats, coords], dim=-1)

        basis = self.trunk(feats)
        weights = self.branch(design)
        batch = weights.shape[0]

        k_r = self.k * self.rank
        w1 = weights[:, :k_r].view(batch, self.k, self.rank)
        w2 = weights[:, k_r : k_r + self.rank].view(batch, self.rank, 1)
        bias = weights[:, -1].view(batch, 1, 1)

        hidden = torch.matmul(basis, w1)
        hidden = self.mixer_activation(hidden)
        return torch.matmul(hidden, w2) + bias
