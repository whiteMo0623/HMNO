from __future__ import annotations

import torch


def compute_pde_residual(
    temp_pred: torch.Tensor,
    coords: torch.Tensor,
    velocity: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Compute a steady advection-diffusion residual on normalized coordinates."""

    grad_outputs = torch.ones_like(temp_pred)
    grad_t = torch.autograd.grad(
        temp_pred,
        coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    laplace = 0.0
    for dim in range(3):
        grad_dim = grad_t[..., dim]
        grad2 = torch.autograd.grad(
            grad_dim,
            coords,
            grad_outputs=torch.ones_like(grad_dim),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0][..., dim]
        laplace = laplace + grad2

    convection = (velocity * grad_t).sum(dim=-1)
    return convection - alpha * laplace
