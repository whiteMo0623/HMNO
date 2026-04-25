from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hmno.data import HNOFieldDataset
from hmno.losses import compute_pde_residual
from hmno.metrics import compute_metrics
from hmno.models import HNOModel
from hmno.utils import load_config, load_json, resolve_config_paths, save_json, set_seed


def is_better(metric: str, current: float, best: float) -> bool:
    return current > best if metric == "R2" else current < best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hmno.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = resolve_config_paths(load_config(args.config), args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 2024)))
    device = torch.device(cfg.get("project", {}).get("device", "cuda"))

    train_ds = HNOFieldDataset(cfg, "train")
    val_ds = HNOFieldDataset(cfg, "val")
    train_cfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg.get("num_workers", 0), pin_memory=train_cfg.get("pin_memory", False))
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg.get("num_workers", 0), pin_memory=train_cfg.get("pin_memory", False))

    model = HNOModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))

    start_epoch = 1
    best_metric_name = train_cfg.get("best_metric", "RMSE_C")
    best_metric = -float("inf") if best_metric_name == "R2" else float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))

    save_dir = Path(train_cfg["save_dir"])
    log_path = save_dir.parent / "logs" / "train_log.jsonl"
    best_path = save_dir / "best_model.pth"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    weights = train_cfg.get("loss_weights", {})
    w_data = float(weights.get("data", 1.0))
    w_pde = float(weights.get("pde", 0.1))
    stats = load_json(cfg["data"]["norm_stats_path"])

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            design, coords, temp, vel = batch
            design = design.to(device)
            coords = coords.to(device).detach().requires_grad_(True)
            temp = temp.to(device).unsqueeze(-1)
            vel = vel.to(device)

            pred = model(design, coords)
            data_loss = torch.nn.functional.mse_loss(pred, temp)
            if w_pde > 0.0:
                residual = compute_pde_residual(pred, coords, vel, model.alpha)
                pde_loss = torch.mean(residual**2)
            else:
                pde_loss = torch.zeros((), device=device)
            loss = w_data * data_loss + w_pde * pde_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                design, coords, temp, _ = batch
                pred = model(design.to(device), coords.to(device)).squeeze(-1)
                preds.append(pred.cpu())
                targets.append(temp)

        pred_all = torch.cat(preds, dim=0)
        target_all = torch.cat(targets, dim=0)
        metrics = compute_metrics(pred_all.flatten(), target_all.flatten(), stats)
        metrics.update({"epoch": epoch, "alpha": float(model.alpha.detach().cpu().item())})

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        current = float(metrics[best_metric_name])
        if is_better(best_metric_name, current, best_metric):
            best_metric = current
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_metric": best_metric, "metrics": metrics, "config": cfg}, best_path)
            save_json(save_dir.parent / "best_metrics.json", metrics)


if __name__ == "__main__":
    main()
