from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hmno.data import HNOFieldDataset
from hmno.metrics import compute_metrics, scene_hotspot_error
from hmno.models import HNOModel
from hmno.utils import load_config, load_json, resolve_config_paths, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hmno.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = resolve_config_paths(load_config(args.config), args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 2024)))
    device = torch.device(cfg.get("project", {}).get("device", "cuda"))

    dataset = HNOFieldDataset(cfg, args.split)
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"].get("num_workers", 0), pin_memory=cfg["train"].get("pin_memory", False))

    model = HNOModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets = [], []
    infer_time = 0.0
    infer_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval [{args.split}]"):
            design, coords, temp, _ = batch
            design = design.to(device)
            coords = coords.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred = model(design, coords).squeeze(-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.perf_counter() - start
            infer_samples += int(design.shape[0])
            preds.append(pred.cpu())
            targets.append(temp)

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    stats = load_json(cfg["data"]["norm_stats_path"])
    metrics = compute_metrics(pred_all.flatten(), target_all.flatten(), stats)
    metrics["HotspotErr_C_scene"] = scene_hotspot_error(pred_all, target_all, stats)
    metrics["InferTime_ms"] = float(infer_time / max(infer_samples, 1) * 1000.0)

    out = Path(args.out) if args.out else Path(cfg["train"]["save_dir"]).parent / "eval_metrics.json"
    save_json(out, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
