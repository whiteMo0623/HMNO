from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def lambda_tag(value: float) -> str:
    return f"lambda_{value:.3f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lambda_sweep.yaml")
    args = parser.parse_args()

    sweep_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg_path = Path(sweep_cfg["base_config"])
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    output_root = Path(sweep_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    for value in sweep_cfg["lambda_values"]:
        value = float(value)
        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["loss_weights"]["pde"] = value
        cfg["train"]["save_dir"] = str(output_root / lambda_tag(value) / "checkpoints")
        cfg_path = output_root / lambda_tag(value) / "config_used.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, "scripts/train_hmno.py", "--config", str(cfg_path)], check=True)
        subprocess.run([sys.executable, "scripts/evaluate_hmno.py", "--config", str(cfg_path), "--checkpoint", str(Path(cfg["train"]["save_dir"]) / "best_model.pth")], check=True)


if __name__ == "__main__":
    main()
