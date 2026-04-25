# HMNO: Physics-Embedded Hyper-Modulated Neural Operator

This repository contains the public research code for **Physics-Embedded Hyper-Modulated Neural Operator (HMNO)**, a neural-operator surrogate for real-time reconstruction of high-resolution 3D thermal fields in data centers.

The code release is intended to improve reproducibility and transparency for the paper:

> Physics-Embedded Hyper-Modulated Neural Operator: Enabling Real-Time Thermal Digital Twins in Data Centers

## What Is Included

- `hmno/`: core HMNO implementation.
- `scripts/preprocess.py`: preprocessing of CFD VTU files into normalized arrays.
- `scripts/train_hmno.py`: HMNO training with data loss and velocity-informed physics residual.
- `scripts/evaluate_hmno.py`: validation/evaluation with physical metrics.
- `scripts/run_rank_sweep.py`: rank sensitivity experiment driver.
- `scripts/run_lambda_sweep.py`: physics-weight sensitivity experiment driver.
- `scripts/data_generation/generate_reality_scripts.py`: generation of Reality DC Design Pro scenario scripts from CSV design tables.
- `scripts/data_generation/inspect_vtu.py`: VTU structure and topology inspection utility.
- `configs/`: example YAML configurations.
- `docs/`: dataset format, reproducibility notes, and reviewer-transparency notes.

Large binary files are intentionally not committed:

- raw CFD datasets (`*.vtu`, `*.vtm`);
- processed arrays (`*.npy`, `*.npz`);
- model checkpoints (`*.pth`, `*.pt`, `*.ckpt`);
- generated logs, figures, and cache files.

This keeps the repository reviewable and avoids pushing multi-gigabyte artifacts into Git.

## Repository Layout

```text
HMNO/
+-- hmno/
|   +-- data.py
|   +-- losses.py
|   +-- metrics.py
|   +-- models.py
|   +-- utils.py
+-- scripts/
|   +-- preprocess.py
|   +-- train_hmno.py
|   +-- evaluate_hmno.py
|   +-- run_rank_sweep.py
|   +-- run_lambda_sweep.py
|   +-- data_generation/
+-- configs/
|   +-- hmno.yaml
|   +-- rank_sweep.yaml
|   +-- lambda_sweep.yaml
+-- docs/
|   +-- DATA.md
|   +-- REPRODUCIBILITY.md
|   +-- REVIEWER4_TRANSPARENCY.md
+-- requirements.txt
+-- README.md
```

## Environment

Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA-enabled PyTorch is recommended for training.

## Expected Dataset Format

The preprocessing script expects a design CSV and a CFD directory:

```text
project_root/
+-- rack_crac_design.csv
+-- data/
    +-- 1/VTM/region_0.vtu
    +-- 2/VTM/region_0.vtu
    +-- ...
```

The design CSV should contain:

```text
Scenario,
Rack1Power,...,Rack10Power,
CRAC1_Temp,CRAC2_Temp,
CRAC1_Fan,CRAC2_Fan
```

Each VTU file should contain cell data arrays:

- `Temperature (C)`: scalar temperature field.
- `Velocity (m/s)`: 3D velocity field used only during training for the physics residual.

See [docs/DATA.md](docs/DATA.md) for the complete format.

## Quick Start

From the repository root:

```bash
python scripts/preprocess.py --config configs/hmno.yaml
python scripts/train_hmno.py --config configs/hmno.yaml
python scripts/evaluate_hmno.py --config configs/hmno.yaml --checkpoint outputs/checkpoints/best_model.pth
```

The default configuration writes:

```text
data_processed/
outputs/checkpoints/
outputs/logs/
outputs/eval_metrics.json
```

## Main Configuration

The default HMNO configuration is [configs/hmno.yaml](configs/hmno.yaml). Key settings:

- 14-dimensional operating vector: 10 rack heat loads, 2 CRAC supply temperatures, and 2 CRAC fan-speed ratios.
- Fourier features: 64.
- Trunk MLP: `[131, 256, 256, 256, 128]`.
- Branch MLP: `[14, 128, 128]`.
- Mixer rank: `16`.
- Loss: `L = L_data + lambda_pde * L_PDE`, with `lambda_pde = 0.1` by default.

## Rank Sensitivity

```bash
python scripts/run_rank_sweep.py --config configs/rank_sweep.yaml
```

## Physics-Weight Sensitivity

```bash
python scripts/run_lambda_sweep.py --config configs/lambda_sweep.yaml
```

## Data Generation Scripts

The script in `scripts/data_generation/generate_reality_scripts.py` converts a design CSV into one Reality DC Design Pro script per scenario. The generated scripts set rack power, CRAC supply temperature, and CRAC fan speed according to each CSV row.

Example:

```bash
python scripts/data_generation/generate_reality_scripts.py --csv rack_crac_design.csv --outdir generated_scenarios --prefix scenario_
```

The generated scripts are intended to be executed inside the Reality DC Design Pro scripting environment, where `Model` and `Solver` APIs are available.

## Reproducibility Notes

The main experiments in the paper used fixed CFD mesh topology across scenarios, shared train/validation split, fixed validation sampling indices, denormalized metrics in degrees Celsius, and velocity fields only as training-time privileged information for the PDE residual.

Detailed notes are provided in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Important Limitations

- The provided model assumes a fixed geometry and fixed CFD mesh topology across scenarios.
- The velocity field is required only during training when the PDE residual is enabled.
- The scalar learnable diffusivity is an effective regularization parameter rather than a full turbulence closure model.
- Raw CFD data may require separate hosting because it is too large for a normal Git repository.

## Citation

If you use this code, please cite the associated paper once bibliographic information is available.
