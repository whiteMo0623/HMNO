# Transparency Notes for Reviewer 4

This repository was prepared to address the request for code and data-generation transparency.

## Code Availability

The repository contains:

- HMNO model implementation;
- preprocessing from VTU CFD files;
- training and evaluation scripts;
- rank-sensitivity and physics-weight-sensitivity drivers;
- Reality DC Design Pro scenario-script generator;
- VTU inspection utility;
- dataset-format documentation.

## Data Generation

The `scripts/data_generation/generate_reality_scripts.py` script converts a CSV design table into one Reality DC Design Pro script per scenario. The generated scripts set:

- rack heat loads;
- CRAC supply temperatures;
- CRAC fan-speed ratios.

The CFD solve itself must be executed in Reality DC Design Pro or an equivalent CFD environment.

## Velocity Field

The velocity field used in the PDE residual is extracted from the CFD simulations. It is privileged information used during training only. During inference, HMNO requires only:

- the operating-condition vector;
- spatial coordinates.

## Large Artifacts

Raw CFD outputs, processed NumPy arrays, checkpoints, logs, and figures are excluded from Git. They should be distributed through a data archive if required for full binary-level reproducibility.
