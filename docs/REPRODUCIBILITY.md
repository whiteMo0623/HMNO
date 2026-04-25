# Reproducibility Notes

## Fixed Geometry Assumption

HMNO in this release assumes a fixed data-center geometry and a fixed CFD mesh topology across scenarios. The coordinates are therefore precomputed once and reused across all operating conditions.

## Train/Validation Split

The split is stored in `data_processed/split_scenarios.json`. Reusing this file is the recommended way to compare HMNO with baselines because it fixes the scenario partition.

## Validation Sampling

Validation uses `data_processed/val_indices.npy`, a fixed set of spatial cell indices. This avoids changes in reported metrics caused by different validation point sampling.

## Metrics

The evaluation script reports denormalized physical metrics:

- `MAE_C`;
- `RMSE_C`;
- `MaxAbs_C`;
- `P95Abs_C`;
- `HotspotErr_C`;
- `HotspotErr_C_scene`;
- `R2`;
- `InferTime_ms`.

`HotspotErr_C` is the global maximum-temperature prediction error. `HotspotErr_C_scene` first computes hotspot error per scenario and then averages across scenarios.

## Physics Residual

The training loss is:

```text
L_total = L_data + lambda_pde * L_PDE
```

where:

```text
L_PDE = || v · grad(T_pred) - alpha * laplacian(T_pred) ||^2
```

The residual is a soft regularizer. It should be interpreted as a transport-consistency prior rather than a strict numerical PDE solver.

## Raw Data

Raw CFD files are not included in Git because they are large binary artifacts. For a public release, host them separately through an archival service or institutional repository and provide the download link here.
