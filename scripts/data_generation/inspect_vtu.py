from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def inspect_vtu(path: Path) -> None:
    mesh = pv.read(path)
    print("VTU structure")
    print(f"  type: {type(mesh).__name__}")
    print(f"  n_points: {mesh.n_points}")
    print(f"  n_cells: {mesh.n_cells}")
    print(f"  bounds: {mesh.bounds}")

    print("\nPoint data")
    if not mesh.point_data.keys():
        print("  none")
    for name in mesh.point_data.keys():
        arr = np.asarray(mesh.point_data[name])
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}")

    print("\nCell data")
    if not mesh.cell_data.keys():
        print("  none")
    for name in mesh.cell_data.keys():
        arr = np.asarray(mesh.cell_data[name])
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vtu", help="Path to region_0.vtu")
    args = parser.parse_args()
    inspect_vtu(Path(args.vtu))


if __name__ == "__main__":
    main()
