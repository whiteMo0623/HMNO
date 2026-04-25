from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


CRAC1_ID = "444837-100783354"
CRAC2_ID = "588352-100783462"
RACK_IDS: List[str] = [
    "388996-100782551",
    "388998-100782626",
    "389000-100782701",
    "389002-100782776",
    "389004-100782851",
    "389007-100782926",
    "389009-100783001",
    "389011-100783076",
    "389013-100783151",
    "389015-100783226",
]


def render_script(row: Dict[str, str]) -> str:
    scenario = int(row["Scenario"])
    lines: List[str] = [f"# Scenario {scenario}"]

    lines.append(f"Model.setSelectEntities([Model.Entity('{CRAC1_ID}')])")
    lines.append(
        f"Model.Editor.setBytesValue(Model.Entity('{CRAC1_ID}'), "
        f"'Controls.Air Temperature Control.Set Point', b'{float(row['CRAC1_Temp']):.1f} C')"
    )
    lines.append(
        f"Model.Editor.setBytesValue(Model.Entity('{CRAC1_ID}'), "
        f"'Controls.Air Flow Controls.Fan Speed', b'{float(row['CRAC1_Fan']):.1f} %')"
    )

    lines.append(f"Model.setSelectEntities([Model.Entity('{CRAC2_ID}')])")
    lines.append(
        f"Model.Editor.setBytesValue(Model.Entity('{CRAC2_ID}'), "
        f"'Controls.Air Temperature Control.Set Point', b'{float(row['CRAC2_Temp']):.1f} C')"
    )
    lines.append(
        f"Model.Editor.setBytesValue(Model.Entity('{CRAC2_ID}'), "
        f"'Controls.Air Flow Controls.Fan Speed', b'{float(row['CRAC2_Fan']):.1f} %')"
    )

    for idx, rack_id in enumerate(RACK_IDS, start=1):
        power = float(row[f"Rack{idx}Power"])
        lines.append(f"Model.setSelectEntities([Model.Entity('{rack_id}')])")
        lines.append(
            f"Model.Editor.setBytesValue(Model.Entity('{rack_id}'), "
            f"'IT Specification.Black Box Cabinet Power', b'{power:.2f} kW')"
        )

    lines.append("# Solver.generateGrid(True)")
    lines.append("# Solver.verifyModel(True)")
    lines.append("# Solver.solve(0, False)")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one Reality DC Design Pro script per scenario.")
    parser.add_argument("--csv", required=True, help="Input design CSV.")
    parser.add_argument("--outdir", required=True, help="Output directory for generated scripts.")
    parser.add_argument("--prefix", default="scenario_", help="Filename prefix.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    width = max(4, len(str(max(int(row["Scenario"]) for row in rows))))
    for row in rows:
        scenario = int(row["Scenario"])
        (outdir / f"{args.prefix}{scenario:0{width}d}.py").write_text(render_script(row), encoding="utf-8")

    print(f"Generated {len(rows)} scenario scripts in {outdir}")


if __name__ == "__main__":
    main()
