"""Resolve fast subset rows from Phase C/D classification matrices."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CELL_RE = re.compile(r"^(?P<instance>.+)\|p=(?P<p>[0-9.]+)\|B=(?P<shots>[0-9]+)\|zne=(?P<zne>[^|]+)$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell_sort_key(row: dict[str, str]) -> tuple[str, float, int, str]:
    match = CELL_RE.match(row["cell_id"])
    if not match:
        return (row.get("instance", ""), float("inf"), 10**9, row.get("zne_order", ""))
    return (
        match.group("instance"),
        float(match.group("p")),
        int(match.group("shots")),
        match.group("zne"),
    )


def selected_rows(path: Path, phase: str) -> list[dict[str, str]]:
    rows = read_rows(path)
    output = []
    for classification in sorted({row["classification"] for row in rows}):
        selected = sorted(
            [row for row in rows if row["classification"] == classification],
            key=cell_sort_key,
        )[0]
        match = CELL_RE.match(selected["cell_id"])
        output.append(
            {
                "phase": phase,
                "source": path.as_posix(),
                "classification": classification,
                "cell_id": selected["cell_id"],
                "instance": selected["instance"],
                "p": match.group("p") if match else "",
                "shots": match.group("shots") if match else selected.get("shots_total", ""),
                "zne": match.group("zne") if match else selected.get("zne_order", ""),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve first lexicographic subset rows.")
    parser.add_argument("--phase-d", default="results/phase_d/global_D1_D2_D3_D4/classification_matrix.csv")
    parser.add_argument("--phase-c", default="results/phase_c_120_analysis/classification_matrix.csv")
    args = parser.parse_args()

    rows = selected_rows(Path(args.phase_d), "D") + selected_rows(Path(args.phase_c), "C")
    for row in rows:
        print(
            f"{row['phase']}\t{row['classification']}\t{row['cell_id']}\t"
            f"instance={row['instance']}\tp={row['p']}\tB={row['shots']}\tzne={row['zne']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
