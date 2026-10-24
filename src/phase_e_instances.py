from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_INSTANCE_VERSION = "phase_e_instances_v0_1_draft"
DEFAULT_SELECTION_SEED = 20260517
DEFAULT_PILOT_SEED = 20260518
DEPTHS_FOR_LIGHTCONE = (1, 2)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_phase_d_instances(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "results" / "phase_d" / "planning" / "phase_d_selected_instances_final.json"
    if not path.exists():
        path = root / "config" / "phase_d_instances_v1.0.json"
    payload = _read_json(path)
    if "selected_instances_H1_to_H12" in payload:
        return payload["selected_instances_H1_to_H12"], path
    return payload, path


def _degree_values(n_nodes: int, edges: list[list[float]]) -> list[int]:
    degrees = [0 for _ in range(n_nodes)]
    for u, v, _ in edges:
        degrees[int(u)] += 1
        degrees[int(v)] += 1
    return degrees


def _adjacency(n_nodes: int, edges: list[list[float]]) -> dict[int, set[int]]:
    adj = {i: set() for i in range(n_nodes)}
    for u, v, _ in edges:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj


def _ball_nodes(adj: dict[int, set[int]], start_nodes: tuple[int, int], radius: int) -> set[int]:
    seen = set(start_nodes)
    queue: deque[tuple[int, int]] = deque((node, 0) for node in start_nodes)
    while queue:
        node, dist = queue.popleft()
        if dist >= radius:
            continue
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return seen


def _structural_rows(instances: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, info in sorted(instances.items(), key=lambda item: int(item[0][1:])):
        edges = info["edges"]
        n_nodes = int(info["n_nodes"])
        n_edges = len(edges)
        degrees = _degree_values(n_nodes, edges)
        weights = np.asarray([float(edge[2]) for edge in edges], dtype=float)
        rows.append(
            {
                "selected_label": label,
                "candidate_id": info["candidate_id"],
                "stratum_id": info["stratum_id"],
                "stratum_name": info["stratum_name"],
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": float(info.get("density", 2.0 * n_edges / (n_nodes * (n_nodes - 1)))),
                "weighted_flag": bool(info.get("weighted_flag", bool(np.std(weights) > 1e-12))),
                "degree_min": min(degrees),
                "degree_mean": float(np.mean(degrees)),
                "degree_max": max(degrees),
                "degree_std": float(np.std(degrees)),
                "weight_min": float(np.min(weights)),
                "weight_mean": float(np.mean(weights)),
                "weight_max": float(np.max(weights)),
                "weight_std": float(np.std(weights)),
                "grid_label": info.get("grid_label", ""),
                "graph_signature": info.get("graph_signature", ""),
            }
        )
    return rows


def _lightcone_rows(instances: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, info in sorted(instances.items(), key=lambda item: int(item[0][1:])):
        edges = info["edges"]
        n_nodes = int(info["n_nodes"])
        adj = _adjacency(n_nodes, edges)
        for depth in DEPTHS_FOR_LIGHTCONE:
            node_counts = []
            edge_counts = []
            weight_sums = []
            for edge_index, (u, v, _w) in enumerate(edges):
                nodes = _ball_nodes(adj, (int(u), int(v)), radius=depth)
                local_edges = [
                    edge
                    for edge in edges
                    if int(edge[0]) in nodes and int(edge[1]) in nodes
                ]
                node_counts.append(len(nodes))
                edge_counts.append(len(local_edges))
                weight_sums.append(sum(float(edge[2]) for edge in local_edges))
                rows.append(
                    {
                        "row_type": "edge_lightcone",
                        "selected_label": label,
                        "candidate_id": info["candidate_id"],
                        "stratum_id": info["stratum_id"],
                        "depth": depth,
                        "edge_index": edge_index,
                        "edge_u": int(u),
                        "edge_v": int(v),
                        "lightcone_nodes": len(nodes),
                        "lightcone_edges": len(local_edges),
                        "lightcone_weight_sum": float(weight_sums[-1]),
                    }
                )
            rows.append(
                {
                    "row_type": "instance_lightcone_summary",
                    "selected_label": label,
                    "candidate_id": info["candidate_id"],
                    "stratum_id": info["stratum_id"],
                    "depth": depth,
                    "edge_index": "",
                    "edge_u": "",
                    "edge_v": "",
                    "lightcone_nodes_mean": float(np.mean(node_counts)),
                    "lightcone_nodes_min": int(np.min(node_counts)),
                    "lightcone_nodes_max": int(np.max(node_counts)),
                    "lightcone_edges_mean": float(np.mean(edge_counts)),
                    "lightcone_edges_min": int(np.min(edge_counts)),
                    "lightcone_edges_max": int(np.max(edge_counts)),
                    "lightcone_weight_sum_mean": float(np.mean(weight_sums)),
                    "lightcone_weight_sum_min": float(np.min(weight_sums)),
                    "lightcone_weight_sum_max": float(np.max(weight_sums)),
                }
            )
    return rows


def _sample_one_per_stratum(instances: dict[str, Any], seed: int) -> list[str]:
    by_stratum: dict[str, list[str]] = {}
    for label, info in instances.items():
        by_stratum.setdefault(str(info["stratum_id"]), []).append(label)
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for stratum in sorted(by_stratum):
        labels = sorted(by_stratum[stratum], key=lambda x: int(x[1:]))
        index = int(rng.integers(0, len(labels)))
        selected.append(labels[index])
    return selected


def _pilot_subset(selected_labels: list[str], instances: dict[str, Any], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    weighted = [label for label in selected_labels if bool(instances[label].get("weighted_flag", False))]
    unweighted = [label for label in selected_labels if not bool(instances[label].get("weighted_flag", False))]
    pilot: list[str] = []
    if unweighted:
        pilot.append(sorted(unweighted, key=lambda x: int(x[1:]))[int(rng.integers(0, len(unweighted)))])
    if weighted:
        pilot.append(sorted(weighted, key=lambda x: int(x[1:]))[int(rng.integers(0, len(weighted)))])
    if len(pilot) < 2:
        remaining = [label for label in selected_labels if label not in pilot]
        pilot.extend(list(rng.choice(sorted(remaining), size=2 - len(pilot), replace=False)))
    return sorted(pilot, key=lambda x: int(x[1:]))


def _coverage_summary(rows: list[dict[str, Any]], selected_labels: list[str]) -> dict[str, Any]:
    selected_rows = [row for row in rows if row["selected_label"] in selected_labels]
    densities = np.asarray([row["density"] for row in selected_rows], dtype=float)
    edges = np.asarray([row["n_edges"] for row in selected_rows], dtype=float)
    nodes = np.asarray([row["n_nodes"] for row in selected_rows], dtype=float)
    return {
        "selected_count": len(selected_rows),
        "selected_labels": selected_labels,
        "strata_covered": sorted({row["stratum_id"] for row in selected_rows}),
        "weighted_count": int(sum(1 for row in selected_rows if row["weighted_flag"])),
        "unweighted_count": int(sum(1 for row in selected_rows if not row["weighted_flag"])),
        "n_nodes_range": [int(np.min(nodes)), int(np.max(nodes))],
        "n_edges_range": [int(np.min(edges)), int(np.max(edges))],
        "density_range": [float(np.min(densities)), float(np.max(densities))],
        "structurally_narrow_warning": bool(
            len({row["stratum_id"] for row in selected_rows}) < 6
            or len(set(int(row["n_nodes"]) for row in selected_rows)) < 2
            or len(set(bool(row["weighted_flag"]) for row in selected_rows)) < 2
        ),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    rows = payload["selected_instances"]
    lines = [
        "# Phase E Instance Selection And Lightcone Diagnostics",
        "",
        f"Created UTC: {payload['created_at_utc']}",
        f"Status: `{payload['status']}`",
        "",
        "This is a draft pre-pilot artifact. It samples one Phase D H-instance per stratum from the already frozen H1-H12 pool using a recorded PRNG seed. It does not use Phase E depth-2 outcomes.",
        "",
        "## Selected Depth-2 Instances",
        "",
        "| Label | Stratum | Candidate | Nodes | Edges | Weighted |",
        "|---|---|---|---:|---:|---|",
    ]
    for label in payload["selected_depth2_instances"]:
        row = rows[label]
        lines.append(
            f"| {label} | {row['stratum_id']} | {row['candidate_id']} | "
            f"{row['n_nodes']} | {row['n_edges']} | {row['weighted_flag']} |"
        )
    lines.extend(
        [
            "",
            f"Pilot subset: `{', '.join(payload['pilot_depth2_instances'])}`",
            "",
            "## Coverage",
            "",
            f"- Strata covered: {', '.join(payload['coverage_summary']['strata_covered'])}",
            f"- Weighted/unweighted count: {payload['coverage_summary']['weighted_count']} / {payload['coverage_summary']['unweighted_count']}",
            f"- Node range: {payload['coverage_summary']['n_nodes_range']}",
            f"- Edge range: {payload['coverage_summary']['n_edges_range']}",
            f"- Density range: {payload['coverage_summary']['density_range']}",
            f"- Structurally narrow warning: `{payload['coverage_summary']['structurally_narrow_warning']}`",
            "",
            "Next step: run the unified depth-2 pilot only after reviewing this selection and keeping the protocol draft status explicit.",
            "",
        ]
    )
    return "\n".join(lines)


def run_phase_e_instance_selection(
    root: Path = PROJECT_ROOT,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    pilot_seed: int = DEFAULT_PILOT_SEED,
    force: bool = False,
) -> dict[str, Any]:
    config_path = root / "config" / "phase_e_instances_v0.1-draft.json"
    planning_dir = ensure_dir(root / "results" / "phase_e" / "planning")
    structural_csv = planning_dir / "phase_e_structural_diagnostics.csv"
    lightcone_csv = planning_dir / "phase_e_lightcone_diagnostics.csv"
    report_md = planning_dir / "phase_e_instance_selection_report.md"
    hash_path = planning_dir / "phase_e_instance_selection_hashes.json"
    if any(path.exists() for path in (config_path, structural_csv, lightcone_csv, report_md, hash_path)) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E instance-selection outputs.")

    instances, source_path = _selected_phase_d_instances(root)
    structural_rows = _structural_rows(instances)
    lightcone_rows = _lightcone_rows(instances)
    selected_labels = _sample_one_per_stratum(instances, selection_seed)
    pilot_labels = _pilot_subset(selected_labels, instances, pilot_seed)
    selected_instances = {label: instances[label] for label in selected_labels}
    coverage = _coverage_summary(structural_rows, selected_labels)

    payload = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_instances_draft_selected",
        "instance_selection_version": PHASE_E_INSTANCE_VERSION,
        "scope": "pre_pilot_depth2_instance_selection_from_phase_d_H1_H12",
        "official_execution_authorized": False,
        "selection_rule": "one eligible H-instance per Phase D stratum sampled uniformly with numpy.default_rng(PCG64) from labels sorted numerically within stratum",
        "selection_seed": selection_seed,
        "pilot_selection_rule": "one unweighted and one weighted selected instance sampled with a separate recorded seed when available",
        "pilot_seed": pilot_seed,
        "selected_depth2_instances": selected_labels,
        "pilot_depth2_instances": pilot_labels,
        "selected_instances": selected_instances,
        "all_phase_d_H1_H12_source": str(source_path.relative_to(root)),
        "source_hashes": {
            "phase_d_selected_instances_source": sha256_file(source_path),
        },
        "coverage_summary": coverage,
        "outputs": {
            "config": "config/phase_e_instances_v0.1-draft.json",
            "structural_diagnostics_csv": "results/phase_e/planning/phase_e_structural_diagnostics.csv",
            "lightcone_diagnostics_csv": "results/phase_e/planning/phase_e_lightcone_diagnostics.csv",
            "report_md": "results/phase_e/planning/phase_e_instance_selection_report.md",
            "hashes": "results/phase_e/planning/phase_e_instance_selection_hashes.json",
        },
        "freeze_notes": [
            "Draft selection is made before any Phase E depth-2 outcomes.",
            "Do not replace selected instances after inspecting pilot/full depth-2 outcomes except via a pre-specified mechanical failure rule.",
            "The unified pilot may use the recorded pilot subset but must not change the scientific hypothesis.",
        ],
    }
    write_json_atomic(config_path, payload, overwrite=True)
    _write_csv(structural_csv, structural_rows)
    _write_csv(lightcone_csv, lightcone_rows)
    report_md.write_text(_markdown_report(payload), encoding="utf-8")
    hashes = {
        "created_at_utc": _utc_now(),
        "instance_selection_version": PHASE_E_INSTANCE_VERSION,
        "hashes": {
            "phase_e_instances_config": sha256_file(config_path),
            "phase_e_structural_diagnostics_csv": sha256_file(structural_csv),
            "phase_e_lightcone_diagnostics_csv": sha256_file(lightcone_csv),
            "phase_e_instance_selection_report_md": sha256_file(report_md),
        },
    }
    write_json_atomic(hash_path, hashes, overwrite=True)
    payload["output_hashes"] = hashes["hashes"]
    return payload

