from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


GENERATOR_VERSION = "phase_d_stratified_generator_v1"
PLAN_STATUS = "phase_d_v0.9_draft_not_frozen_not_executed"
MASTER_SEED = 20260513
PLANNING_DIR = PROJECT_ROOT / "results" / "phase_d" / "planning"
SELECTED_PER_STRATUM = 2
RESERVES_PER_STRATUM = 2
MAX_ATTEMPTS_PER_STRATUM = 10000
MAX_NODES = 6
MAX_EDGES = 13
WEIGHT_LOW = 0.50
WEIGHT_HIGH = 1.50

STRATA: tuple[dict[str, Any], ...] = (
    {
        "stratum_id": "S1",
        "name": "continuity-like sparse",
        "selected_h": ["H1", "H7"],
        "technical_rationale": (
            "Tests whether the Phase C discordance pattern persists near the "
            "original small sparse benchmark regime while using newly generated graphs."
        ),
    },
    {
        "stratum_id": "S2",
        "name": "regular / near-regular",
        "selected_h": ["H2", "H8"],
        "technical_rationale": (
            "Connects Phase D to common MaxCut/QAOA regular-graph benchmark families "
            "and reduces degree heterogeneity as a confound."
        ),
    },
    {
        "stratum_id": "S3",
        "name": "Erdos-Renyi",
        "selected_h": ["H3", "H9"],
        "technical_rationale": (
            "Provides a standard random-topology baseline and varies connectivity "
            "without hand-designing graph topology."
        ),
    },
    {
        "stratum_id": "S4",
        "name": "weighted heterogeneous",
        "selected_h": ["H4", "H10"],
        "technical_rationale": (
            "Tests whether discordance depends on nonuniform positive edge costs "
            "rather than only unweighted topology."
        ),
    },
    {
        "stratum_id": "S5",
        "name": "dense / near-complete positive-weight",
        "selected_h": ["H5", "H11"],
        "technical_rationale": (
            "Stresses denser small-graph coupling while staying inside the current "
            "positive-weight MaxCut definition."
        ),
    },
    {
        "stratum_id": "S6",
        "name": "hardware-friendly / control",
        "selected_h": ["H6", "H12"],
        "technical_rationale": (
            "Tests low-degree path, ladder, or native-friendly topology without "
            "selecting instances by expected outcome."
        ),
    },
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    stratum_id: str
    stratum_name: str
    attempt_index: int
    stratum_seed: int
    n_nodes: int
    edges: tuple[tuple[int, int, float], ...]
    structural_pass: bool
    structural_fail_reason: str
    stratum_queue_rank: int | None = None
    selected_label: str = ""
    reserve_rank: int | None = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _stratum_seed(master_seed: int, stratum_id: str) -> int:
    payload = f"{master_seed}|{stratum_id}|{GENERATOR_VERSION}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _all_pairs(n_nodes: int) -> list[tuple[int, int]]:
    return [(u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes)]


def _canonical_edges(edges: Iterable[tuple[int, int, float]]) -> tuple[tuple[int, int, float], ...]:
    out = []
    for u, v, weight in edges:
        a, b = sorted((int(u), int(v)))
        out.append((a, b, round(float(weight), 2)))
    return tuple(sorted(out))


def _draw_weights(
    rng: np.random.Generator,
    pairs: Iterable[tuple[int, int]],
    mode: str,
) -> tuple[tuple[int, int, float], ...]:
    weighted_edges = []
    for u, v in pairs:
        if mode == "unweighted":
            weight = 1.0
        elif mode == "uniform_0p50_1p50":
            weight = round(float(rng.uniform(WEIGHT_LOW, WEIGHT_HIGH)), 2)
        else:
            raise ValueError(f"Unknown weight mode: {mode}")
        weighted_edges.append((u, v, weight))
    return _canonical_edges(weighted_edges)


def _degrees(n_nodes: int, edges: tuple[tuple[int, int, float], ...]) -> list[int]:
    degree = [0 for _ in range(n_nodes)]
    for u, v, _ in edges:
        degree[u] += 1
        degree[v] += 1
    return degree


def _connected(n_nodes: int, edges: tuple[tuple[int, int, float], ...]) -> bool:
    if n_nodes <= 0:
        return False
    adj = [[] for _ in range(n_nodes)]
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == n_nodes


def _random_tree_pairs(rng: np.random.Generator, n_nodes: int) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for node in range(1, n_nodes):
        parent = int(rng.integers(0, node))
        pairs.add(tuple(sorted((node, parent))))
    return pairs


def _sample_extra_pairs(
    rng: np.random.Generator,
    n_nodes: int,
    existing: set[tuple[int, int]],
    target_edges: int,
) -> set[tuple[int, int]]:
    pairs = set(existing)
    available = [pair for pair in _all_pairs(n_nodes) if pair not in pairs]
    need = max(0, target_edges - len(pairs))
    if need:
        chosen = rng.choice(len(available), size=need, replace=False)
        for idx in np.atleast_1d(chosen):
            pairs.add(available[int(idx)])
    return pairs


def _weight_mode_for_valid_rank(stratum_id: str, valid_rank: int | None) -> str:
    if stratum_id in {"S4", "S5"}:
        return "uniform_0p50_1p50"
    if stratum_id in {"S2", "S3"}:
        return "unweighted"
    if valid_rank is not None and valid_rank % 2 == 0:
        return "uniform_0p50_1p50"
    return "unweighted"


def _generate_topology_s1(rng: np.random.Generator) -> tuple[int, set[tuple[int, int]]]:
    n_nodes = int(rng.choice([4, 5, 6], p=[0.25, 1.0 / 3.0, 5.0 / 12.0]))
    target_edges = int(rng.integers(n_nodes - 1, n_nodes + 2))
    pairs = _sample_extra_pairs(rng, n_nodes, _random_tree_pairs(rng, n_nodes), target_edges)
    return n_nodes, pairs


def _generate_topology_s2(rng: np.random.Generator) -> tuple[int, set[tuple[int, int]]]:
    n_nodes = int(rng.choice([4, 6]))
    if n_nodes == 4:
        return n_nodes, set(_all_pairs(4))
    nodes = list(rng.permutation(6))
    if float(rng.random()) < 0.5:
        left = nodes[:3]
        right = nodes[3:]
        pairs = {tuple(sorted((u, v))) for u in left for v in right}
    else:
        top = nodes[:3]
        bottom = nodes[3:]
        pairs = {
            tuple(sorted((top[0], top[1]))),
            tuple(sorted((top[1], top[2]))),
            tuple(sorted((top[2], top[0]))),
            tuple(sorted((bottom[0], bottom[1]))),
            tuple(sorted((bottom[1], bottom[2]))),
            tuple(sorted((bottom[2], bottom[0]))),
            tuple(sorted((top[0], bottom[0]))),
            tuple(sorted((top[1], bottom[1]))),
            tuple(sorted((top[2], bottom[2]))),
        }
    return n_nodes, pairs


def _generate_topology_s3(rng: np.random.Generator) -> tuple[int, set[tuple[int, int]]]:
    n_nodes = int(rng.choice([4, 5, 6], p=[0.25, 1.0 / 3.0, 5.0 / 12.0]))
    p_edge = float(rng.uniform(0.30, 0.70))
    pairs = {pair for pair in _all_pairs(n_nodes) if float(rng.random()) < p_edge}
    return n_nodes, pairs


def _generate_topology_s4(rng: np.random.Generator) -> tuple[int, set[tuple[int, int]]]:
    n_nodes = int(rng.choice([4, 5, 6], p=[0.25, 1.0 / 3.0, 5.0 / 12.0]))
    complete_edges = len(_all_pairs(n_nodes))
    target_edges = int(rng.integers(n_nodes - 1, min(complete_edges, MAX_EDGES) + 1))
    chosen = rng.choice(complete_edges, size=target_edges, replace=False)
    pairs = {_all_pairs(n_nodes)[int(idx)] for idx in np.atleast_1d(chosen)}
    return n_nodes, pairs


def _generate_topology_s5(rng: np.random.Generator) -> tuple[int, set[tuple[int, int]]]:
    n_nodes = int(rng.choice([5, 6]))
    complete_pairs = _all_pairs(n_nodes)
    complete_edges = len(complete_pairs)
    min_edges = int(math.ceil(0.75 * complete_edges))
    target_edges = int(rng.integers(min_edges, min(complete_edges, MAX_EDGES) + 1))
    chosen = rng.choice(complete_edges, size=target_edges, replace=False)
    pairs = {complete_pairs[int(idx)] for idx in np.atleast_1d(chosen)}
    return n_nodes, pairs


def _generate_topology_s6(rng: np.random.Generator) -> tuple[int, set[tuple[int, int]]]:
    n_nodes = int(rng.choice([5, 6]))
    pairs = {tuple(sorted((i, i + 1))) for i in range(n_nodes - 1)}
    if n_nodes == 6 and float(rng.random()) < 0.5:
        pairs = {(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)}
    else:
        extras = [(0, 2), (1, 3), (2, 4), (3, 5)]
        rng.shuffle(extras)
        for u, v in extras:
            if u >= n_nodes or v >= n_nodes:
                continue
            trial = pairs | {tuple(sorted((u, v)))}
            if max(_degrees(n_nodes, _draw_weights(rng, trial, "unweighted"))) <= 3 and float(rng.random()) < 0.5:
                pairs = trial
    return n_nodes, pairs


def _generate_topology(stratum_id: str, rng: np.random.Generator) -> tuple[int, set[tuple[int, int]]]:
    if stratum_id == "S1":
        return _generate_topology_s1(rng)
    if stratum_id == "S2":
        return _generate_topology_s2(rng)
    if stratum_id == "S3":
        return _generate_topology_s3(rng)
    if stratum_id == "S4":
        return _generate_topology_s4(rng)
    if stratum_id == "S5":
        return _generate_topology_s5(rng)
    if stratum_id == "S6":
        return _generate_topology_s6(rng)
    raise ValueError(f"Unknown stratum_id: {stratum_id}")


def _structural_check(stratum_id: str, n_nodes: int, edges: tuple[tuple[int, int, float], ...]) -> tuple[bool, str]:
    n_edges = len(edges)
    if n_nodes < 4 or n_nodes > MAX_NODES:
        return False, "n_nodes_outside_envelope"
    if n_edges < n_nodes - 1:
        return False, "too_few_edges"
    if n_edges > MAX_EDGES:
        return False, "too_many_edges"
    if not _connected(n_nodes, edges):
        return False, "disconnected"
    weights = [w for _, _, w in edges]
    if any(weight < WEIGHT_LOW or weight > WEIGHT_HIGH for weight in weights):
        return False, "weight_outside_range"
    degrees = _degrees(n_nodes, edges)
    if stratum_id == "S1" and max(degrees) > 3:
        return False, "s1_max_degree_gt_3"
    if stratum_id == "S2" and max(degrees) - min(degrees) > 1:
        return False, "s2_degree_spread_gt_1"
    if stratum_id == "S6" and max(degrees) > 3:
        return False, "s6_max_degree_gt_3"
    return True, ""


def _graph_metrics(n_nodes: int, edges: tuple[tuple[int, int, float], ...]) -> dict[str, Any]:
    degrees = _degrees(n_nodes, edges)
    weights = [float(w) for _, _, w in edges]
    n_edges = len(edges)
    complete_edges = n_nodes * (n_nodes - 1) / 2.0
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(n_edges / complete_edges) if complete_edges else 0.0,
        "degree_min": min(degrees) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "degree_std": float(np.std(degrees)) if degrees else 0.0,
        "weighted_flag": any(abs(weight - 1.0) > 1e-12 for weight in weights),
        "weight_min": min(weights) if weights else 0.0,
        "weight_max": max(weights) if weights else 0.0,
        "weight_mean": float(np.mean(weights)) if weights else 0.0,
        "weight_std": float(np.std(weights)) if weights else 0.0,
    }


def _edge_payload(edges: tuple[tuple[int, int, float], ...]) -> list[list[float | int]]:
    return [[int(u), int(v), float(w)] for u, v, w in edges]


def _density_class(density: float) -> str:
    if density < 0.40:
        return "sparse"
    if density < 0.75:
        return "medium"
    return "dense"


def _topology_signature(edges: tuple[tuple[int, int, float], ...]) -> str:
    return _json_hash([[int(u), int(v)] for u, v, _ in edges])


def _weight_distribution_label(edges: tuple[tuple[int, int, float], ...]) -> str:
    weights = [float(w) for _, _, w in edges]
    if all(abs(weight - 1.0) <= 1e-12 for weight in weights):
        return "all_edges_1p00_unweighted"
    return "uniform_0p50_1p50_rounded_2dp_positive"


def _candidate_row(candidate: Candidate, master_seed: int) -> dict[str, Any]:
    metrics = _graph_metrics(candidate.n_nodes, candidate.edges)
    edge_payload = _edge_payload(candidate.edges)
    if not candidate.structural_pass:
        candidate_status = "rejected_structural"
    elif candidate.selected_label:
        candidate_status = "selected"
    elif candidate.reserve_rank is not None:
        candidate_status = "reserve"
    else:
        candidate_status = "valid_queue_extra"
    rejection_reason = "" if candidate.structural_pass else candidate.structural_fail_reason
    weighted = bool(metrics["weighted_flag"])
    instance_signature = _json_hash(edge_payload)
    row = {
        "candidate_id": candidate.candidate_id,
        "stratum_id": candidate.stratum_id,
        "stratum_name": candidate.stratum_name,
        "instance_name": candidate.selected_label or candidate.candidate_id,
        "attempt_index": candidate.attempt_index,
        "master_seed": master_seed,
        "stratum_seed": candidate.stratum_seed,
        "candidate_status": candidate_status,
        "rejection_reason": rejection_reason,
        "structural_pass": candidate.structural_pass,
        "structural_fail_reason": candidate.structural_fail_reason,
        "stratum_queue_rank": "" if candidate.stratum_queue_rank is None else candidate.stratum_queue_rank,
        "selected_label": candidate.selected_label,
        "reserve_rank": "" if candidate.reserve_rank is None else candidate.reserve_rank,
        "replacement_for": "",
        "grid_adequacy_status": "pending_grid_adequacy_pre_gate",
        "weighted": weighted,
        "weight_distribution": _weight_distribution_label(candidate.edges),
        "density_class": _density_class(float(metrics["density"])),
        "graph_density": metrics["density"],
        "max_degree": metrics["degree_max"],
        "topology_signature": _topology_signature(candidate.edges),
        "instance_signature": instance_signature,
        "graph_signature": instance_signature,
        "edge_list_json": _canonical_json(edge_payload),
        "edges_json": _canonical_json(edge_payload),
    }
    row.update(metrics)
    return row


CANDIDATE_QUEUE_FIELDS = [
    "candidate_id",
    "stratum_id",
    "stratum_name",
    "instance_name",
    "attempt_index",
    "master_seed",
    "stratum_seed",
    "candidate_status",
    "rejection_reason",
    "structural_pass",
    "structural_fail_reason",
    "stratum_queue_rank",
    "selected_label",
    "reserve_rank",
    "replacement_for",
    "grid_adequacy_status",
    "n_nodes",
    "n_edges",
    "weighted",
    "weight_distribution",
    "density_class",
    "graph_density",
    "density",
    "degree_min",
    "degree_max",
    "max_degree",
    "degree_mean",
    "degree_std",
    "weighted_flag",
    "weight_min",
    "weight_max",
    "weight_mean",
    "weight_std",
    "topology_signature",
    "instance_signature",
    "graph_signature",
    "edge_list_json",
    "edges_json",
]


def _candidate_queue_csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CANDIDATE_QUEUE_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite CSV: {path}")
    path.write_bytes(_candidate_queue_csv_text(rows).encode("utf-8"))


def _generator_config(master_seed: int) -> dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "plan_status": PLAN_STATUS,
        "master_seed": master_seed,
        "rng": "numpy.random.Generator(PCG64)",
        "selection_rule": "first_2_structurally_valid_candidates_per_stratum",
        "reserve_rule": "next_2_structurally_valid_candidates_per_stratum",
        "selected_per_stratum": SELECTED_PER_STRATUM,
        "reserves_per_stratum": RESERVES_PER_STRATUM,
        "max_attempts_per_stratum": MAX_ATTEMPTS_PER_STRATUM,
        "envelope": {
            "n_nodes": [4, 5, 6],
            "max_n_nodes": MAX_NODES,
            "max_edges": MAX_EDGES,
            "connected_required": True,
            "unique_graph_signatures_across_valid_queue": True,
            "weight_range": [WEIGHT_LOW, WEIGHT_HIGH],
            "weight_rounding_decimals": 2,
            "positive_weights_only": True,
        },
        "strata": [
            {
                **stratum,
                "stratum_seed": _stratum_seed(master_seed, str(stratum["stratum_id"])),
            }
            for stratum in STRATA
        ],
        "anti_selection_rules": [
            "no_diversity_score",
            "no_selection_score",
            "no_tie_breakers",
            "no_similarity_penalty_to_phase_c",
            "no_manual_reordering",
            "no_landscape_shape_filter",
            "no_noisy_or_zne_outcome_filter",
        ],
    }


def generate_phase_d_candidates(master_seed: int = MASTER_SEED) -> dict[str, Any]:
    config = _generator_config(master_seed)
    candidates: list[Candidate] = []
    valid_by_stratum: dict[str, list[Candidate]] = {}
    accepted_signatures: set[str] = set()

    for stratum in STRATA:
        stratum_id = str(stratum["stratum_id"])
        stratum_name = str(stratum["name"])
        stratum_seed = _stratum_seed(master_seed, stratum_id)
        rng = np.random.Generator(np.random.PCG64(stratum_seed))
        valid_candidates: list[Candidate] = []
        attempt = 1
        while len(valid_candidates) < SELECTED_PER_STRATUM + RESERVES_PER_STRATUM:
            if attempt > MAX_ATTEMPTS_PER_STRATUM:
                raise RuntimeError(f"{stratum_id} exhausted max attempts before enough valid candidates.")
            n_nodes, pairs = _generate_topology(stratum_id, rng)
            provisional_edges = _draw_weights(rng, pairs, "unweighted")
            provisional_pass, provisional_reason = _structural_check(stratum_id, n_nodes, provisional_edges)
            valid_rank = len(valid_candidates) + 1 if provisional_pass else None
            weight_mode = _weight_mode_for_valid_rank(stratum_id, valid_rank)
            edges = _draw_weights(rng, pairs, weight_mode)
            structural_pass, fail_reason = _structural_check(stratum_id, n_nodes, edges)
            if provisional_pass and not structural_pass:
                valid_rank = None
            graph_signature = _json_hash(_edge_payload(edges))
            if structural_pass and graph_signature in accepted_signatures:
                structural_pass = False
                fail_reason = "duplicate_valid_graph_signature_global"
                valid_rank = None
            candidate = Candidate(
                candidate_id=f"{stratum_id}_C{attempt:06d}",
                stratum_id=stratum_id,
                stratum_name=stratum_name,
                attempt_index=attempt,
                stratum_seed=stratum_seed,
                n_nodes=n_nodes,
                edges=edges,
                structural_pass=structural_pass,
                structural_fail_reason=fail_reason,
                stratum_queue_rank=valid_rank if structural_pass else None,
            )
            if structural_pass:
                valid_candidates.append(candidate)
                accepted_signatures.add(graph_signature)
            candidates.append(candidate)
            attempt += 1
        valid_by_stratum[stratum_id] = valid_candidates

    selected_lookup: dict[str, str] = {}
    reserve_lookup: dict[str, int] = {}
    selected_instances: dict[str, dict[str, Any]] = {}
    reserves: dict[str, list[dict[str, Any]]] = {}

    h_index = 1
    for valid_rank in range(1, SELECTED_PER_STRATUM + 1):
        for stratum in STRATA:
            stratum_id = str(stratum["stratum_id"])
            candidate = valid_by_stratum[stratum_id][valid_rank - 1]
            label = f"H{h_index}"
            selected_lookup[candidate.candidate_id] = label
            selected_instances[label] = {
                "instance_id": label,
                "candidate_id": candidate.candidate_id,
                "stratum_id": candidate.stratum_id,
                "stratum_name": candidate.stratum_name,
                "stratum_queue_rank": candidate.stratum_queue_rank,
                "n_nodes": candidate.n_nodes,
                "edges": _edge_payload(candidate.edges),
                "graph_signature": _json_hash(_edge_payload(candidate.edges)),
            }
            h_index += 1

    for stratum in STRATA:
        stratum_id = str(stratum["stratum_id"])
        reserves[stratum_id] = []
        for reserve_rank, candidate in enumerate(valid_by_stratum[stratum_id][SELECTED_PER_STRATUM:], start=1):
            reserve_lookup[candidate.candidate_id] = reserve_rank
            reserves[stratum_id].append({
                "reserve_rank": reserve_rank,
                "candidate_id": candidate.candidate_id,
                "stratum_queue_rank": candidate.stratum_queue_rank,
                "n_nodes": candidate.n_nodes,
                "edges": _edge_payload(candidate.edges),
                "graph_signature": _json_hash(_edge_payload(candidate.edges)),
            })

    enriched: list[Candidate] = []
    for candidate in candidates:
        enriched.append(
            Candidate(
                **{
                    **candidate.__dict__,
                    "selected_label": selected_lookup.get(candidate.candidate_id, ""),
                    "reserve_rank": reserve_lookup.get(candidate.candidate_id),
                }
            )
        )

    rows = [_candidate_row(candidate, master_seed) for candidate in enriched]
    stratum_queues = {
        stratum_id: {
            "valid_candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "stratum_queue_rank": candidate.stratum_queue_rank,
                    "selected_label": selected_lookup.get(candidate.candidate_id, ""),
                    "reserve_rank": reserve_lookup.get(candidate.candidate_id),
                    "graph_signature": _json_hash(_edge_payload(candidate.edges)),
                    "edges": _edge_payload(candidate.edges),
                }
                for candidate in valid_candidates
            ]
        }
        for stratum_id, valid_candidates in valid_by_stratum.items()
    }
    weight_manifest = {
        "generator_version": GENERATOR_VERSION,
        "weight_rules": {
            "S1": "odd valid queue ranks unweighted; even valid queue ranks Uniform(0.50,1.50)",
            "S2": "unweighted",
            "S3": "unweighted",
            "S4": "Uniform(0.50,1.50)",
            "S5": "Uniform(0.50,1.50)",
            "S6": "odd valid queue ranks unweighted; even valid queue ranks Uniform(0.50,1.50)",
        },
        "rounding_decimals": 2,
        "positive_weights_only": True,
    }
    summary = {
        "created_at_utc": _utc_now(),
        "generator_version": GENERATOR_VERSION,
        "plan_status": PLAN_STATUS,
        "master_seed": master_seed,
        "candidate_count": len(rows),
        "structural_pass_count": sum(1 for row in rows if row["structural_pass"]),
        "selected_count": len(selected_instances),
        "reserve_count": sum(len(items) for items in reserves.values()),
        "selected_instances": selected_instances,
        "reserves": reserves,
        "config_hash": _json_hash(config),
        "selected_instances_hash": _json_hash(selected_instances),
        "stratum_queues_hash": _json_hash(stratum_queues),
        "weight_manifest_hash": _json_hash(weight_manifest),
    }
    return {
        "config": config,
        "rows": rows,
        "stratum_queues": stratum_queues,
        "selected_instances": selected_instances,
        "weight_manifest": weight_manifest,
        "summary": summary,
    }


def write_phase_d_generator_outputs(
    root: Path = PROJECT_ROOT,
    master_seed: int = MASTER_SEED,
    force: bool = False,
) -> dict[str, Any]:
    generated = generate_phase_d_candidates(master_seed)
    output_dir = ensure_dir(root / "results" / "phase_d" / "planning")
    paths = {
        "config": output_dir / "phase_d_generator_config.json",
        "seed": output_dir / "phase_d_generator_seed.txt",
        "candidate_queue": output_dir / "phase_d_candidate_queue.csv",
        "stratum_queues": output_dir / "phase_d_stratum_queues.json",
        "selected_instances": output_dir / "phase_d_selected_instances.json",
        "weight_manifest": output_dir / "phase_d_weight_generation_manifest.json",
        "summary": output_dir / "phase_d_generator_summary.json",
    }
    if not force:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing generator outputs: " + ", ".join(str(p) for p in existing))

    write_json_atomic(paths["config"], generated["config"], overwrite=True)
    paths["seed"].write_text(f"{master_seed}\n", encoding="utf-8")
    _write_csv(paths["candidate_queue"], generated["rows"], overwrite=True)
    write_json_atomic(paths["stratum_queues"], generated["stratum_queues"], overwrite=True)
    write_json_atomic(paths["selected_instances"], generated["selected_instances"], overwrite=True)
    write_json_atomic(paths["weight_manifest"], generated["weight_manifest"], overwrite=True)

    summary = dict(generated["summary"])
    summary["output_paths"] = {key: path.relative_to(root).as_posix() for key, path in paths.items() if key != "summary"}
    summary["candidate_queue_hash"] = sha256_file(paths["candidate_queue"])
    summary["generator_config_file_hash"] = sha256_file(paths["config"])
    summary["stratum_queues_file_hash"] = sha256_file(paths["stratum_queues"])
    summary["selected_instances_file_hash"] = sha256_file(paths["selected_instances"])
    summary["weight_manifest_file_hash"] = sha256_file(paths["weight_manifest"])
    write_json_atomic(paths["summary"], summary, overwrite=True)
    return summary


def verify_phase_d_generator_reproducibility(
    root: Path = PROJECT_ROOT,
    master_seed: int = MASTER_SEED,
    write_report: bool = True,
) -> dict[str, Any]:
    output_dir = root / "results" / "phase_d" / "planning"
    existing_summary_path = output_dir / "phase_d_generator_summary.json"
    if not existing_summary_path.is_file():
        raise FileNotFoundError(f"Missing generator summary: {existing_summary_path}")
    existing = json.loads(existing_summary_path.read_text(encoding="utf-8-sig"))
    regenerated = generate_phase_d_candidates(master_seed)
    regenerated_candidate_csv = _candidate_queue_csv_text(regenerated["rows"])
    hash_pairs = {
        "generator_config": {
            "original_hash": existing.get("config_hash"),
            "regenerated_hash": _json_hash(regenerated["config"]),
        },
        "candidate_queue_file": {
            "original_hash": existing.get("candidate_queue_hash"),
            "regenerated_hash": hashlib.sha256(regenerated_candidate_csv.encode("utf-8")).hexdigest(),
        },
        "selected_instances": {
            "original_hash": existing.get("selected_instances_hash"),
            "regenerated_hash": _json_hash(regenerated["selected_instances"]),
        },
        "stratum_queues": {
            "original_hash": existing.get("stratum_queues_hash"),
            "regenerated_hash": _json_hash(regenerated["stratum_queues"]),
        },
        "weight_manifest": {
            "original_hash": existing.get("weight_manifest_hash"),
            "regenerated_hash": _json_hash(regenerated["weight_manifest"]),
        },
    }
    for pair in hash_pairs.values():
        pair["match"] = pair["original_hash"] == pair["regenerated_hash"]
    checks = {
        "config_hash_match": bool(hash_pairs["generator_config"]["match"]),
        "candidate_queue_hash_match": bool(hash_pairs["candidate_queue_file"]["match"]),
        "selected_instances_hash_match": bool(hash_pairs["selected_instances"]["match"]),
        "stratum_queues_hash_match": bool(hash_pairs["stratum_queues"]["match"]),
        "weight_manifest_hash_match": bool(hash_pairs["weight_manifest"]["match"]),
        "candidate_count_match": int(existing.get("candidate_count", -1)) == len(regenerated["rows"]),
        "selected_count_match": int(existing.get("selected_count", -1)) == len(regenerated["selected_instances"]),
    }
    report = {
        "created_at_utc": _utc_now(),
        "generator_version": GENERATOR_VERSION,
        "plan_status": PLAN_STATUS,
        "master_seed": master_seed,
        "hash_pairs": hash_pairs,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    if write_report:
        write_json_atomic(output_dir / "phase_d_generator_reproducibility.json", report, overwrite=True)
    return report
