from __future__ import annotations

import csv
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import GraphInstance, cut_value
from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .mitiq_wrapper import deterministic_seed, richardson_weights
from .phase_e_c0_jitter import DEFAULT_C0_JITTER_SCALE_FACTORS, _ideal_reference_range, _load_graphs
from .phase_e_optimizers import SPSAConfig, make_paired_initial_thetas, spsa_minimize
from .phase_e_sim_backend import evaluate_custom_density_objective, richardson_zne_objective


PHASE_E_C0_JITTER_ATTRIBUTION_VERSION = "phase_e_c0_jitter_attribution_v0_1_draft"
PLAN_PATH = Path("PHASE_E_C0_JITTER_ATTRIBUTION_PLAN_v0.1-draft.md")
EXACT_CUT_ENUMERATION_QUBIT_CEILING = 20
EXACT_CUT_ENUMERATION_SECONDS_PER_GRAPH = 60.0
JITTER_STABLE_THRESHOLD = 0.010
JITTER_UNSTABLE_THRESHOLD = 0.050
PHASE_B_DEFAULT_CASES = (("H4", 0.02), ("H4", 0.003))
PHASE_B_DEFAULT_MODES = (
    "ideal_deterministic",
    "noisy_deterministic",
    "noisy_sampled",
    "zne_deterministic",
    "zne_sampled",
)
PHASE_B_DEFAULT_DUPLICATE_SEEDS = (101, 202, 303, 404, 505)
PHASE_B_DEFAULT_SHOTS = 512
PHASE_B_DEFAULT_T_BUDGET = 250
PHASE_B_DEFAULT_STARTS = 20
PHASE_B_DEFAULT_BASE_SEED = 42
PHASE_B_PRIMARY_REFERENCE_POINTS = 1024


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    return float(value)


def _classify_jitter(value: float) -> str:
    if not math.isfinite(value):
        return "nonfinite"
    if value < JITTER_STABLE_THRESHOLD:
        return "stable"
    if value < JITTER_UNSTABLE_THRESHOLD:
        return "marginal"
    return "unstable"


def _summary_stats(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {
            "count": 0,
            "p50": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "max": math.nan,
            "classification": "nonfinite",
        }
    p95 = float(np.percentile(finite, 95))
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": p95,
        "max": float(np.max(finite)),
        "classification": _classify_jitter(p95),
    }


def _case_key(row: dict[str, Any]) -> tuple[str, float, int, int]:
    return (
        str(row["instance"]),
        float(row["p_dep"]),
        int(row["shots_total"]),
        int(row["T_requested"]),
    )


def _exact_cut_range(graph: GraphInstance) -> dict[str, Any]:
    n = int(graph.n_qubits)
    if n > EXACT_CUT_ENUMERATION_QUBIT_CEILING:
        return {
            "exact_cut_status": "not_computed_qubit_ceiling_exceeded",
            "exact_cut_min": math.nan,
            "exact_cut_max": math.nan,
            "exact_cut_range": math.nan,
            "exact_cut_bitstrings": 0,
            "exact_cut_elapsed_seconds": 0.0,
        }

    started = time.perf_counter()
    values: list[float] = []
    for bits in itertools.product((0, 1), repeat=n):
        values.append(float(cut_value(bits, graph.edges)))
        if time.perf_counter() - started > EXACT_CUT_ENUMERATION_SECONDS_PER_GRAPH:
            return {
                "exact_cut_status": "timed_out",
                "exact_cut_min": math.nan,
                "exact_cut_max": math.nan,
                "exact_cut_range": math.nan,
                "exact_cut_bitstrings": len(values),
                "exact_cut_elapsed_seconds": time.perf_counter() - started,
            }

    cut_min = float(min(values))
    cut_max = float(max(values))
    return {
        "exact_cut_status": "computed",
        "exact_cut_min": cut_min,
        "exact_cut_max": cut_max,
        "exact_cut_range": float(cut_max - cut_min),
        "exact_cut_bitstrings": len(values),
        "exact_cut_elapsed_seconds": time.perf_counter() - started,
    }


def _git_status(root: Path) -> dict[str, Any]:
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {"git_head": head, "git_status_short": status, "git_status_short_sha256": _sha256_text(status)}
    except Exception as exc:
        return {"git_head": None, "git_status_short": f"git_status_unavailable: {exc}", "git_status_short_sha256": None}


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for name in ("scipy", "pandas", "qiskit"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = None
    return versions


def _build_reference_ranges(
    root: Path,
    endpoints: list[dict[str, str]],
    progress: bool,
) -> tuple[dict[tuple[str, float, int, int], dict[str, Any]], dict[str, dict[str, Any]], str]:
    labels = tuple(sorted({str(row["instance"]) for row in endpoints}))
    graphs, _, instances_hash = _load_graphs(root, labels)
    by_label = {graph.name: graph for graph in graphs}
    exact = {label: _exact_cut_range(graph) for label, graph in by_label.items()}

    ranges: dict[tuple[str, float, int, int], dict[str, Any]] = {}
    keys = sorted({_case_key(row) for row in endpoints})
    for index, key in enumerate(keys, start=1):
        instance, p_dep, shots, t_budget = key
        if progress:
            print(f"[jitter-attribution] reference ranges {index}/{len(keys)} {instance} p={p_dep} shots={shots} T={t_budget}", flush=True)
        graph = by_label[instance]
        r512 = _ideal_reference_range(graph, p_dep, shots, t_budget, 512)
        r1024 = _ideal_reference_range(graph, p_dep, shots, t_budget, 1024)
        ranges[key] = {
            "range_512_reference_points": 512,
            "range_512_min": r512["ideal_reference_min"],
            "range_512_max": r512["ideal_reference_max"],
            "range_512_reference": r512["ideal_objective_range"],
            "range_1024_reference_points": 1024,
            "range_1024_min": r1024["ideal_reference_min"],
            "range_1024_max": r1024["ideal_reference_max"],
            "range_1024_reference": r1024["ideal_objective_range"],
            **exact[instance],
        }
    return ranges, exact, instances_hash


def _renormalized_pairwise_rows(
    pairwise_rows: list[dict[str, str]],
    endpoints: list[dict[str, str]],
    ranges: dict[tuple[str, float, int, int], dict[str, Any]],
    denominator: str,
) -> list[dict[str, Any]]:
    endpoint_by_case = {str(row["case_id"]): row for row in endpoints}
    out: list[dict[str, Any]] = []
    for row in pairwise_rows:
        case_id = str(row["case_id"])
        endpoint = endpoint_by_case[case_id]
        key = _case_key(endpoint)
        ref = ranges[key]
        denom = float(ref[denominator])
        abs_diff = float(row["absolute_ideal_regret_difference"])
        fraction = abs_diff / denom if denom > 0.0 and math.isfinite(denom) else math.nan
        out.append(
            {
                **row,
                "original_jitter_fraction_of_ideal_range": row["jitter_fraction_of_ideal_range"],
                "renormalized_denominator_name": denominator,
                "renormalized_denominator": denom,
                "renormalized_jitter_fraction": fraction,
                "renormalized_jitter_classification": _classify_jitter(fraction),
                "range_32_existing": endpoint["ideal_objective_range"],
                "range_512_reference": ref["range_512_reference"],
                "range_1024_reference": ref["range_1024_reference"],
                "exact_cut_range": ref["exact_cut_range"],
                "exact_cut_status": ref["exact_cut_status"],
            }
        )
    return out


def _summary_rows(rows_by_name: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for denominator, rows in rows_by_name.items():
        values = [float(row["renormalized_jitter_fraction"]) for row in rows]
        stats = _summary_stats(values)
        out.append({"scope": "overall", "key": "all", "denominator": denominator, **stats})
        for field in ("instance", "objective_family", "p_dep"):
            for value in sorted({str(row[field]) for row in rows}):
                subset = [float(row["renormalized_jitter_fraction"]) for row in rows if str(row[field]) == value]
                out.append({"scope": field, "key": value, "denominator": denominator, **_summary_stats(subset)})
    return out


def _variance(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return 0.0
    return float(np.var(np.asarray(finite, dtype=float), ddof=1))


def _objective_for_mode(
    graph: GraphInstance,
    mode: str,
    p_dep: float,
    shots: int,
    base_seed: int,
    t_budget: int,
) -> Any:
    objective_seed = deterministic_seed(
        "phase_e_c0_jitter_attribution_fixed_objective",
        graph.name,
        p_dep,
        shots,
        base_seed,
        mode,
        t_budget,
    )

    def theta_seed(label: str, theta: tuple[float, ...]) -> int:
        return deterministic_seed(
            label,
            graph.name,
            p_dep,
            shots,
            objective_seed,
            tuple(round(float(x), 12) for x in theta),
        )

    if mode == "ideal_deterministic":
        return lambda theta: evaluate_custom_density_objective(graph, theta, p_dep=0.0, shots=None).objective_value
    if mode == "noisy_deterministic":
        return lambda theta: evaluate_custom_density_objective(graph, theta, p_dep=p_dep, shots=None).objective_value
    if mode == "noisy_sampled":
        return lambda theta: evaluate_custom_density_objective(
            graph,
            theta,
            p_dep=p_dep,
            shots=shots,
            seed=theta_seed("phase_e_c0_jitter_attribution_noisy_sampled", theta),
        ).objective_value
    if mode == "zne_deterministic":
        return lambda theta: richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=DEFAULT_C0_JITTER_SCALE_FACTORS,
            shots_total=None,
        ).objective_value
    if mode == "zne_sampled":
        return lambda theta: richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=DEFAULT_C0_JITTER_SCALE_FACTORS,
            shots_total=shots,
            seed=theta_seed("phase_e_c0_jitter_attribution_zne_sampled", theta),
        ).objective_value
    raise ValueError(f"Unsupported Phase B diagnostic mode: {mode}")


def _diagnostic_case_id(
    instance: str,
    p_dep: float,
    mode: str,
    shots: int,
    base_seed: int,
    t_budget: int,
    starts: int,
) -> str:
    return f"{instance}|p={p_dep}|mode={mode}|shots={shots}|base_seed={base_seed}|T={t_budget}|K={starts}"


def _spsa_gain_schedule_summary(config: SPSAConfig, t_budget: int) -> dict[str, float | int | bool]:
    evaluations = 1
    iterations = max(0, (int(t_budget) - evaluations) // 3)
    if iterations <= 0:
        ak = [float(config.a)]
        ck = [float(config.c)]
    else:
        ak = [float(config.a / ((k + 1 + config.stability_constant) ** config.alpha)) for k in range(iterations)]
        ck = [float(config.c / ((k + 1) ** config.gamma)) for k in range(iterations)]
    return {
        "a": float(config.a),
        "c": float(config.c),
        "alpha": float(config.alpha),
        "gamma": float(config.gamma),
        "stability_constant_A": float(config.stability_constant),
        "blocking": bool(config.blocking),
        "iterations": int(iterations),
        "ak_first": float(ak[0]),
        "ak_median": float(np.median(np.asarray(ak, dtype=float))),
        "ak_final": float(ak[-1]),
        "ck_first": float(ck[0]),
        "ck_median": float(np.median(np.asarray(ck, dtype=float))),
        "ck_final": float(ck[-1]),
    }


def _phase_b_reference_ranges(
    root: Path,
    cases: tuple[tuple[str, float], ...],
    shots: int,
    t_budget: int,
    progress: bool,
) -> tuple[dict[tuple[str, float], dict[str, Any]], dict[str, GraphInstance], str]:
    labels = tuple(sorted({case[0] for case in cases}))
    graphs, _, instances_hash = _load_graphs(root, labels)
    by_label = {graph.name: graph for graph in graphs}
    out: dict[tuple[str, float], dict[str, Any]] = {}
    for index, (instance, p_dep) in enumerate(cases, start=1):
        if progress:
            print(f"[jitter-attribution] phase B reference range {index}/{len(cases)} {instance} p={p_dep}", flush=True)
        graph = by_label[instance]
        r1024 = _ideal_reference_range(graph, float(p_dep), int(shots), int(t_budget), PHASE_B_PRIMARY_REFERENCE_POINTS)
        out[(instance, float(p_dep))] = {
            "range_1024_reference_points": PHASE_B_PRIMARY_REFERENCE_POINTS,
            "range_1024_min": r1024["ideal_reference_min"],
            "range_1024_max": r1024["ideal_reference_max"],
            "range_1024_reference": r1024["ideal_objective_range"],
            **_exact_cut_range(graph),
        }
    return out, by_label, instances_hash


def _simple_clusters(values: list[float], threshold: float) -> list[list[int]]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    if not indexed:
        return []
    clusters: list[list[int]] = [[indexed[0][0]]]
    previous = float(indexed[0][1])
    for original_index, value in indexed[1:]:
        if abs(float(value) - previous) < float(threshold):
            clusters[-1].append(original_index)
        else:
            clusters.append([original_index])
        previous = float(value)
    return clusters


def _bootstrap_p95_ci(values: list[float], seed: int) -> dict[str, float | int]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {"bootstrap_count": 0, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(int(seed))
    boot = []
    for _ in range(1000):
        sample = rng.choice(finite, size=finite.size, replace=True)
        boot.append(float(np.percentile(sample, 95)))
    return {
        "bootstrap_count": 1000,
        "ci_low": float(np.percentile(np.asarray(boot, dtype=float), 5)),
        "ci_high": float(np.percentile(np.asarray(boot, dtype=float), 95)),
    }


def _ci_overlap_fraction(a: dict[str, float], b: dict[str, float]) -> float:
    low = max(float(a["ci_low"]), float(b["ci_low"]))
    high = min(float(a["ci_high"]), float(b["ci_high"]))
    overlap = max(0.0, high - low)
    width_a = max(0.0, float(a["ci_high"]) - float(a["ci_low"]))
    width_b = max(0.0, float(b["ci_high"]) - float(b["ci_low"]))
    denom = min(width_a, width_b)
    return overlap / denom if denom > 0.0 else 0.0


def run_phase_e_c0_jitter_attribution_phase_a(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_jitter_attribution")
    input_dir = root / "results" / "phase_e" / "calibration" / "c0_jitter"
    paths = {
        "renormalized_512": output_dir / "existing_endpoint_jitter_renormalized_512.csv",
        "renormalized_1024": output_dir / "existing_endpoint_jitter_renormalized_1024.csv",
        "renormalized_exact_cut_range": output_dir / "existing_endpoint_jitter_renormalized_exact_cut_range.csv",
        "summary": output_dir / "existing_endpoint_renormalized_summary.json",
        "summary_table": output_dir / "existing_endpoint_renormalized_summary.csv",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E C0 jitter attribution Phase A outputs.")

    endpoints_path = input_dir / "duplicate_endpoint_jitter_endpoints.csv"
    pairwise_path = input_dir / "duplicate_endpoint_jitter_pairwise.csv"
    endpoints = _read_csv(endpoints_path)
    pairwise = _read_csv(pairwise_path)

    started = time.perf_counter()
    ranges, exact_cut_ranges, instances_hash = _build_reference_ranges(root, endpoints, progress=progress)
    rows_512 = _renormalized_pairwise_rows(pairwise, endpoints, ranges, "range_512_reference")
    rows_1024 = _renormalized_pairwise_rows(pairwise, endpoints, ranges, "range_1024_reference")
    rows_exact = _renormalized_pairwise_rows(pairwise, endpoints, ranges, "exact_cut_range")
    rows_by_name = {
        "range_512_reference": rows_512,
        "range_1024_reference": rows_1024,
        "exact_cut_range": rows_exact,
    }
    summary_table = _summary_rows(rows_by_name)

    _write_csv(paths["renormalized_512"], rows_512)
    _write_csv(paths["renormalized_1024"], rows_1024)
    _write_csv(paths["renormalized_exact_cut_range"], rows_exact)
    _write_csv(paths["summary_table"], summary_table)

    overall = {row["denominator"]: row for row in summary_table if row["scope"] == "overall" and row["key"] == "all"}
    p95_1024 = float(overall["range_1024_reference"]["p95"])
    p95_exact = float(overall["exact_cut_range"]["p95"])
    if p95_1024 < JITTER_STABLE_THRESHOLD and p95_exact < JITTER_STABLE_THRESHOLD:
        phase_a_decision = "dominant_cause_candidate_denominator_artifact"
    elif p95_1024 >= JITTER_UNSTABLE_THRESHOLD:
        phase_a_decision = "denominator_artifact_insufficient_to_explain_block"
    else:
        phase_a_decision = "denominator_artifact_material_but_not_decisive"

    plan_abs = root / PLAN_PATH
    code_paths = [
        root / "src" / "phase_e_c0_jitter_attribution.py",
        root / "scripts" / "65_phase_e_c0_jitter_attribution.py",
    ]
    input_hashes = {
        str(endpoints_path.relative_to(root)).replace("\\", "/"): sha256_file(endpoints_path),
        str(pairwise_path.relative_to(root)).replace("\\", "/"): sha256_file(pairwise_path),
        str((root / "config" / "phase_e_instances_v0.1-draft.json").relative_to(root)).replace("\\", "/"): instances_hash,
    }
    if plan_abs.exists():
        input_hashes[str(PLAN_PATH).replace("\\", "/")] = sha256_file(plan_abs)
    code_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in code_paths
        if path.exists()
    }

    finite_values = [float(row["renormalized_jitter_fraction"]) for rows in rows_by_name.values() for row in rows]
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_c0_jitter_attribution_phase_a_complete",
        "scope": "diagnostic_existing_endpoint_renormalization_not_protocol_amendment",
        "attribution_version": PHASE_E_C0_JITTER_ATTRIBUTION_VERSION,
        "diagnostic_only": True,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "thresholds_changed": False,
        "paper_scope_changed": False,
        "final_hyperparameter_selection": False,
        "phase_a_decision": phase_a_decision,
        "original_overall_p95": _summary_stats([float(row["jitter_fraction_of_ideal_range"]) for row in pairwise])["p95"],
        "range_512_overall": overall["range_512_reference"],
        "range_1024_overall": overall["range_1024_reference"],
        "exact_cut_range_overall": overall["exact_cut_range"],
        "exact_cut_ranges_by_instance": exact_cut_ranges,
        "case_count": len({row["case_id"] for row in endpoints}),
        "endpoint_rows": len(endpoints),
        "pairwise_rows": len(pairwise),
        "all_checked_numeric_finite": all(math.isfinite(value) for value in finite_values),
        "elapsed_seconds": time.perf_counter() - started,
        "time_box_seconds": 3600,
        "time_box_status": "within_time_box" if time.perf_counter() - started <= 3600 else "exceeded_time_box",
        "input_hashes": input_hashes,
        "code_hashes": code_hashes,
        "git": _git_status(root),
        "key_library_versions": _library_versions(),
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "limitations": [
            "This is a diagnostic renormalization of existing endpoints only.",
            "It does not modify the completed official C0 jitter gate result.",
            "Exact cut range is an upper-bound/control denominator, not automatically a replacement for ideal QAOA reference range.",
        ],
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    return summary


def run_phase_e_c0_jitter_attribution_phase_b(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    progress: bool = True,
    cases: tuple[tuple[str, float], ...] = PHASE_B_DEFAULT_CASES,
    modes: tuple[str, ...] = PHASE_B_DEFAULT_MODES,
    duplicate_seeds: tuple[int, ...] = PHASE_B_DEFAULT_DUPLICATE_SEEDS,
    t_budget: int = PHASE_B_DEFAULT_T_BUDGET,
    starts: int = PHASE_B_DEFAULT_STARTS,
    shots: int = PHASE_B_DEFAULT_SHOTS,
    base_seed: int = PHASE_B_DEFAULT_BASE_SEED,
    output_tag: str = "",
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_jitter_attribution")
    if output_tag:
        output_dir = ensure_dir(output_dir / output_tag)
    paths = {
        "all_start_endpoints": output_dir / "diagnostic_all_start_endpoints.csv",
        "best_of_k_pairwise": output_dir / "diagnostic_best_of_k_pairwise.csv",
        "jitter_summary": output_dir / "diagnostic_jitter_by_mode_summary.json",
        "jitter_summary_table": output_dir / "diagnostic_jitter_by_mode_summary.csv",
        "basin_proxy": output_dir / "diagnostic_basin_proxy_summary.csv",
        "basin_convergence": output_dir / "diagnostic_basin_convergence.csv",
        "variance_decomposition": output_dir / "diagnostic_variance_decomposition.csv",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E C0 jitter attribution Phase B outputs.")

    started = time.perf_counter()
    reference, graphs, instances_hash = _phase_b_reference_ranges(root, cases, shots, t_budget, progress=progress)
    config = SPSAConfig(max_evaluations=int(t_budget))
    gain_schedule = _spsa_gain_schedule_summary(config, int(t_budget))

    all_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    total_groups = len(cases) * len(modes) * len(duplicate_seeds)
    group_index = 0
    for instance, p_dep in cases:
        graph = graphs[instance]
        ref = reference[(instance, float(p_dep))]
        for mode in modes:
            case_id = _diagnostic_case_id(instance, float(p_dep), mode, int(shots), int(base_seed), int(t_budget), int(starts))
            objective = _objective_for_mode(graph, mode, float(p_dep), int(shots), int(base_seed), int(t_budget))
            for duplicate_id, duplicate_seed in enumerate(duplicate_seeds):
                group_index += 1
                if progress:
                    print(
                        f"[jitter-attribution] phase B {group_index}/{total_groups} {case_id} duplicate_seed={duplicate_seed}",
                        flush=True,
                    )
                theta0s = make_paired_initial_thetas(
                    depth=2,
                    starts=int(starts),
                    seed=deterministic_seed("phase_e_c0_jitter_attribution_starts", case_id, duplicate_seed, starts),
                )
                started_group = time.perf_counter()
                start_rows: list[dict[str, Any]] = []
                for start_id, theta0 in enumerate(theta0s):
                    opt_seed = deterministic_seed("phase_e_c0_jitter_attribution_opt", case_id, duplicate_seed, start_id)
                    result = spsa_minimize(objective, theta0, config, seed=opt_seed)
                    ideal_value = float(evaluate_custom_density_objective(graph, result.best_theta, p_dep=0.0, shots=None).objective_value)
                    ideal_regret = float(ideal_value - float(ref["range_1024_min"]))
                    final_vs_best_gap = float(result.final_value - result.best_value)
                    row = {
                        "diagnostic_case_id": case_id,
                        "instance": instance,
                        "p_dep": float(p_dep),
                        "mode": mode,
                        "shots_total": int(shots),
                        "base_seed": int(base_seed),
                        "T_requested": int(t_budget),
                        "starts_per_duplicate": int(starts),
                        "duplicate_id": int(duplicate_id),
                        "duplicate_seed": int(duplicate_seed),
                        "start_id": int(start_id),
                        "optimizer_seed": int(opt_seed),
                        "initial_theta_json": json.dumps(tuple(float(x) for x in theta0)),
                        "theta_best_json": json.dumps(result.best_theta),
                        "theta_final_json": json.dumps(result.final_theta),
                        "best_observed_value": float(result.best_value),
                        "final_observed_value": float(result.final_value),
                        "final_vs_best_gap": final_vs_best_gap,
                        "final_vs_best_gap_fraction_exact_cut": final_vs_best_gap / float(ref["exact_cut_range"]) if float(ref["exact_cut_range"]) > 0.0 else math.nan,
                        "endpoint_regression_detected": bool(final_vs_best_gap / float(ref["exact_cut_range"]) > 0.05) if float(ref["exact_cut_range"]) > 0.0 else False,
                        "ideal_value_at_best_theta": ideal_value,
                        "ideal_regret_vs_reference_min": ideal_regret,
                        "actual_num_evaluations": int(result.num_evaluations),
                        "range_1024_reference": float(ref["range_1024_reference"]),
                        "range_1024_min": float(ref["range_1024_min"]),
                        "range_1024_max": float(ref["range_1024_max"]),
                        "exact_cut_range": float(ref["exact_cut_range"]),
                        "exact_cut_status": ref["exact_cut_status"],
                        **gain_schedule,
                    }
                    start_rows.append(row)
                    all_rows.append(row)
                best = min(start_rows, key=lambda item: float(item["best_observed_value"]))
                best_rows.append({**best, "best_start_id": int(best["start_id"]), "operation": "best_of_k_duplicate_endpoint"})
                runtime_rows.append(
                    {
                        "diagnostic_case_id": case_id,
                        "instance": instance,
                        "p_dep": float(p_dep),
                        "mode": mode,
                        "duplicate_id": int(duplicate_id),
                        "duplicate_seed": int(duplicate_seed),
                        "elapsed_seconds": time.perf_counter() - started_group,
                        "starts_per_duplicate": int(starts),
                    }
                )

    pairwise_rows: list[dict[str, Any]] = []
    grouped_best: dict[str, list[dict[str, Any]]] = {}
    for row in best_rows:
        grouped_best.setdefault(str(row["diagnostic_case_id"]), []).append(row)
    for case_id, rows in grouped_best.items():
        rows = sorted(rows, key=lambda item: int(item["duplicate_id"]))
        for left, right in itertools.combinations(rows, 2):
            abs_diff = abs(float(left["ideal_regret_vs_reference_min"]) - float(right["ideal_regret_vs_reference_min"]))
            frac_1024 = abs_diff / float(left["range_1024_reference"]) if float(left["range_1024_reference"]) > 0.0 else math.nan
            frac_exact = abs_diff / float(left["exact_cut_range"]) if float(left["exact_cut_range"]) > 0.0 else math.nan
            pairwise_rows.append(
                {
                    "diagnostic_case_id": case_id,
                    "instance": left["instance"],
                    "p_dep": left["p_dep"],
                    "mode": left["mode"],
                    "shots_total": left["shots_total"],
                    "T_requested": left["T_requested"],
                    "starts_per_duplicate": left["starts_per_duplicate"],
                    "left_duplicate_id": left["duplicate_id"],
                    "right_duplicate_id": right["duplicate_id"],
                    "left_duplicate_seed": left["duplicate_seed"],
                    "right_duplicate_seed": right["duplicate_seed"],
                    "left_best_start_id": left["best_start_id"],
                    "right_best_start_id": right["best_start_id"],
                    "left_ideal_regret": left["ideal_regret_vs_reference_min"],
                    "right_ideal_regret": right["ideal_regret_vs_reference_min"],
                    "absolute_ideal_regret_difference": abs_diff,
                    "range_1024_reference": left["range_1024_reference"],
                    "exact_cut_range": left["exact_cut_range"],
                    "jitter_fraction_1024": frac_1024,
                    "jitter_fraction_exact_cut": frac_exact,
                    "jitter_classification_1024": _classify_jitter(frac_1024),
                    "jitter_classification_exact_cut": _classify_jitter(frac_exact),
                }
            )

    jitter_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", pairwise_rows)]
    for mode in sorted({str(row["mode"]) for row in pairwise_rows}):
        scopes.append(("mode", mode, [row for row in pairwise_rows if str(row["mode"]) == mode]))
    for case_id in sorted({str(row["diagnostic_case_id"]) for row in pairwise_rows}):
        scopes.append(("diagnostic_case", case_id, [row for row in pairwise_rows if str(row["diagnostic_case_id"]) == case_id]))
    for scope, key, rows in scopes:
        values = [float(row["jitter_fraction_1024"]) for row in rows]
        stats = _summary_stats(values)
        ci = _bootstrap_p95_ci(values, deterministic_seed("phase_e_c0_jitter_attribution_bootstrap", scope, key, len(rows)))
        jitter_rows.append({"scope": scope, "key": key, **stats, **ci})

    basin_rows: list[dict[str, Any]] = []
    for key, rows in itertools.groupby(
        sorted(all_rows, key=lambda item: (str(item["diagnostic_case_id"]), int(item["duplicate_id"]))),
        key=lambda item: (str(item["diagnostic_case_id"]), int(item["duplicate_id"])),
    ):
        case_id, duplicate_id = key
        group = list(rows)
        ideal_values = [float(row["ideal_value_at_best_theta"]) for row in group]
        exact_range = float(group[0]["exact_cut_range"])
        thresholds = {name: mult * exact_range for name, mult in (("t005", 0.005), ("t010", 0.010), ("t020", 0.020))}
        n_clusters: dict[str, int] = {}
        cluster_sizes: dict[str, list[int]] = {}
        dominant: dict[str, float] = {}
        best_cluster: dict[str, int] = {}
        best_start = min(range(len(group)), key=lambda idx: float(group[idx]["best_observed_value"]))
        for label, threshold in thresholds.items():
            clusters = _simple_clusters(ideal_values, threshold)
            n_clusters[label] = len(clusters)
            cluster_sizes[label] = [len(cluster) for cluster in clusters]
            dominant[label] = max(cluster_sizes[label]) / len(group) if group else math.nan
            best_cluster[label] = next((idx for idx, cluster in enumerate(clusters) if best_start in cluster), -1)
        primary_n = n_clusters["t010"]
        if primary_n == 1:
            basin_structure = "single_basin"
        elif dominant["t010"] >= 0.80:
            basin_structure = "dominant_basin_with_outliers"
        else:
            basin_structure = "multi_basin"
        basin_rows.append(
            {
                "diagnostic_case_id": case_id,
                "duplicate_id": duplicate_id,
                "duplicate_seed": group[0]["duplicate_seed"],
                "instance": group[0]["instance"],
                "p_dep": group[0]["p_dep"],
                "mode": group[0]["mode"],
                "exact_cut_range": exact_range,
                "n_clusters_by_threshold_json": json.dumps(n_clusters, sort_keys=True),
                "cluster_sizes_by_threshold_json": json.dumps(cluster_sizes, sort_keys=True),
                "dominant_cluster_fraction_by_threshold_json": json.dumps(dominant, sort_keys=True),
                "best_endpoint_cluster_id_by_threshold_json": json.dumps(best_cluster, sort_keys=True),
                "basin_count_threshold_sensitive": bool(max(n_clusters.values()) - min(n_clusters.values()) >= 2),
                "basin_structure_primary": basin_structure,
                "within_duplicate_start_spread_ideal": float(max(ideal_values) - min(ideal_values)),
                "best_of_k_ideal_value": float(group[best_start]["ideal_value_at_best_theta"]),
            }
        )

    variance_rows: list[dict[str, Any]] = []
    mode_order = ("ideal_deterministic", "noisy_deterministic", "noisy_sampled", "zne_deterministic", "zne_sampled")
    for instance, p_dep in sorted({(str(row["instance"]), float(row["p_dep"])) for row in best_rows}):
        values_by_mode: dict[str, list[float]] = {}
        for mode in mode_order:
            values_by_mode[mode] = [
                float(row["ideal_value_at_best_theta"])
                for row in best_rows
                if str(row["instance"]) == instance and float(row["p_dep"]) == p_dep and str(row["mode"]) == mode
            ]
        if not all(values_by_mode[mode] for mode in mode_order):
            continue
        v_ideal = _variance(values_by_mode["ideal_deterministic"])
        v_noisy_det = _variance(values_by_mode["noisy_deterministic"])
        v_noisy_sampled = _variance(values_by_mode["noisy_sampled"])
        v_zne_det = _variance(values_by_mode["zne_deterministic"])
        v_zne_sampled = _variance(values_by_mode["zne_sampled"])
        components = {
            "V_landscape": v_ideal,
            "V_noise_channel": v_noisy_det - v_ideal,
            "V_shot": v_noisy_sampled - v_noisy_det,
            "V_zne_landscape": v_zne_det - v_noisy_det,
            "V_zne_shot": v_zne_sampled - v_zne_det,
        }
        total_noisy = v_noisy_sampled
        total_zne = v_zne_sampled

        def safe_fraction(value: float, denom: float) -> float:
            return max(0.0, float(value)) / float(denom) if denom > 0.0 else math.nan

        negative_ratios = {
            "V_noise_channel": abs(min(0.0, components["V_noise_channel"])) / total_noisy if total_noisy > 0.0 else 0.0,
            "V_shot": abs(min(0.0, components["V_shot"])) / total_noisy if total_noisy > 0.0 else 0.0,
            "V_zne_landscape": abs(min(0.0, components["V_zne_landscape"])) / total_zne if total_zne > 0.0 else 0.0,
            "V_zne_shot": abs(min(0.0, components["V_zne_shot"])) / total_zne if total_zne > 0.0 else 0.0,
        }
        additive_valid = all(value <= 0.10 for value in negative_ratios.values())
        weights = richardson_weights(DEFAULT_C0_JITTER_SCALE_FACTORS)
        theoretical = float(sum(weight * weight for weight in weights))
        empirical = total_zne / total_noisy if total_noisy > 0.0 else math.nan
        if math.isfinite(empirical) and empirical > 2.0 * theoretical:
            richardson_flag = "excess_variance_beyond_Richardson_coefficient_prediction"
        elif math.isfinite(empirical) and empirical < 0.5 * theoretical:
            richardson_flag = "anomalously_low_empirical_zne_variance"
        else:
            richardson_flag = "variance_amplification_consistent_with_Richardson_coefficients"
        fractions = {
            "fraction_noisy_landscape": safe_fraction(components["V_landscape"], total_noisy),
            "fraction_noisy_noise_channel": safe_fraction(components["V_noise_channel"], total_noisy),
            "fraction_noisy_shot": safe_fraction(components["V_shot"], total_noisy),
            "fraction_zne_landscape_reference": safe_fraction(v_ideal, total_zne),
            "fraction_zne_landscape_distortion": safe_fraction(components["V_zne_landscape"], total_zne),
            "fraction_zne_shot": safe_fraction(components["V_zne_shot"], total_zne),
        }
        dominant = max(fractions.items(), key=lambda item: -1.0 if math.isnan(item[1]) else item[1])[0] if additive_valid else "not_assigned_non_additive"
        variance_rows.append(
            {
                "instance": instance,
                "p_dep": p_dep,
                "V_ideal": v_ideal,
                "V_noisy_det": v_noisy_det,
                "V_noisy_sampled": v_noisy_sampled,
                "V_zne_det": v_zne_det,
                "V_zne_sampled": v_zne_sampled,
                "V_total_noisy": total_noisy,
                "V_total_zne": total_zne,
                "V_zne_total_amplification": empirical,
                **components,
                **fractions,
                "negative_component_ratios_json": json.dumps(negative_ratios, sort_keys=True),
                "additive_decomposition_valid": bool(additive_valid),
                "dominant_variance_source": dominant,
                "richardson_weights_json": json.dumps(weights),
                "richardson_coefficient_variance_factor": theoretical,
                "empirical_zne_variance_ratio": empirical,
                "richardson_variance_flag": richardson_flag,
            }
        )

    _write_csv(paths["all_start_endpoints"], all_rows)
    _write_csv(paths["best_of_k_pairwise"], pairwise_rows)
    _write_csv(paths["jitter_summary_table"], jitter_rows)
    _write_csv(paths["basin_proxy"], basin_rows)
    _write_csv(paths["basin_convergence"], basin_rows)
    _write_csv(paths["variance_decomposition"], variance_rows)

    overall = next(row for row in jitter_rows if row["scope"] == "overall" and row["key"] == "all")
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_c0_jitter_attribution_phase_b_complete",
        "scope": "diagnostic_causal_attribution_not_protocol_amendment",
        "attribution_version": PHASE_E_C0_JITTER_ATTRIBUTION_VERSION,
        "diagnostic_only": True,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "thresholds_changed": False,
        "paper_scope_changed": False,
        "final_hyperparameter_selection": False,
        "cases": [{"instance": instance, "p_dep": p_dep} for instance, p_dep in cases],
        "modes": list(modes),
        "duplicate_seeds": list(duplicate_seeds),
        "T_requested": int(t_budget),
        "starts_per_duplicate": int(starts),
        "shots_total": int(shots),
        "base_seed": int(base_seed),
        "overall_jitter_summary": overall,
        "all_start_rows": len(all_rows),
        "pairwise_rows": len(pairwise_rows),
        "best_endpoint_rows": len(best_rows),
        "basin_rows": len(basin_rows),
        "variance_rows": len(variance_rows),
        "spsa_gain_schedule_summary": gain_schedule,
        "elapsed_seconds": time.perf_counter() - started,
        "time_box_seconds": 8 * 3600,
        "time_box_status": "within_time_box" if time.perf_counter() - started <= 8 * 3600 else "exceeded_time_box",
        "all_checked_numeric_finite": all(
            math.isfinite(float(row["ideal_value_at_best_theta"])) and math.isfinite(float(row["best_observed_value"]))
            for row in all_rows
        ),
        "input_hashes": {
            str((root / "config" / "phase_e_instances_v0.1-draft.json").relative_to(root)).replace("\\", "/"): instances_hash,
            str(PLAN_PATH).replace("\\", "/"): sha256_file(root / PLAN_PATH) if (root / PLAN_PATH).exists() else None,
        },
        "code_hashes": {
            "src/phase_e_c0_jitter_attribution.py": sha256_file(root / "src" / "phase_e_c0_jitter_attribution.py"),
            "scripts/65_phase_e_c0_jitter_attribution.py": sha256_file(root / "scripts" / "65_phase_e_c0_jitter_attribution.py"),
        },
        "git": _git_status(root),
        "key_library_versions": _library_versions(),
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "limitations": [
            "This is a diagnostic attribution run and does not modify the completed C0 jitter gate.",
            "Dominant cause assignment requires synthesis review, especially if variance components are non-additive or ambiguity remains.",
        ],
    }
    write_json_atomic(paths["jitter_summary"], summary, overwrite=True)
    return summary


def _diagnostic_case_sort_key(case_id: str) -> tuple[str, float, str]:
    parts = dict(part.split("=", 1) for part in case_id.split("|") if "=" in part)
    instance = case_id.split("|", 1)[0]
    return (instance, float(parts.get("p", "nan")), parts.get("mode", ""))


def _diagnostic_case_metadata(case_id: str) -> dict[str, Any]:
    parts = dict(part.split("=", 1) for part in case_id.split("|") if "=" in part)
    return {
        "instance": case_id.split("|", 1)[0],
        "p_dep": float(parts["p"]),
        "mode": parts["mode"],
        "shots_total": int(parts.get("shots", 0)),
        "base_seed": int(parts.get("base_seed", 0)),
        "T_requested": int(parts.get("T", 0)),
        "starts": int(parts.get("K", 0)),
    }


def _pairwise_jitter_for_selected(rows: list[dict[str, Any]], denominator_field: str) -> list[float]:
    values: list[float] = []
    sorted_rows = sorted(rows, key=lambda item: int(item["duplicate_id"]))
    for left, right in itertools.combinations(sorted_rows, 2):
        denom = float(left[denominator_field])
        diff = abs(float(left["ideal_regret_vs_reference_min"]) - float(right["ideal_regret_vs_reference_min"]))
        values.append(diff / denom if denom > 0.0 else math.nan)
    return values


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def run_phase_e_c0_jitter_attribution_endpoint_k_postprocess(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_jitter_attribution")
    paths = {
        "selection_summary": output_dir / "endpoint_k_selection_summary.csv",
        "duplicate_selection": output_dir / "endpoint_k_duplicate_selection.csv",
        "jitter_by_selection": output_dir / "endpoint_k_jitter_by_selection.csv",
        "k_sensitivity": output_dir / "endpoint_k_observed_k_sensitivity.csv",
        "summary": output_dir / "endpoint_k_followup_summary.json",
        "report": output_dir / "endpoint_k_followup_report.md",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing endpoint/K postprocess outputs.")

    input_paths = [
        output_dir / "diagnostic_all_start_endpoints.csv",
        output_dir / "h4_p001_required" / "diagnostic_all_start_endpoints.csv",
    ]
    existing_inputs = [path for path in input_paths if path.exists()]
    if not existing_inputs:
        raise FileNotFoundError("No Phase B all-start endpoint CSVs found.")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for path in existing_inputs:
        if progress:
            print(f"[jitter-attribution] endpoint/K postprocess load {path.relative_to(root)}", flush=True)
        loaded = _read_csv(path)
        for row in loaded:
            row["source_file"] = str(path.relative_to(root)).replace("\\", "/")
        rows.extend(loaded)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["diagnostic_case_id"]), int(row["duplicate_id"])), []).append(row)

    duplicate_rows: list[dict[str, Any]] = []
    observed_selected: dict[str, list[dict[str, Any]]] = {}
    oracle_selected: dict[str, list[dict[str, Any]]] = {}
    for (case_id, duplicate_id), group in sorted(grouped.items(), key=lambda item: (_diagnostic_case_sort_key(item[0][0]), item[0][1])):
        group = sorted(group, key=lambda item: int(item["start_id"]))
        observed = min(group, key=lambda item: float(item["best_observed_value"]))
        oracle = min(group, key=lambda item: float(item["ideal_value_at_best_theta"]))
        exact_range = float(group[0]["exact_cut_range"])
        selection_gap = float(observed["ideal_value_at_best_theta"]) - float(oracle["ideal_value_at_best_theta"])
        selection_gap_fraction = selection_gap / exact_range if exact_range > 0.0 else math.nan
        ideal_sorted = sorted(group, key=lambda item: float(item["ideal_value_at_best_theta"]))
        observed_rank = 1 + next(
            index for index, item in enumerate(ideal_sorted) if int(item["start_id"]) == int(observed["start_id"])
        )
        thresholds = {
            "t005": 0.005 * exact_range,
            "t010": 0.010 * exact_range,
            "t020": 0.020 * exact_range,
        }
        good_counts = {
            label: sum(
                1
                for item in group
                if float(item["ideal_value_at_best_theta"]) <= float(oracle["ideal_value_at_best_theta"]) + threshold
            )
            for label, threshold in thresholds.items()
        }
        regression_count = sum(1 for item in group if str(item.get("endpoint_regression_detected", "False")) == "True")
        out = {
            "diagnostic_case_id": case_id,
            "instance": group[0]["instance"],
            "p_dep": float(group[0]["p_dep"]),
            "mode": group[0]["mode"],
            "duplicate_id": duplicate_id,
            "duplicate_seed": int(group[0]["duplicate_seed"]),
            "starts": len(group),
            "observed_best_start_id": int(observed["start_id"]),
            "oracle_ideal_best_start_id": int(oracle["start_id"]),
            "observed_equals_oracle": int(observed["start_id"]) == int(oracle["start_id"]),
            "observed_ideal_rank": observed_rank,
            "observed_ideal_value": float(observed["ideal_value_at_best_theta"]),
            "oracle_ideal_value": float(oracle["ideal_value_at_best_theta"]),
            "selection_gap_ideal_value": selection_gap,
            "selection_gap_fraction_exact_cut": selection_gap_fraction,
            "endpoint_definition_miss_t005": selection_gap > thresholds["t005"],
            "endpoint_definition_miss_t010": selection_gap > thresholds["t010"],
            "endpoint_definition_miss_t020": selection_gap > thresholds["t020"],
            "good_start_count_t005": good_counts["t005"],
            "good_start_count_t010": good_counts["t010"],
            "good_start_count_t020": good_counts["t020"],
            "good_start_fraction_t010": good_counts["t010"] / len(group) if group else math.nan,
            "endpoint_regression_count": regression_count,
            "endpoint_regression_fraction": regression_count / len(group) if group else math.nan,
            "range_1024_reference": float(group[0]["range_1024_reference"]),
            "exact_cut_range": exact_range,
        }
        duplicate_rows.append(out)
        observed_selected.setdefault(case_id, []).append(observed)
        oracle_selected.setdefault(case_id, []).append(oracle)

    selection_summary_rows: list[dict[str, Any]] = []
    for case_id in sorted({row["diagnostic_case_id"] for row in duplicate_rows}, key=_diagnostic_case_sort_key):
        case_rows = [row for row in duplicate_rows if row["diagnostic_case_id"] == case_id]
        gap_values = [float(row["selection_gap_fraction_exact_cut"]) for row in case_rows]
        ranks = [float(row["observed_ideal_rank"]) for row in case_rows]
        selection_summary_rows.append(
            {
                "diagnostic_case_id": case_id,
                "instance": case_rows[0]["instance"],
                "p_dep": case_rows[0]["p_dep"],
                "mode": case_rows[0]["mode"],
                "duplicates": len(case_rows),
                "observed_equals_oracle_count": sum(1 for row in case_rows if bool(row["observed_equals_oracle"])),
                "endpoint_definition_miss_rate_t005": sum(1 for row in case_rows if bool(row["endpoint_definition_miss_t005"])) / len(case_rows),
                "endpoint_definition_miss_rate_t010": sum(1 for row in case_rows if bool(row["endpoint_definition_miss_t010"])) / len(case_rows),
                "endpoint_definition_miss_rate_t020": sum(1 for row in case_rows if bool(row["endpoint_definition_miss_t020"])) / len(case_rows),
                "selection_gap_fraction_p50": float(np.percentile(np.asarray(gap_values, dtype=float), 50)),
                "selection_gap_fraction_max": float(np.max(np.asarray(gap_values, dtype=float))),
                "observed_ideal_rank_p50": float(np.percentile(np.asarray(ranks, dtype=float), 50)),
                "observed_ideal_rank_max": float(np.max(np.asarray(ranks, dtype=float))),
                "mean_good_start_fraction_t010": float(np.mean([float(row["good_start_fraction_t010"]) for row in case_rows])),
                "endpoint_regression_fraction_mean": float(np.mean([float(row["endpoint_regression_fraction"]) for row in case_rows])),
            }
        )

    jitter_rows: list[dict[str, Any]] = []
    for case_id in sorted(observed_selected, key=_diagnostic_case_sort_key):
        for selection_type, selected in (
            ("observed_best", observed_selected[case_id]),
            ("oracle_ideal_best", oracle_selected[case_id]),
        ):
            for denominator_field, label in (("range_1024_reference", "range_1024"), ("exact_cut_range", "exact_cut")):
                values = _pairwise_jitter_for_selected(selected, denominator_field)
                stats = _summary_stats(values)
                jitter_rows.append(
                    {
                        "diagnostic_case_id": case_id,
                        "instance": selected[0]["instance"],
                        "p_dep": float(selected[0]["p_dep"]),
                        "mode": selected[0]["mode"],
                        "selection_type": selection_type,
                        "denominator": label,
                        **stats,
                    }
                )

    k_sensitivity_rows: list[dict[str, Any]] = []
    k_values = (1, 2, 5, 10, 20)
    for case_id in sorted({row["diagnostic_case_id"] for row in rows}, key=_diagnostic_case_sort_key):
        case_duplicate_groups = {
            duplicate_id: sorted(group, key=lambda item: int(item["start_id"]))
            for (group_case_id, duplicate_id), group in grouped.items()
            if group_case_id == case_id
        }
        for k_value in k_values:
            if any(len(group) < k_value for group in case_duplicate_groups.values()):
                continue
            selected = []
            for duplicate_id in sorted(case_duplicate_groups):
                subset = [row for row in case_duplicate_groups[duplicate_id] if int(row["start_id"]) < k_value]
                selected.append(min(subset, key=lambda item: float(item["best_observed_value"])))
            values = _pairwise_jitter_for_selected(selected, "range_1024_reference")
            stats = _summary_stats(values)
            k_sensitivity_rows.append(
                {
                    "diagnostic_case_id": case_id,
                    "instance": selected[0]["instance"],
                    "p_dep": float(selected[0]["p_dep"]),
                    "mode": selected[0]["mode"],
                    "observed_selection_k": k_value,
                    **stats,
                }
            )

    _write_csv(paths["duplicate_selection"], duplicate_rows)
    _write_csv(paths["selection_summary"], selection_summary_rows)
    _write_csv(paths["jitter_by_selection"], jitter_rows)
    _write_csv(paths["k_sensitivity"], k_sensitivity_rows)

    oracle_unstable = [
        row
        for row in jitter_rows
        if row["selection_type"] == "oracle_ideal_best"
        and row["denominator"] == "range_1024"
        and row["classification"] == "unstable"
    ]
    observed_unstable = [
        row
        for row in jitter_rows
        if row["selection_type"] == "observed_best"
        and row["denominator"] == "range_1024"
        and row["classification"] == "unstable"
    ]
    endpoint_definition_cases = [
        row for row in selection_summary_rows if float(row["endpoint_definition_miss_rate_t010"]) >= 0.40
    ]
    k20_rows = [row for row in k_sensitivity_rows if int(row["observed_selection_k"]) == 20]
    k20_unstable = [row for row in k20_rows if row["classification"] == "unstable"]

    report_rows = [
        {
            "p_dep": row["p_dep"],
            "mode": row["mode"],
            "miss_rate_t010": row["endpoint_definition_miss_rate_t010"],
            "rank_p50": row["observed_ideal_rank_p50"],
            "gap_max": row["selection_gap_fraction_max"],
            "good_fraction_t010": row["mean_good_start_fraction_t010"],
            "regression_fraction": row["endpoint_regression_fraction_mean"],
        }
        for row in selection_summary_rows
    ]
    oracle_rows = [
        {
            "p_dep": row["p_dep"],
            "mode": row["mode"],
            "selection": row["selection_type"],
            "p95": row["p95"],
            "class": row["classification"],
        }
        for row in jitter_rows
        if row["denominator"] == "range_1024"
    ]
    k_report_rows = [
        {
            "p_dep": row["p_dep"],
            "mode": row["mode"],
            "K": row["observed_selection_k"],
            "p95": row["p95"],
            "class": row["classification"],
        }
        for row in k_sensitivity_rows
        if int(row["observed_selection_k"]) in (5, 10, 20)
    ]

    endpoint_selection_rule_supported = len(endpoint_definition_cases) > 0
    oracle_selection_sufficient = len(oracle_unstable) == 0
    k50_supported_by_postprocess = False

    report = "\n".join(
        [
            "# Phase E C0 Endpoint/K Follow-up",
            "",
            f"created_at_utc: {_utc_now()}",
            "diagnostic_only: true",
            "stage_c_status: blocked",
            "official_execution_authorized: false",
            "scope_changed: false",
            "thresholds_changed: false",
            "",
            "## Summary",
            "",
            "This follow-up reuses the saved Phase B all-start endpoints. No optimizer runs were executed.",
            "",
            f"observed_unstable_cases: {len(observed_unstable)} / {len(k20_rows)}",
            f"oracle_ideal_unstable_cases: {len(oracle_unstable)} / {len(k20_rows)}",
            f"endpoint_definition_cases_miss_rate_ge_0_40: {len(endpoint_definition_cases)}",
            f"endpoint_selection_rule_issue_supported: {str(endpoint_selection_rule_supported).lower()}",
            f"oracle_selection_sufficient_to_stabilize: {str(oracle_selection_sufficient).lower()}",
            f"K50_supported_by_postprocess_alone: {str(k50_supported_by_postprocess).lower()}",
            "",
            "Interpretation: oracle ideal selection among the existing K=20 starts remains unstable in most cases, so changing only the observed-best endpoint-selection rule is not sufficient. The problem is better described as endpoint protocol instability under multi-basin best-of-K, not as a simple observed-vs-ideal selection mistake.",
            "",
            "## Endpoint Selection Summary",
            "",
            _markdown_table(report_rows, ["p_dep", "mode", "miss_rate_t010", "rank_p50", "gap_max", "good_fraction_t010", "regression_fraction"]),
            "",
            "## Observed vs Oracle Jitter",
            "",
            _markdown_table(oracle_rows, ["p_dep", "mode", "selection", "p95", "class"]),
            "",
            "## Observed K Sensitivity",
            "",
            _markdown_table(k_report_rows, ["p_dep", "mode", "K", "p95", "class"]),
            "",
            "## Decision",
            "",
            "recommended_next_step: write_C0_K_or_endpoint_definition_amendment",
            "",
            "Do not run K=50 yet. The existing K=20 all-start endpoints do not show a clean K-escalation signal: K sensitivity is mixed/non-monotone, and oracle selection within K=20 is still unstable. Write a C0 amendment that treats the endpoint protocol itself as unstable, or run only a narrow confirmation test with explicit approval.",
            "",
            "Stage C remains blocked. No final hyperparameters were selected.",
            "",
        ]
    )
    paths["report"].write_text(report, encoding="utf-8")

    summary = {
        "created_at_utc": _utc_now(),
        "status": "endpoint_k_postprocess_complete",
        "scope": "postprocess_existing_all_start_endpoints_not_protocol_amendment",
        "diagnostic_only": True,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "thresholds_changed": False,
        "paper_scope_changed": False,
        "final_hyperparameter_selection": False,
        "input_files": [str(path.relative_to(root)).replace("\\", "/") for path in existing_inputs],
        "all_start_rows": len(rows),
        "duplicate_groups": len(grouped),
        "selection_summary_rows": len(selection_summary_rows),
        "observed_unstable_cases": len(observed_unstable),
        "oracle_ideal_unstable_cases": len(oracle_unstable),
        "endpoint_definition_cases_miss_rate_ge_0_40": len(endpoint_definition_cases),
        "endpoint_selection_rule_issue_supported": endpoint_selection_rule_supported,
        "oracle_selection_sufficient_to_stabilize": oracle_selection_sufficient,
        "K50_supported_by_postprocess_alone": k50_supported_by_postprocess,
        "k20_unstable_cases": len(k20_unstable),
        "recommended_next_step": "write_C0_K_or_endpoint_definition_amendment",
        "elapsed_seconds": time.perf_counter() - started,
        "input_hashes": {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in existing_inputs},
        "code_hashes": {
            "src/phase_e_c0_jitter_attribution.py": sha256_file(root / "src" / "phase_e_c0_jitter_attribution.py"),
            "scripts/65_phase_e_c0_jitter_attribution.py": sha256_file(root / "scripts" / "65_phase_e_c0_jitter_attribution.py"),
        },
        "git": _git_status(root),
        "key_library_versions": _library_versions(),
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    return summary


def _load_diagnostic_case_jitter_rows(paths: list[Path]) -> dict[tuple[float, str, int], dict[str, Any]]:
    out: dict[tuple[float, str, int], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            if str(row.get("scope")) != "diagnostic_case":
                continue
            metadata = _diagnostic_case_metadata(str(row["key"]))
            out[(float(metadata["p_dep"]), str(metadata["mode"]), int(metadata["T_requested"]))] = {
                **row,
                **metadata,
                "p95": float(row["p95"]),
                "p50": float(row["p50"]),
                "p90": float(row["p90"]),
                "max": float(row["max"]),
                "count": int(row["count"]),
                "classification": str(row["classification"]),
                "source_path": str(path),
            }
    return out


def run_phase_e_c0_jitter_attribution_phase_c_minimal(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    progress: bool = True,
    cases: tuple[tuple[str, float], ...] = (("H4", 0.003), ("H4", 0.01), ("H4", 0.02)),
    mode: str = "ideal_deterministic",
    t_budget: int = 500,
    starts: int = PHASE_B_DEFAULT_STARTS,
    duplicate_seeds: tuple[int, ...] = PHASE_B_DEFAULT_DUPLICATE_SEEDS,
    shots: int = PHASE_B_DEFAULT_SHOTS,
    base_seed: int = PHASE_B_DEFAULT_BASE_SEED,
    output_tag: str = "t500_ideal_h4",
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_jitter_attribution")
    paths = {
        "comparison": output_dir / "t_scaling_minimal_comparison.csv",
        "summary": output_dir / "t_scaling_minimal_summary.json",
        "report": output_dir / "t_scaling_minimal_report.md",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase C minimal outputs.")

    started = time.perf_counter()
    if progress:
        print("[jitter-attribution] phase C minimal run T=500 ideal_deterministic H4 controls", flush=True)
    run_summary = run_phase_e_c0_jitter_attribution_phase_b(
        root=root,
        force=force,
        progress=progress,
        cases=cases,
        modes=(mode,),
        duplicate_seeds=duplicate_seeds,
        t_budget=t_budget,
        starts=starts,
        shots=shots,
        base_seed=base_seed,
        output_tag=output_tag,
    )

    baseline_rows = _load_diagnostic_case_jitter_rows(
        [
            output_dir / "diagnostic_jitter_by_mode_summary.csv",
            output_dir / "h4_p001_required" / "diagnostic_jitter_by_mode_summary.csv",
        ]
    )
    t_rows = _load_diagnostic_case_jitter_rows(
        [output_dir / output_tag / "diagnostic_jitter_by_mode_summary.csv"]
    )

    comparison_rows: list[dict[str, Any]] = []
    for instance, p_dep in cases:
        baseline = baseline_rows[(float(p_dep), mode, 250)]
        current = t_rows[(float(p_dep), mode, int(t_budget))]
        baseline_p95 = float(baseline["p95"])
        current_p95 = float(current["p95"])
        ratio = current_p95 / baseline_p95 if baseline_p95 > 0.0 else math.nan
        improvement_fraction = (baseline_p95 - current_p95) / baseline_p95 if baseline_p95 > 0.0 else math.nan
        non_monotonic = math.isfinite(ratio) and ratio > 1.2
        if str(current["classification"]) == "stable":
            decision = "T500_stabilized_case"
        elif non_monotonic:
            decision = "T500_non_monotonic_worse_stop_case"
        elif improvement_fraction >= 0.50:
            decision = "T500_strong_improvement_not_yet_stable"
        elif improvement_fraction > 0.0:
            decision = "T500_modest_improvement"
        else:
            decision = "T500_no_improvement"
        comparison_rows.append(
            {
                "instance": instance,
                "p_dep": float(p_dep),
                "mode": mode,
                "baseline_T": 250,
                "current_T": int(t_budget),
                "baseline_p95": baseline_p95,
                "current_p95": current_p95,
                "p95_ratio_current_over_baseline": ratio,
                "improvement_fraction": improvement_fraction,
                "baseline_classification": baseline["classification"],
                "current_classification": current["classification"],
                "non_monotonic_guard_triggered": bool(non_monotonic),
                "case_decision": decision,
            }
        )

    _write_csv(paths["comparison"], comparison_rows)

    stable_count = sum(1 for row in comparison_rows if row["current_classification"] == "stable")
    marginal_count = sum(1 for row in comparison_rows if row["current_classification"] == "marginal")
    unstable_count = sum(1 for row in comparison_rows if row["current_classification"] == "unstable")
    non_monotonic_count = sum(1 for row in comparison_rows if bool(row["non_monotonic_guard_triggered"]))
    strong_improvement_count = sum(1 for row in comparison_rows if float(row["improvement_fraction"]) >= 0.50)
    if stable_count > 0:
        recommended = "write_C0_T_recalibration_amendment_or_continue_T1000_for_remaining_cases"
    elif non_monotonic_count > 0:
        recommended = "stop_T_scaling_for_non_monotonic_cases_and_evaluate_learning_rate_or_endpoint_protocol"
    elif strong_improvement_count == len(comparison_rows):
        recommended = "continue_phase_C_T1000_with_explicit_approval"
    elif strong_improvement_count > 0 or marginal_count > 0:
        recommended = "consider_narrow_T1000_confirmation_with_explicit_approval"
    else:
        recommended = "T500_does_not_support_T_insufficiency_pause_or_change_optimizer_protocol"

    report_rows = [
        {
            "p_dep": row["p_dep"],
            "T250_p95": row["baseline_p95"],
            "T500_p95": row["current_p95"],
            "ratio": row["p95_ratio_current_over_baseline"],
            "improvement": row["improvement_fraction"],
            "class": row["current_classification"],
            "decision": row["case_decision"],
        }
        for row in comparison_rows
    ]
    report = "\n".join(
        [
            "# Phase E C0 Minimal T Scaling Diagnostic",
            "",
            f"created_at_utc: {_utc_now()}",
            "diagnostic_only: true",
            "stage_c_status: blocked",
            "official_execution_authorized: false",
            "scope_changed: false",
            "thresholds_changed: false",
            "final_hyperparameter_selection: false",
            "",
            "## Design",
            "",
            "This minimal Phase C diagnostic tests whether the base ideal deterministic H4 endpoint jitter improves when SPSA budget increases from T=250 to T=500.",
            "",
            "```text",
            "cases = H4 p_dep in [0.003, 0.010, 0.020]",
            "mode = ideal_deterministic",
            "K = 20",
            "duplicate_seeds = [101, 202, 303, 404, 505]",
            "```",
            "",
            "## Comparison",
            "",
            _markdown_table(report_rows, ["p_dep", "T250_p95", "T500_p95", "ratio", "improvement", "class", "decision"]),
            "",
            "## Decision",
            "",
            f"recommended_next_step: {recommended}",
            "",
            "No final T is selected by this diagnostic. Stage C remains blocked.",
            "",
        ]
    )
    paths["report"].write_text(report, encoding="utf-8")

    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_c0_jitter_attribution_phase_c_minimal_complete",
        "scope": "minimal_T500_ideal_deterministic_H4_diagnostic_not_protocol_amendment",
        "diagnostic_only": True,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "thresholds_changed": False,
        "paper_scope_changed": False,
        "final_hyperparameter_selection": False,
        "cases": [{"instance": instance, "p_dep": p_dep} for instance, p_dep in cases],
        "mode": mode,
        "baseline_T": 250,
        "current_T": int(t_budget),
        "starts_per_duplicate": int(starts),
        "duplicate_seeds": list(duplicate_seeds),
        "stable_count": stable_count,
        "marginal_count": marginal_count,
        "unstable_count": unstable_count,
        "non_monotonic_guard_count": non_monotonic_count,
        "strong_improvement_count": strong_improvement_count,
        "recommended_next_step": recommended,
        "run_summary": run_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "code_hashes": {
            "src/phase_e_c0_jitter_attribution.py": sha256_file(root / "src" / "phase_e_c0_jitter_attribution.py"),
            "scripts/65_phase_e_c0_jitter_attribution.py": sha256_file(root / "scripts" / "65_phase_e_c0_jitter_attribution.py"),
        },
        "git": _git_status(root),
        "key_library_versions": _library_versions(),
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    return summary
