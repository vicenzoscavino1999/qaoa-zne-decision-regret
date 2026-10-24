from __future__ import annotations

import csv
import importlib.metadata
import time
from pathlib import Path

import numpy as np

from .circuits import graph_from_config
from .config import MetricsConfig, SUPPORTED_FOLDING_BACKEND
from .figures import save_smoke_heatmaps
from .io_utils import PROJECT_ROOT, configure_logger, ensure_dir, read_yaml, save_array, write_json
from .landscapes import generate_ideal_landscape, generate_noisy_landscape, generate_zne_landscape
from .manifest import build_manifest
from .metrics import comparison_metrics, method_metrics
from .mitiq_wrapper import all_protocol_weights, try_import_mitiq


def _package_versions() -> dict[str, str]:
    names = ["numpy", "scipy", "pandas", "matplotlib", "PyYAML", "qiskit", "qiskit-aer", "mitiq", "ply"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def write_package_versions(path: Path) -> dict[str, str]:
    versions = _package_versions()
    with path.open("w", encoding="utf-8") as f:
        for key, value in sorted(versions.items()):
            f.write(f"{key}=={value}\n")
    return versions


def _write_metrics_csv(path: Path, rows: list[dict[str, object]], overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assert_can_write_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing smoke outputs. Use --force only for an intentional rerun.\n{joined}")


def run_smoke_test(config_path: str | Path | None = None, overwrite: bool = False) -> dict[str, object]:
    root = PROJECT_ROOT
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    smoke = cfg["smoke_test"]
    graph = graph_from_config(smoke["instance"], cfg)
    size = int(smoke["grid_size"])
    p_noise = float(smoke["noise"])
    shots_total = int(smoke["shots_total"])
    seed = int(smoke["seed"])
    scale_factors = [int(x) for x in smoke["scale_factors"]]
    zne_order = str(smoke["zne_order"])
    folding_backend = str(cfg["zne"]["folding_backend"])
    if folding_backend != SUPPORTED_FOLDING_BACKEND:
        raise ValueError(f"Configured folding backend is unsupported: {folding_backend}")
    metrics_config = MetricsConfig.from_experiment_config(cfg)

    base = root / "data" / "raw" / graph.name / f"p_{p_noise:g}" / f"B_{shots_total}" / f"seed_{seed}"
    raw_outputs = [
        root / "data" / "raw" / graph.name / "C_ideal.npy",
        base / "C_noisy.npy",
        base / f"C_zne_{zne_order}.npy",
        base / "C_noisy_p0.npy",
        base / f"C_zne_{zne_order}_p0.npy",
        root / "results" / "metrics_per_seed.csv",
        root / "results" / "comparisons_per_seed.csv",
        root / "results" / "phase_a_smoke_summary.json",
    ]
    _assert_can_write_outputs(raw_outputs, overwrite=overwrite)

    logger = configure_logger(root / "logs" / "phase_a_smoke.log")
    logger.info("Starting Phase A smoke test")
    logger.info("instance=%s grid=%s p=%s shots=%s seed=%s", graph.name, size, p_noise, shots_total, seed)

    mitiq_ok, mitiq_info = try_import_mitiq()
    if not mitiq_ok:
        raise RuntimeError(f"Mitiq import failed: {mitiq_info}")
    logger.info("Mitiq import OK: %s", mitiq_info)

    def progress(done: int, total: int, tag: str) -> None:
        logger.info("%s progress %s/%s", tag, done, total)

    t0 = time.perf_counter()
    ideal = generate_ideal_landscape(graph, size, progress=progress)
    noisy_p0 = generate_noisy_landscape(graph, size, 0.0, shots_total, seed, "noisy_p0", progress=progress)
    zne_p0, zne_p0_meta = generate_zne_landscape(
        graph,
        size,
        0.0,
        shots_total,
        seed,
        scale_factors,
        zne_order,
        folding_backend=folding_backend,
        progress=progress,
    )
    noisy = generate_noisy_landscape(graph, size, p_noise, shots_total, seed, "noisy", progress=progress)
    zne, zne_meta = generate_zne_landscape(
        graph,
        size,
        p_noise,
        shots_total,
        seed,
        scale_factors,
        zne_order,
        folding_backend=folding_backend,
        progress=progress,
    )
    elapsed = time.perf_counter() - t0

    save_array(root / "data" / "raw" / graph.name / "C_ideal.npy", ideal, overwrite=overwrite)
    save_array(base / "C_noisy.npy", noisy, overwrite=overwrite)
    save_array(base / f"C_zne_{zne_order}.npy", zne, overwrite=overwrite)
    save_array(base / "C_noisy_p0.npy", noisy_p0, overwrite=overwrite)
    save_array(base / f"C_zne_{zne_order}_p0.npy", zne_p0, overwrite=overwrite)

    weights = all_protocol_weights()
    write_json(root / "richardson_weights_v1.6.json", weights)

    metric_rows = []
    for method_name, arr in [("noisy", noisy), (f"zne_{zne_order}", zne), ("noisy_p0", noisy_p0), (f"zne_{zne_order}_p0", zne_p0)]:
        noisy_reference = noisy_p0 if "p0" in method_name else noisy
        metrics = method_metrics(arr, ideal, metrics_config=metrics_config, noisy_reference=noisy_reference)
        metric_rows.append({
            "experiment_id": cfg["experiment_id"],
            "instance": graph.name,
            "noise": p_noise if "p0" not in method_name else 0.0,
            "shots_total": shots_total,
            "seed": seed,
            "method": method_name,
            **metrics,
        })
    _write_metrics_csv(root / "results" / "metrics_per_seed.csv", metric_rows, overwrite=overwrite)

    comparison = comparison_metrics(noisy, zne, ideal)
    comparison_p0 = comparison_metrics(noisy_p0, zne_p0, ideal)
    comparison_row = {
        "experiment_id": cfg["experiment_id"],
        "instance": graph.name,
        "noise": p_noise,
        "shots_total": shots_total,
        "seed": seed,
        "zne_order": zne_order,
        **comparison,
        "below_detection_floor": method_metrics(noisy, ideal, metrics_config=metrics_config, noisy_reference=noisy)["below_detection_floor"],
        "extrapolation_breakdown": method_metrics(zne, ideal, metrics_config=metrics_config, noisy_reference=noisy)["extrapolation_breakdown"],
    }
    _write_metrics_csv(root / "results" / "comparisons_per_seed.csv", [comparison_row], overwrite=overwrite)

    save_smoke_heatmaps(root / "results" / "figures" / "phase_a_smoke_heatmaps.png", ideal, noisy, zne)

    versions = write_package_versions(root / "package_versions.txt")
    metadata = {
        "elapsed_seconds": elapsed,
        "mitiq_version": mitiq_info,
        "package_versions": versions,
        "zne_metadata": zne_meta,
        "zne_p0_metadata": zne_p0_meta,
        "comparison": comparison,
        "comparison_p0": comparison_p0,
        "folding_backend": folding_backend,
        "p0_noisy_mse": method_metrics(noisy_p0, ideal, metrics_config=metrics_config)["mse"],
        "p0_zne_mse": method_metrics(zne_p0, ideal, metrics_config=metrics_config)["mse"],
    }
    write_json(root / "results" / "phase_a_smoke_summary.json", metadata)
    build_manifest(root, root / "manifest_v1.6.json")
    logger.info("Phase A smoke test completed in %.2f seconds", elapsed)
    return metadata
