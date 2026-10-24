# Reproducibility Package Changelog

## 2026-05-24 - Reproducibility scaffold

- Created the reproducibility scaffold for the v0.2 upgrade plan.
- Added a pending baseline anchor in `reproducibility/config/baseline_commit.yaml`.
- Added `REPRODUCIBILITY_PHILOSOPHY.md` as the human-readable entry point.
- Added workstream specification files under `REPRODUCIBILITY_SPEC/`.
- No Phase C/D official reruns, manifest regeneration, Zenodo rebuild, or DOI
  insertion were performed.

## 2026-05-24 - Docker and Makefile verification entry points

- Added `Dockerfile`, `.dockerignore`, and `Makefile`.
- Added `reproducibility/check_env.py`, `reproducibility/show_file.py`, and a
  progressive `reproducibility/verify.py`.
- Wrote the first local Windows environment fingerprint and `frozen_env.txt`.
- Built Docker image `zne-verify`.
- Verified `docker run --rm zne-verify make verify` exits successfully with
  expected `PASS_WITH_WARNINGS` while later reference and paper-mapping files
  were still being assembled.

## 2026-05-24 - Reference manifest, subset, and tolerances

- Added `subset_specification.yaml`, `schemas.yaml`, and `tolerances.yaml`.
- Added `reference_manifest.json` and `reference_manifest_version.json`.
- Added helper scripts `resolve_subset_cells.py` and `build_reference_manifest.py`.
- Updated `reproducibility/verify.py` so `--reference-only` validates paths,
  byte sizes, and SHA-256 hashes.
- Verified 17 local reference outputs with `PASS`.
- Verified the existing numerical subset locally with `overall_pass=true`.
- Docker remains `PASS_WITH_WARNINGS` until a small data bundle or mounted
  `data/`/`results/` path is available.

## 2026-05-24 - Verifier, paper mapping, and runtime classes

- Added `paper_mapping.yaml`, `runtime_classes.yaml`, and
  `verify_report_schema.json`.
- Added `REPRODUCTION_COSTS.md`.
- Updated `verify.py --paper-mapping` to check mapped outputs, generators, data
  dependencies, and Phase E/F absence.
- Updated Docker progressive behavior so paper-mapping data dependencies under
  `results/` are warnings in the MVP image rather than failures.
- Measured local fast verification targets.
- Verified local `--paper-mapping` returns `PASS`.

## 2026-05-25 - Tests and seed registry

- Added `reproducibility/config/seed_registry.yaml`.
- Added the `unittest` suite for reference manifests, scope guard,
  verifier smoke checks, schema integrity, subset resolution, paper mapping,
  tolerance justification, cross-platform notes, verify-report schema shape,
  seed registry consistency, and adversarial corruption detection.
- Corrected the Phase B seed-selection schema to match the current
  `results/phase_b_debug/seed_selection_table.csv` columns.
- Verified `.venv\Scripts\python.exe -m unittest discover -s tests` runs 35
  tests successfully.
- Verified `.venv\Scripts\python.exe reproducibility\verify.py --fast`
  remains `PASS_WITH_WARNINGS`, expected while the release manifest is stale.

## 2026-05-25 - Documentation and CI wiring

- Added `.github/workflows/verify.yml`.
- Updated README, reproduction guide, environment notes, reproducibility
  philosophy, and file inventory so the fast verification path is easy to find.
- Updated `reproducibility/verify.py` to support CI clean checkouts with
  `REPRO_CI_ALLOW_MISSING_ARTIFACTS=1`, downgrading absent large
  `data/`/`results/` artifacts to structured warnings.
- Added CI workflow tests and a missing-large-artifact warning regression test.
- Verified `.venv\Scripts\python.exe -m unittest discover -s tests` runs 40
  tests successfully.
- Verified `.venv\Scripts\python.exe reproducibility\verify.py --fast`
  remains `PASS_WITH_WARNINGS`.

## 2026-05-25 - Submission-minimum readiness review

- Added `SUBMISSION_MINIMUM_READINESS.md`.
- Resolved the remaining environment-fingerprint placeholder in
  `baseline_commit.yaml` by recording the current `pip_freeze_sha256`.
- Added a submission-minimum placeholder regression test.
- Verified local tests, environment, reference manifest, paper mapping, and
  fast verifier.
- Rebuilt Docker image `zne-verify` and verified `docker run --rm zne-verify
  make verify` exits successfully with documented warnings.
- Recorded that final plain `PASS`, baseline commit, regenerated release
  manifest, DOI, and Zenodo rebuild remain deferred to final archival
  integration.

## 2026-05-27 - FDR reconstruction audit

- Updated `scripts/verify_numerical_reproducibility_subset.py` so Phase C and
  Phase D classifications are reconstructed from `p_conjunction` using the
  recorded BH/BY/Holm procedures instead of fixed integer constants.
- Added adversarial tests for corrupted adjusted q-values and corrupted
  classification labels.
- Updated current verifier wording and version to remove internal workstream
  labels from the user-facing reports.
