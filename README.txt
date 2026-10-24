REPRODUCIBILITY PACKAGE - 1.0.0
Decision-aware validation of zero-noise extrapolation in QAOA optimization landscapes
Vicenzo Scavino Alfaro

STATUS AND SCOPE
This package accompanies the study named above.
Repository: https://github.com/vicenzoscavino1999/qaoa-zne-decision-regret
Archived version 1.0.0: https://doi.org/10.5281/zenodo.23607395
Release date: 2026-10-23
The single current manuscript is MANUSCRITO_ACTUAL/manuscrito.pdf (27 pages, six appendices,
34 cited works); editable sources are in MANUSCRITO_ACTUAL/fuentes/.

The package contains the saved historical seed-level metrics and reanalysis,
the bounded bias-variance control, all four-condition confirmation inputs,
probability caches and sampled landscapes, allocation sensitivity, and current
figure/source files. The historical screen is exploratory: its 31-bit seeds
collide. Reproducing its numbers does not certify its independence or significance.
The fresh confirmation concerns four selected conditions, not prevalence among
arbitrary graphs. The two H8 orders share a baseline. Reported tests concern
two marginal directional probabilities, and regret effects are small.

The original 27,015 historical raw C/D arrays are not included in this compact
package. Twelve historical ideal arrays ARE included for native effect sizes.
Historical verification here starts from the 20,250 saved seed-metric records;
the earlier full-archive audit is retained as provenance, not rerun by this package.
Complete fresh confirmation landscapes and its 400 paired endpoints are included.

ENVIRONMENT
Use Python 3.12.10 exactly: the frozen confirmation verifies that version,
NumPy 2.2.6, SciPy 1.17.1, Qiskit 1.4.5 and Qiskit Aer 0.15.1.
requirements-reproduction-lock.txt pins the complete installed distribution set
from the independently created Windows verification environment. The original
requirements-lock.txt records direct dependencies. Neither file bundles Python,
packages, fonts, Poppler or LaTeX. Other platforms have not been certified.

Create the environment OUTSIDE the extracted package so integrity checking stays
read-only. From the extracted package root, with Python 3.12.10 selected:
  python -m venv ../qaoa-env
Activate it using the usual command for your shell, then:
  python -m pip install -r requirements-reproduction-lock.txt
  python -m pip check
For Windows PowerShell activation: ../qaoa-env/Scripts/Activate.ps1
For a POSIX shell activation: . ../qaoa-env/bin/activate
Alternatively call the environment's Python executable explicitly.

READ-ONLY NUMERICAL VERIFICATION
From the extracted root:
  python -B reproducibility/verify_release.py --report ../verification.json

This verifies the full payload manifest, recomputes all 675 historical condition
summaries and independent binomial/multiplicity calculations, checks the control's
24 configurations, 3456 pointwise entries and 12,000 repeated outcomes, reconstructs
400 confirmation endpoints, verifies the 90-file protocol lock and 1600 recorded
stream identities, replays the first three points per stream, compares six bounded
Aer cases, verifies both current main manuscript tables numerically, and recalculates
four allocation-sensitivity conditions from saved exact probabilities. It then
checks that every packaged file remains unchanged. A root .git directory or
worktree pointer is excluded from the payload check; other extra files are rejected. Reports go outside the package.
No long historical campaign or new confirmatory sampling is launched.

Individual components:
  python -B revision_20261015/verify_revision.py
  python -B confirmation_20261019/verify_confirmation.py
  python -B revision_20261015/allocation_sensitivity_20261020/allocation_sensitivity.py --output ../allocation-recomputed.json
Do not overwrite the archived allocation result or regenerate the frozen protocol.

The manuscript checker validates the two labelled table schemas and displayed
rounding against saved results. It accepts ordinary whitespace/style changes;
it intentionally rejects unsupported table layouts. It is not a TeX interpreter.

REGENERATE THE THREE CURRENT FIGURES
Install Poppler (pdfinfo, pdftotext, pdftoppm) and place its executables on PATH:
  python -B reproducibility/regenerate_figures.py --output ../figures-reproduced
The output folder must be NEW and external. If needed, add:
  --pdf-bin "PATH TO POPPLER EXECUTABLES"
The generator uses saved statistics/control outputs only and compares geometry,
text and 150-dpi pixels against the three manuscript PDFs. Matching fonts and
plotting/rendering versions are required for pixel equality: the current figures
use Arial, with Helvetica/DejaVu Sans fallbacks. The supplied PDFs remain the
reference on systems without Arial. PDF creation dates may change byte hashes.
No historical proposal folder is required by this standalone generator.

COMPILE THE SINGLE MANUSCRIPT
Use a working LaTeX installation with pdflatex and bibtex. First copy
MANUSCRITO_ACTUAL/ to a separate directory, preserving the delivered package:
  python -c "import shutil; shutil.copytree('MANUSCRITO_ACTUAL', '../manuscript-build')"
  python ../manuscript-build/compilar.py
If LaTeX executables are not on PATH, add --tex-bin "PATH TO LATEX EXECUTABLES".
The build writes only in that separate copy. PDF metadata may change; compare
content rather than requiring an identical PDF hash. No separate supplement
or old build_manuscripts.py is needed.

FILES AND PROVENANCE
- REVIEW_BUNDLE_MANIFEST.json: SHA-256 for every payload file, excluding itself.
- reproducibility/: release builder, current table checks, aggregate verifier,
  figure generator and package-local instructions.
- confirmation_20261019/protocol_lock.json: original frozen 90-file lock.
  All src/ modules, including unused Phase E code, are included because the lock
  names them. Their inclusion is not evidence that those additional studies ran.
- confirmation_20261019/inputs, probabilities, samples, analysis: complete fresh
  confirmation material, including shared H8 baseline and exact stream records.
- revision_20261015/statistics and bias_variance: corrected historical summaries
  and saved bounded-control moments/outcomes.
- revision_20261015/allocation_sensitivity_20261020: exact-moment sensitivity.
- revision_20261015/prior_audit and followup_review/math_code: dated provenance
  reports, including the historical seed limitation; not new release validation.
- manuscript_qst/tables: three historical auxiliary DATA tables only. No older
  manuscript or separate supplement is presented as current.
- CITATION.cff and .zenodo.json: version-specific citation and deposit metadata,
  including the repository URL, archival DOI and release date.
- .gitattributes: disables line-ending conversion so Git preserves frozen bytes.
- .gitignore: excludes local execution artifacts without ignoring packaged data.

REBUILD THE ARCHIVE
Run from this package or the working project, with NEW external output paths:
  python -B reproducibility/build_release.py --destination ../new-release-stage --archive ../new-release.zip
The builder selects explicit scientific inputs and current manuscript sources,
checks the frozen lock, creates a new manifest, and fixes ZIP timestamps/order.
Identical input bytes and the same Python/zlib produce repeatable ZIP bytes;
cross-toolchain compression identity is not promised.

CITATION AND VERSION
Cite version 1.0.0 using DOI 10.5281/zenodo.23607395. CITATION.cff provides structured
citation metadata. The Git tag v1.0.0 identifies the same source snapshot.
The original confirmation protocol and its locked scientific files are retained
unchanged. Later changes should receive a new version and an updated manifest.
Code and data/document licenses are summarized in LICENSE_BUNDLE.md.
