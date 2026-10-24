HISTORICAL REANALYSIS - CURRENT PACKAGE INSTRUCTIONS (23 OCTOBER 2026)

This directory contains the post-analysis reanalysis of the 135 discovery and
540 replication conditions, plus the bounded bias-variance control. The archived
31-bit random seeds collide. The four Holm / four BY / twelve BH replication
counts are numerically reproducible, model-based exploratory results; PASS does
not certify inferential calibration for that historical generator.

The primary independent-stream evidence is the four-condition targeted confirmation
in confirmation_20261019. See REPRODUCIBILITY.txt at the package root for the
current environment, overall verification and scientific limits.

Use Python 3.12.10 and the supplied requirements-reproduction-lock.txt.
  python -B revision_20261015/verify_revision.py
This recalculates the 675 condition summaries from saved seed CSVs, independently
checks directional p-values and adjustments, reconstructs control identities and
regret, and checks the current unified manuscript tables numerically.
It writes only temporary files/stdout unless --report is supplied.

The single current manuscript and its editable sources are in MANUSCRITO_ACTUAL/.
Its figures are regenerated without new sampling by:
  python -B reproducibility/regenerate_figures.py --output ../figures-reproduced
The output directory must be new and external. Follow the root instructions for
Poppler and exact visual-comparison requirements.

Do not use historical build_manuscripts.py/main_body.tex/supplement_template.tex
to edit the current manuscript. The old builders are omitted from this release.
Saved protocol/source/data hashes are preserved. The 27,015 original raw C/D
arrays are outside this compact package; verification starts from the archived
seed metrics. The twelve D ideal grids and complete new confirmation data are
included. Older audit reports describe their original scope and execution date.
