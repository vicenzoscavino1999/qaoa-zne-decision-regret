Post hoc oracle allocation sensitivity (2026-10-20)

Requirements: Python 3.12 and the project's existing NumPy dependency.
Run from the project root:

python revision_20261015/allocation_sensitivity_20261020/allocation_sensitivity.py --output ../allocation-recomputed.json

Use a new output file outside the release to preserve the archived JSON and its manifest.

The project root is derived from the script location; --root PATH overrides it.
The script reads the existing confirmation protocol, graphs, exact probability
caches and ideal landscapes. It verifies cache hashes against their existing
manifest and records root-relative hashes of every numerical input used,
including each ideal landscape. It writes only the explicit --output JSON.
No machine-specific dependency paths are embedded in the script.

For fixed total shots B, independent scales have variance
V=sum_j lambda_j^2 v_j/n_j. The continuous pointwise oracle uses
n_j proportional to abs(lambda_j)*sqrt(v_j), with
V*=(sum_j abs(lambda_j)*sqrt(v_j))^2/B. The exact positive-integer oracle
starts with n_j=1 and assigns remaining shots by the largest decreasing
marginal reduction lambda_j^2*v_j/[n_j(n_j+1)]. A fixed-per-cell oracle
instead minimizes the grid-mean variance with one allocation vector.
Recorded implemented allocations are read from the frozen protocol.

This is a post hoc sensitivity analysis with oracle knowledge of variances.
An experimental allocation would need calibration resources charged to its
budget. The calculation draws no shots, runs no circuits, and does not rerun
regret, argmin selections, or confirmatory tests. Exact moments do not establish
the decision-endpoint outcomes under a changed allocation. The four existing
confirmation cells are the intended scope. No source data are modified.
