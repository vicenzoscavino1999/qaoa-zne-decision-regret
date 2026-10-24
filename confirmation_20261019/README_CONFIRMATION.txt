TARGETED CONFIRMATION - PACKAGE INSTRUCTIONS UPDATED 23 OCTOBER 2026

The frozen study and scientific files date from 19 October 2026. Its four selected
conditions pass IUT/Holm using 100 fresh repetitions each. The historical
540-condition screen remains exploratory because its 31-bit seeds collide.
The confirmation is conditional on those four selected targets and is not a
prevalence estimate. The two H8 conditions share a noisy baseline.

Current manuscript: MANUSCRITO_ACTUAL/manuscrito.pdf, one document with appendices.
Read REPRODUCIBILITY.txt at the project/package root for the tested environment,
complete instructions and limits. Use Python 3.12.10 exactly and the supplied
requirements-reproduction-lock.txt in an environment outside the package.

Read-only verification from the root:
  python -B confirmation_20261019/verify_confirmation.py
Or check the whole compact release:
  python -B reproducibility/verify_release.py --report ../verification.json

The confirmation check verifies the frozen 90-file lock, all 400 saved endpoints,
exact binomial/IUT/Holm arithmetic, 1600 stream paths/states, three replayed points
per stream, six bounded Aer comparisons and current manuscript table cells.
It does not replay every sample or prove mathematical independence of PRNGs.

PROTOCOL.txt, protocol.json and protocol_lock.json preserve the locally timestamped
protocol; it was not externally registered. Inputs, probability caches, complete
new landscapes, stream metadata and analysis outputs are included unchanged.
Do not run prepare_protocol.py to replace the protocol or use the old manuscript
integration/release generators. Optional full replay belongs in a separate clean
copy and is not required by the documented verification command.

GitHub/Zenodo publication remains pending. No repository URL or DOI is assigned.
