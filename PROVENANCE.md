# Code provenance

PFLlib by Jianqing Zhang and collaborators: https://github.com/TsingZ0/PFLlib
Pinned revision: `0169ba7e412c9856a08bb3faefab1e35f538a3c1`.
The complete, unchanged source and Apache-2.0 license are in `vendor/PFLlib` (git submodule).
Initialize with `git submodule update --init --recursive` after cloning PPRTP.

`pprtp/client.py` adapts `system/flcore/clients/clientproto.py`'s training loop.
It inherits client/model/optimizer state, calls upstream `agg_func` and `set_protos`,
and imports the upstream CNN without copying it. Changes: configurable injection,
actual class representation counts, and H01 diagnostics. Slow-client simulation
is disabled for H01. Tests compare full two-round weights/prototypes to upstream.

H01 server reduction weights client means by class sample counts; upstream uses
equal client weighting. Both H01 training arms use exactly the same rule.
Dataset partition/evaluation adaptations are documented in research_log/H01_plan.md.

Remote-control PowerShell scripts copied unchanged from the user's existing
`D:/work/claude-autodl/autodl-workflow-clean/scripts` workflow, except fetch output
is directed into project-local research_log/remote_runs.
