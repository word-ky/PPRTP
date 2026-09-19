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

H02-A adds `pprtp/fedgh.py` as a minimal shared-head control around the same
PFLlib CNN, client SGD and online per-class feature means. Dedicated server
SGD (.01, no momentum/decay) performs 20 batch-size-one steps, sorted by client
ID and class ID, per round. This protocol implements the research lead's
FedGH-style mechanism; it is not a reproduction of FedGH's published benchmark.


## H15-A matched FedTGP port

Official algorithm source: https://github.com/TsingZ0/FedTGP/tree/c77cbbb31eb30d13066cd11f7f4a2e732aeaae24,
pinned unchanged as vendor/FedTGP (Apache-2.0). Primary files are
system/flcore/clients/clienttgp.py, servers/servertgp.py, clients/clientbase.py,
and system/run_me.sh. pprtp/fedtgp.py adapts the generator, equal-client mean
margin calculation, individual-prototype server SGD, and observed-label MSE.
Existing PFLlib H01Client supplies model/SGD/CE/MSE; pprtp.run supplies split,
local batches, evaluation and logging. Both upstream repositories stay unedited.

Pinned clientTGP.train calls collect_protos before saving the newly trained model;
collect_protos reloads the prior checkpoint using torch.load. The matched port
preserves these round-start eval-mode class means through a snapshot. It does
not silently substitute online or post-update means. A test executes the pinned
class methods (AST-extracted only to avoid colliding flcore package imports) and
compares two cycles of model updates and uploaded means. A server test compares
initial generator weights, adaptive gaps and SGD updates to pinned methods.

Assigned constants follow author run_me.sh: lambda10, server_epochs100,
margin_threshold100, serverlr=locallr=.01, hidden=featuredim512. Uploads train
individually; unweighted client averages only compute per-class minimum distance,
whose maximum capped at100 is the positive-class margin. Server objective uses
negative Euclidean distance; official prediction minimizes squared Euclidean/MSE
across all100 generated prototypes. The local head is diagnostic only.

Deliberate matched-protocol changes: historical H12-A full CIFAR100 split and
homogeneous CNN initialization; all49744 nonanchor samples; batch32 shuffled
local loaders and drop_last=False (upstream client loader is unshuffled and
drop_last=True); exactly10clientcycles rather than global_rounds+1; final
post-server readout; in-memory state instead of disk roundtrips. Server RNG
is deterministically seeded0, with a private shuffle generator; vectorized
margin/prediction calculations preserve the equations. No parameter tuning.
FedTGP consumes neither reserved anchors nor their labels/features/correspondence.
No server weights or PPRTP transforms are sent to clients, only prototypes.

This is a matched short-budget baseline, not an exact published-dataset recipe
or a claim of converged/best FedTGP performance. The official README discusses
potentially much longer convergence (>1000 communication iterations); the frozen
research assignment intentionally matches only10localcycles. Report that limit,
round-start upload timing, all server costs, and PPRTP extraanchor sideinformation.
