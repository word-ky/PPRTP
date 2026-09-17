# PPRTP — H01 on PFLlib

Research task and reports: [CHATGPT_CODEX_BRIDGE.md](CHATGPT_CODEX_BRIDGE.md).
Pinned upstream: Jianqing Zhang's [PFLlib](https://github.com/TsingZ0/PFLlib),
revision `0169ba7e412c9856a08bb3faefab1e35f538a3c1`.
See [PROVENANCE.md](PROVENANCE.md) and [protocol](research_log/H01_plan.md).

## Reproduce

```bash
git submodule update --init --recursive
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
PPRTP_SOURCE_SHA=$(git rev-parse HEAD) python -m pprtp.run \
  --data /path/to/cifar10 --output research_log/h01 \
  --device cuda:0 --seeds 0 1 2
```

Dependencies: Python, PyTorch, torchvision, NumPy, scikit-learn, h5py.
Validated remote environment: Python 3.12, Torch 2.4.0+cu121,
torchvision 0.19.0+cu121. No dependency upgrades were made to the shared environment.
The runner uses a small real CIFAR-10 subset, not a benchmark-scale reproduction.

Three selectable training modes share the upstream CNN and prototype collection:
`local` (CE), `fedproto` (CE + same-class raw-feature MSE), `gpc` (CE + detached
all-valid-class cosine CE). All use the same split, initial weights, batches and SGD.
Protocol adaptation: no dropped final batch, so each allocated training example
contributes; otherwise upstream online prototype collection is preserved.

Each run writes metadata, exact split indices, per-round/per-client metrics,
class counts/correct counts, loss terms, client-0 gradient/probability diagnostics,
prototype cosine matrices/norms and a representative client-0 checkpoint.
All three inference rules are reported for all training modes. The local model's
prototype readouts require posthoc communication and are diagnostics, not a claim
of communication-free local deployment. Raw prototype communication is identical
between FedProto/GPC; both additionally send class counts for weighted reduction.
Payload fields count float32 vectors and int64 counts, excluding serialization and
class-ID headers. The local row records hypothetical diagnostic aggregation traffic.

Remote execution uses project-local `scripts/autodl-{deploy,run,logs,fetch}.ps1`
and private `.autodl/config.json`; artifacts stay under `research_log/` locally
and `/home/wenchang/asdasdsad/wjq/PPRTP` remotely.

H01-D (research-lead instruction in BRIDGE) uses `--modes fedproto gpc_all_match
gpc_seen_match --seen-lamda 0.03498`. FedProto retains lambda 1, all-class GPC
uses .002, and the seen-only value is fixed before execution. Omitting
`--seen-lamda` preserves H01-C's .002 setting. Client-0 diagnostics compute both
unscaled GPC feature gradients on the same pre-update tensor and prototype bank;
their norm ratio concerns `dL/dz`, while the strength gate concerns gradients into
base parameters. Reports preserve these as separate quantities.
