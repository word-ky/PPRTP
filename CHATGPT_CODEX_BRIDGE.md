# ChatGPT ↔ Codex Bridge

This file is the persistent coordination log. Do not delete history. ChatGPT writes the active scientific task; Codex executes it and appends evidence/results.

---

## ACTIVE — Block H01: Bootstrap + falsify the simplest GPC hypothesis

### Scientific hypothesis

FedProto's class-missing weakness may come less from the limited capacity of prototypes and more from **how prototypes are injected into local optimization**. Original FedProto mainly uses a same-class attraction term. A Global Prototype Classifier (GPC) instead uses every global class prototype in one softmax, so locally missing classes become explicit competitors in every local classification gradient.

For normalized feature `z` and global prototypes `P={p_c}`:

```text
logit_c = scale * cosine(z, p_c)
L_GPC = CE(logits over ALL globally known classes, y)
```

Primary comparison must keep prototype construction and aggregation identical between FedProto and GPC whenever possible.

### Objective of this block

Create the smallest reproducible codebase that can run a **mechanism test**, not a publication-scale benchmark.

### Required implementation

1. Bootstrap a compact PyTorch FL harness with configuration-driven algorithm selection.
2. Dataset: CIFAR-10 as the first real dataset. If download/runtime blocks execution, add a deterministic synthetic classification dataset only as an implementation smoke test; do not substitute synthetic results for the CIFAR-10 mechanism test.
3. Client split: implement a severe label-subset split where each client owns only `K` of the `C` classes. Make `K` configurable; initial target `K=2`, `C=10`. Log the class set of every client.
4. Model: use one simple common backbone first (e.g. small CNN/ResNet18 if already readily available). We are isolating class-missing heterogeneity before model heterogeneity.
5. Implement three selectable modes:
   - `local`: local CE only;
   - `fedproto`: local CE + original-style same-class global-prototype alignment;
   - `gpc`: local CE + all-class global prototype softmax.
6. Global prototype aggregation: class-wise, sample-count-aware aggregation over clients that observed that class. Track a validity mask for classes not observed in a round. Never silently treat an invalid class prototype as a real zero prototype.
7. GPC details for first test:
   - L2-normalize feature and valid global prototypes;
   - use cosine logits;
   - configurable `scale` or temperature;
   - global prototypes are detached targets in local training;
   - all valid global classes enter the denominator, including classes absent locally;
   - if a class has never received a valid global prototype yet, exclude it using the validity mask and log this event.
8. Metrics must include, per client and aggregated:
   - accuracy on classes present in that client's training set (`seen-class acc`);
   - accuracy on classes absent from that client's training set (`missing-class acc`);
   - all-class test accuracy;
   - macro-per-class accuracy if easy to add.
9. Add at least one unit/smoke test proving the key mechanism: for a client lacking class `c`, changing global prototype `p_c` must change GPC loss/gradient on a local sample; the same change must not affect the pure same-class FedProto prototype-alignment term for that sample.

### Very important experimental fairness

For the first comparison, do **not** give GPC extra semantic information. FedProto and GPC should use the same client prototypes, same aggregation rule, same backbone, same client split, same optimizer/local epochs/rounds, and same seeds. The intended causal difference is knowledge injection: same-class attraction vs all-class competition.

### First run matrix

Start tiny and fast; expand only if code is stable.

```text
algorithms = [local, fedproto, gpc]
K_classes_per_client = 2
num_clients = small but enough to cover all 10 classes
seeds = [one smoke-test seed first]
```

Then, if stable, run the same configuration with at least two additional seeds.

### Diagnostics to save

For at least one representative client:

- local class set;
- global prototype validity mask per round;
- cosine similarity matrix among global prototypes;
- mean GPC probability assigned to locally missing classes on seen-class samples;
- gradient norm into the feature extractor from the global knowledge term;
- separately log `L_local` and `L_proto`/`L_gpc`.

### Deliverables to append below

Codex must append a `CODEX REPORT H01` section with:

- STATUS: DONE / PARTIAL / BLOCKED
- repository structure created
- exact commands to reproduce
- implementation notes
- smoke-test/unit-test results
- first experiment table if completed
- any bug or scientific ambiguity noticed
- commit SHA(s)
- recommended H02 action, but do **not** independently change the scientific objective

### Stop / escalation conditions

Stop adding features and report immediately if any of these occur:

- GPC cannot be compared with FedProto under the same prototype construction;
- missing-class metric is ill-defined because of the evaluation loader/split;
- global prototype validity causes label-index mismatch;
- results are suspiciously identical across algorithms;
- numerical instability / collapsed prototypes appears.

Do not add FedRE, relational GPC, learnable prototypes, ETF anchors, server-side head training, or model heterogeneity in H01. Those are later stages contingent on this falsification test.

---

## CODEX REPORT H01

_Pending. Codex: append your report here; do not overwrite the ACTIVE task above._

### CODEX PROGRESS H01 — 2026-09-16 bootstrap / PFLlib requirement

User explicitly requested Jianqing Zhang's PFLlib. Pinned official TsingZ0/PFLlib
0169ba7e412c9856a08bb3faefab1e35f538a3c1 as unchanged submodule.
Detailed execution protocol and reuse map: research_log/H01_plan.md.
A6000 root created: /home/wenchang/asdasdsad/wjq/PPRTP. CUDA works despite NVML error.
Scientific ambiguity recorded before implementation: upstream FedProto inference
uses raw nearest-prototype L2, whereas GPC suggests cosine. To isolate training
injection, evaluate every arm with both common prototype readouts plus the local
head, explicitly distinguishing local-only deployment from posthoc prototype probes.
Upstream client-unweighted aggregation changed to H01-required sample weighting
identically for FedProto/GPC. Original client path will be reproduced separately.
