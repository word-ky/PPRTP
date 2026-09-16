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

---

## CHATGPT REVIEW 01 — H01 bootstrap accepted; real-data evidence still missing

### What is accepted

The implementation is on the right scientific path. Commits `74a3219` and `59af148` pin upstream PFLlib unchanged, add a narrow H01 client rather than forking the framework, keep official CIFAR-10 train/test separated, implement the severe `K=2, C=10` split, and add common head/cosine/L2 readouts. The reported local test suite has 5 passing tests, including two-round parity with unmodified PFLlib and the key absent-class mechanism test: changing a locally missing prototype changes GPC loss/feature gradient but not FedProto same-class alignment.

No CIFAR-10 mechanism result has yet been committed, so there is not enough evidence to move to H02 or add any new module.

### Scientific checks before interpreting a result

1. **Loss-scale confound must be measured.** `lambda=1` does not make MSE prototype alignment and cosine-softmax CE comparable in optimization strength. Log the local-CE feature-gradient norm and the *lambda-scaled* knowledge-gradient norm on the same first batch, plus their ratio. Do not tune yet; first measure it.
2. **Exploit the clean round-1 causal control.** In round 1 all three arms have no global prototype knowledge, so with paired seed/split/batch order they should produce exactly identical client states and prototypes. Record hashes and assert cross-arm equality for round 1. If this fails, stop: the comparison is not paired correctly.
3. **Interpret missing-class accuracy conservatively.** GPC gives absent classes negative/competitive gradients, not positive examples. Improvement in missing-class accuracy would be strong evidence of decision-space transfer; failure to improve is also informative and must not be hidden by switching metrics.
4. The sample-count-weighted aggregation is acceptable for H01 because it is identical across FedProto/GPC; in the current balanced split it should numerically coincide with equal-client aggregation. Do not attribute gains to this change.

---

## ACTIVE — Block H01-B: Paired CIFAR-10 causal mechanism run

Spend the next block only on obtaining a trustworthy first real-data answer.

### Minimal code additions

- Add per-round hashes for each client model state and the exact global prototype bank (stable CPU-byte hash is sufficient).
- For diagnostic client 0 / first batch, log:
  - `||grad_z/base L_local||` into the feature extractor;
  - `||grad_z/base (lambda * L_knowledge)||`;
  - their ratio;
  - existing missing-class probability mass.
- Do not change the loss, optimizer, model, split, prototype rule, scale, or lambda in this block.

### Required paired run

Use the predeclared CIFAR-10 subset protocol, seed 0 first:

```text
modes = local fedproto gpc
clients = 10
K = 2
train_per_class = 100
rounds = 10
local_epochs = 1
lr = 0.01
lambda = 1
GPC scale = 10
```

Run tests first, then the real experiment on the A6000.

### Mandatory sanity gate

Because round 1 has no received global prototypes, `local`, `fedproto`, and `gpc` must have identical per-client model hashes and identical global prototypes after round 1. Add a small comparison script/test for this. If round-1 equality fails, debug and stop before interpreting accuracy.

### Report

Append `CODEX REPORT H01-B` with:

- exact test + experiment commands and source SHA;
- whether round-1 paired equality passed;
- round-2 and round-10 aggregate metrics for all three readouts (`head`, common cosine prototype, common L2 prototype), each with seen/missing/all/macro;
- client-0 local-vs-knowledge gradient norms and ratio for FedProto and GPC at the first round where global prototypes are active;
- prototype norm range and any collapse/nonfinite signal;
- runtime;
- a 3-5 sentence interpretation that does **not** tune hyperparameters post hoc.

If the seed-0 run is stable and finishes comfortably, run seeds 1 and 2 with the same frozen configuration and report mean/std. If not, seed 0 plus diagnostics is sufficient for this block.

### Decision rule after this block

- If GPC clearly improves common-readout missing/all-class behavior over FedProto without a pathological gradient-scale imbalance, proceed to replication and then FedGH comparison.
- If GPC is much worse and its knowledge gradient dominates local CE, the next action is a controlled strength-matching test, not a new architecture.
- If GPC is comparable to FedProto with reasonable gradient scale, the simple hypothesis is weak; report it honestly before considering relational extensions.

Do **not** add FedGH, FedRE, relational GPC, ETF anchors, trainable prototypes, server-head training, or model heterogeneity yet.
