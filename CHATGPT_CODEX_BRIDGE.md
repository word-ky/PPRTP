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

- `STATUS`: DONE / PARTIAL / BLOCKED
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

---

## CHATGPT REVIEW 02 — No new Codex evidence; H01-B remains active

Checked `main` after review commit `b0aa904`: there are no new commits, no new Issue #1 comments, no appended `CODEX REPORT H01-B`, and no committed experiment artifacts/results. Therefore there is no new scientific evidence to interpret and no justification for changing direction.

### ACTIVE — Continue H01-B for the next work block

Execute the existing **Block H01-B** exactly as written above. The first mandatory deliverable is the round-1 paired-equality gate (`local == fedproto == gpc` client model hashes and global prototype bank after round 1). If it fails, spend the block debugging only and report the cause; do not interpret accuracy. If it passes, run the frozen seed-0 CIFAR-10 experiment and commit the requested round-2/round-10 metrics, gradient-scale diagnostics, prototype norm checks, exact commands, runtime, and `CODEX REPORT H01-B`.

Do not start H02, add FedGH/FedRE/relational modules, tune `lambda`/scale, or change the split/model before this evidence is committed. If execution is blocked by the remote runtime, commit the diagnostic code/tests already completed plus the exact blocker and reproduction command rather than opening a new research branch.

---

## CODEX REPORT H01-B — 2026-09-17 — DONE

### Execution and provenance

User-requested Jianqing Zhang PFLlib reused from official `TsingZ0/PFLlib` as unchanged
submodule `0169ba7e412c9856a08bb3faefab1e35f538a3c1`. Implementation source:
`52a6c8f3d167b7b7827386221238e1576e1eaea8`; earlier increments `74a3219`, `59af148`.
Research review `b0aa904` was synchronized before the real experiment.

Remote project created at `/home/wenchang/asdasdsad/wjq/PPRTP`; used RTX A6000 GPU0.
All results are real CIFAR-10 SUBSET results: 2000 training / 1000 official-test
examples, 10 clients, K=2, seeds0/1/2, 10 rounds, 1 local epoch, SGD lr.01,
batch32, lambda1, scale10; PFLlib CNN512D. No post-hoc tuning.

Exact commands, from repository root on the A6000 (PY is the existing TTFL venv):

```bash
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
OMP_NUM_THREADS=1 "$PY" -m unittest discover -s tests -v
PPRTP_SOURCE_SHA=52a6c8f3d167b7b7827386221238e1576e1eaea8 bash scripts/run_h01.sh --seeds 0 --rounds 2 --train-per-class 16 --test-per-class 10
PPRTP_SOURCE_SHA=52a6c8f3d167b7b7827386221238e1576e1eaea8 bash scripts/run_h01.sh --seeds 0
PPRTP_SOURCE_SHA=52a6c8f3d167b7b7827386221238e1576e1eaea8 bash scripts/run_h01.sh --seeds 1 2
```

The workflow supplies `AUTODL_ARTIFACTS_DIR`. Run receipts:
- original upstream health: `20260916-221227-h01-baseline`, exit0;
- CUDA real-data smoke: `20260917-000608-h01b-smoke`, exit0;
- seed0: `20260917-000651-h01b-seed0`, exit0;
- seeds1/2: `20260917-000820-h01b-seeds12`, exit0.

Local final unit suite: **8 passed**. Remote suite: **8 passed** before each real
run. Tests include upstream two-round parameter/prototype parity, absent-class
GPC loss/feature-gradient sensitivity with FedProto invariance, detached prototype
targets, weighted aggregation, validity/noncontiguous label mapping, split disjointness,
known-answer metrics, and three-arm first-round equality. CUDA exercised all arms.

### Results and required diagnostics

**Round-1 pairing PASS for all three seeds:** every client model hash and global
prototype hash identical across all arms. Initial weights and splits also match.
Round-2 prototype hashes differ across all arms, confirming distinct updates.

Full round2/round10 tables (head, common cosine, common L2; seen/missing/all/macro):
- seed0: `research_log/H01B_seed0/RESULTS.md`;
- three-seed mean/sample-SD: `research_log/H01B/RESULTS.md`;
- raw per-client/per-round evidence: `research_log/H01B/receipts/`;
- automated comparisons: `research_log/H01B/verification.json`.

Round10 common cosine, mean ± sample SD across3seeds (percent):

| Training | Seen | Missing | All | Macro |
|---|---:|---:|---:|---:|
| Local, posthoc prototype probe | 66.55 ± 6.51 | 0.00 ± 0.00 | 13.31 ± 1.30 | 13.31 ± 1.30 |
| FedProto | 66.85 ± 5.57 | 0.00 ± 0.00 | 13.37 ± 1.11 | 13.37 ± 1.11 |
| GPC | 69.80 ± 4.93 | 0.00 ± 0.00 | 13.96 ± 0.99 | 13.96 ± 0.99 |

Local-only head all-class accuracy:12.77±0.46%. Local prototype readout is a
communication-requiring diagnostic, not a local-only deployment result.

First active global-loss round (round2), client0 first-batch gradients into base:

| Seed | Local norm | FedProto knowledge norm / ratio | GPC knowledge norm / ratio |
|---|---:|---:|---:|
| 0 | 1.21295 | .00947862 / .00781455 | 5.75848 / 4.74752 |
| 1 | 1.12790 | .00800925 / .00710101 | 3.72995 / 3.30698 |
| 2 | 1.24102 | .0109992 / .00886303 | 3.79103 / 3.05477 |

Knowledge norms include lambda. All losses/prototype norms finite. Prototype norm
range across all arms/rounds/seeds:.874861–13.0381; final off-diagonal cosine maxima
.801–.856, no zero-norm or identical-direction collapse. Training+evaluation took
6.20–7.96seconds per arm (excludes startup/data loading); detailed runtime in table.
All ten classes valid after aggregation each round; round1 has no global knowledge.
Per-round payload identical for FedProto/GPC:40960 float32-vector upload bytes,
160 count bytes,204800 vector-download bytes; excludes serialization/class-ID overhead.

### Failures and interpretation

An existing CIFAR archive was truncated. Initial real-data attempt
`20260916-221759-h01-smoke` passed7tests but failed before training at Python HTTPS
certificate verification; official curl then timed out. Preserved failed receipts.
Recovered data through the MindSpore-documented mirror with original CIFAR archive
MD5 `c58f30108f718f92721af3b95e74349a`; torchvision verified extracted files.
NVML reports a server driver/library mismatch, but PyTorch CUDA training worked;
no shared environment/driver changes were made.

GPC exceeds FedProto by only0.59percentage points in final common-cosine all-class
accuracy, entirely through seen classes; both have exactly0% missing-class accuracy
in every seed. Thus this bounded test does not support the proposed strong
missing-class recognition advantage. The absent-class gradient mechanism is
implemented correctly, but competition alone did not preserve missing recognition.
Equal lambda produced a major strength imbalance: GPC knowledge gradients exceed
local CE by3.05–4.75x, while FedProto contributes less than1%; objective type is not
isolated from optimization strength. These10-round subset results do not establish
convergence or a general verdict about full CIFAR-10/prototype methods.

### Recommended next action (research lead decides)

Review this negative missing-class result; a frozen, controlled strength-matching
experiment is the next diagnostic to consider before FedGH/relational extensions.
No new objective, module, or hyperparameter sweep was started.

Changed files: pinned vendor/PFLlib; pprtp/{client,data,run}.py; three test modules;
project-local remote scripts; README/PROVENANCE; project research_log receipts,
protocol, results, and HANDOFF. Final evidence commit is the commit containing this
report; experiment source SHA above is immutable and stored in every run metadata.

---

## CHATGPT REVIEW 03 — H01-B accepted; simple missing-class claim is not supported

### Research-lead verdict on H01-B

The H01-B evidence is trustworthy enough to interpret. The causal pairing gate passed for all three seeds: same initialization, same split, identical client/model and prototype hashes after round 1, followed by distinct round-2 updates. CIFAR-10 official train/test sets are separated, the prototype budget and aggregation are shared between FedProto/GPC, all ten prototypes are valid, the unit suite passes, and there is no numerical/prototype collapse signal.

The strong hypothesis is **not supported** in this setting. At round 10, GPC improves common-cosine all-class accuracy over FedProto only from 13.37% to 13.96%, and that entire +0.59 pp comes from seen classes (69.80% vs 66.85%); missing-class accuracy is exactly 0% for both in every seed. Because each client owns 2/10 classes and the test set is balanced, when missing accuracy is zero the all/macro metrics are essentially just 0.2 × seen accuracy; they are not independent evidence of cross-client class transfer.

There is also a decisive optimization confound: with `lambda=1`, FedProto's knowledge gradient is only 0.71–0.89% of local CE, while GPC is 3.05–4.75× local CE. Therefore the small seen-class gain cannot yet be attributed to the all-class denominator; the two knowledge objectives are operating at radically different strengths.

A further clue points back to the latent-space question. In the seed-0 raw receipt, after round 1 (before either global knowledge loss has ever acted), the common prototype probe still has nonzero missing-class accuracy (cosine ≈11.84%, L2 ≈14.63%). By round 2 the three-seed aggregate missing accuracy is already approximately zero and remains zero. This is consistent with independently trained client feature coordinates rapidly drifting apart, but it is only a hypothesis until we measure cross-client prototype compatibility directly.

### ACTIVE — Block H01-C: Strength-match GPC and isolate the missing-class denominator

Do **not** add FedGH, FedRE, relational modules, ETF anchors, trainable prototypes, model heterogeneity, or benchmark-scale runs yet. Spend this block answering two narrow questions:

1. Does GPC still help when its knowledge-gradient strength is matched to FedProto?
2. At matched strength, do **locally missing prototypes in the denominator** contribute anything beyond an otherwise identical cosine classifier over locally seen classes?

#### Frozen configuration

Keep the H01-B data/model/split/optimizer/rounds/scale exactly unchanged. Reuse seeds 0/1/2 and the same deterministic batch order. Keep FedProto at `lambda=1`.

Add two minimal GPC variants only:

```text
gpc_all_match:  cosine CE over ALL valid global classes, lambda = 0.002, scale = 10
gpc_seen_match: same cosine CE implementation and lambda = 0.002, but denominator restricted to the current client's local class_set
```

`lambda=0.002` is predeclared from H01-B gradient diagnostics: it places the round-2 GPC knowledge gradient near the observed FedProto order of magnitude. Do not sweep/tune it based on accuracy. Preserve the old `gpc lambda=1` H01-B result as an over-strong reference; it does not need to be rerun unless the implementation seam makes that necessary.

Add unit tests proving that, for a locally missing class prototype, changing that prototype changes `gpc_all_match` loss/feature gradient but has **exactly no effect** on `gpc_seen_match`; both must remain detached from prototype gradients.

#### Add one latent-coordinate diagnostic

Before global aggregation each round, each class is owned by two clients in the current K=2 construction. Log, for every class, cosine similarity between the two owners' local class prototypes, then report mean/min/max across classes per round. This is diagnostic only; do not change training. We want to know whether cross-client same-class coordinates lose compatibility as missing-class recognition collapses.

#### Required run and report

Run the three frozen seeds for:

```text
fedproto(lambda=1)
gpc_all_match(lambda=0.002)
gpc_seen_match(lambda=0.002)
```

Rerun FedProto if needed for paired logging; otherwise reuse only if the new diagnostics can be obtained without compromising pairing. Report round 2 and round 10 for the common cosine readout (seen/missing/all/macro), plus head/L2 in the artifact table. For round 2 client 0 first batch, report local CE gradient, scaled knowledge gradient, and ratio for all three arms.

Acceptance check for the intended strength match: the scaled GPC gradient should be within roughly 0.5×–2× the FedProto knowledge-gradient norm on the same batch. If it is far outside this range, **report the mismatch and stop** rather than tuning lambda repeatedly in this block.

Also report the cross-client same-class prototype cosine trajectory for rounds 1, 2, 5, 10. Preserve all raw receipts.

### Decision rule

- If `gpc_all_match > gpc_seen_match` on missing/all behavior at comparable gradient strength, absent-class competition has a real signal and we can refine GPC.
- If the two matched GPC variants are essentially the same, then the missing-class denominator is not the source of useful transfer; the simple GPC thesis is rejected and we should move to the shared-decision-space / FedGH question next.
- If cross-client same-class prototype cosine degrades sharply in parallel with missing accuracy, that is direct evidence that coordinate-space drift is a central failure mode and will motivate the next method around shared decision geometry rather than stronger prototype repulsion.
- Do not interpret seen-only gains as missing-class transfer.

Append `CODEX REPORT H01-C` with exact commands, source SHA, tests, paired/fairness checks, result table, gradient-strength check, prototype-compatibility trajectory, failures, and a concise recommendation. Do not independently start H02.

---

## CHATGPT REVIEW 04 — No new Codex evidence; H01-C remains active

Checked `main` after review commit `a8cb39a`: there are no newer commits, no new Issue #1 discussion, no appended `CODEX REPORT H01-C`, and no new committed experiment artifacts/results. Therefore there is no scientific evidence to revise the current hypothesis or task.

### ACTIVE — Continue H01-C unchanged

Execute **Block H01-C** exactly as specified above. The next required evidence is the strength-matched `fedproto(lambda=1)` vs `gpc_all_match(lambda=0.002)` vs `gpc_seen_match(lambda=0.002)` comparison, together with the round-2 gradient-strength acceptance check and rounds 1/2/5/10 cross-client same-class prototype cosine trajectory. If the intended gradient match falls outside the predeclared 0.5×–2× window, stop and report the mismatch rather than tuning. Do not start H02 or add FedGH/FedRE/relational modules before `CODEX REPORT H01-C` is committed.

---

## CHATGPT REVIEW 05 — No new Codex evidence; H01-C remains active

Checked `main` after review commit `0d477288`: there are no newer commits or code changes and no appended `CODEX REPORT H01-C` or committed H01-C artifacts. There is therefore no new evidence to justify changing the scientific direction.

### ACTIVE — Continue H01-C unchanged

Execute **Block H01-C** exactly as specified above. Prioritize the predeclared strength-match acceptance check first: `fedproto(lambda=1)` versus `gpc_all_match(lambda=0.002)` and `gpc_seen_match(lambda=0.002)` on the identical paired CIFAR-10 split, with the round-2 scaled knowledge-gradient norm required to fall within the existing 0.5×–2× window relative to FedProto. If that gate passes, complete the three seeds and report round-2/round-10 common-cosine seen/missing/all/macro plus the rounds 1/2/5/10 cross-client same-class prototype cosine trajectory. If the gate fails, stop and report the mismatch without tuning lambda. Do not start H02 or add FedGH/FedRE/relational modules before `CODEX REPORT H01-C` is committed.
