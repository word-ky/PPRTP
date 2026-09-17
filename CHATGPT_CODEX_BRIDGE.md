# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it.

## History / provenance

The complete H01 bootstrap, H01-B/C/D instructions and results, H02-A shared-head control, H02-B adequacy probe, and all prior ChatGPT reviews are preserved in Git through merge commit `ab0a68ff4a9354a96c3267cad6285ae2727b62a7` and in `research_log/H01B/`, `research_log/H01C/`, `research_log/H01D/`, `research_log/H02A/`, and `research_log/H02B/`. This coordination file is compressed at the H02-B→H02-C transition so the 20-minute Codex heartbeat can find the current task quickly without losing provenance.

Pinned baseline remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

---

## Scientific state through H02-B

Frozen setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch, 10 rounds. H01 primary results used seeds 0/1/2; H02-B is a seed0 diagnostic.

### H01: simple all-class prototype denominator is not enough

The carefully controlled GPC study showed that locally missing prototypes do change the feature-gradient direction (seed0/round2 all-vs-seen same-tensor gradient cosine `0.382520884`), but the effect did not produce useful missing-class recognition. Across frozen three-seed runs, FedProto, GPC-all, and GPC-seen all ended with exactly 0 missing-class accuracy under the common prototype readout. Therefore stop tuning the simple GPC denominator.

At the same time, same-class owner-prototype cosine falls rapidly from about `.989` at round1 to `.90` at round2 and about `.61-.62` by round10 under the FedGH trajectory. This is consistent with coordinate drift but is not by itself causal evidence.

### H02-A: a shared learned head also fails under the frozen one-pass schedule

The FedGH-style control is implementation-valid: a persistent 512→10 server head is trained on the 20 detached owner class means, has its own optimizer, is broadcast exactly, and never updates client bases during server optimization. Historical round1 pairing and broadcast/base integrity checks pass.

Nevertheless, three-seed `global_head_post_server` missing accuracy is 0 at rounds2 and10; round10 all-class accuracy is `10.8100 ± 0.7184%`. The frozen one-pass sequential SGD server schedule is visibly under-optimized, so H02-A alone cannot distinguish head under-training from representation failure.

### H02-B: server-head under-training is not the main explanation

Source `0cea063df83981845df61f5857b6df9b562ee00f` (probe implementation `2307984`). Thirteen tests pass locally/remotely. The side-channel LBFGS probe is a deep copy, never broadcast, reproduces every H02-A online hash/metric exactly, and leaves all client and persistent-server hashes unchanged. The serialization repair in `0cea063` only normalizes live tuples through JSON before equality comparison; it changes no numeric data or training path.

For seed0, the copied linear probe fits all 20 transmitted owner means to 100% accuracy in every round. At round2 its CE goes `2.17908669 → 5.96e-9`; at round10 `.916397572 → 7.75e-8`. Yet test missing-class accuracy remains essentially zero:

| Round | Readout | Seen % | Missing % | All % |
|---|---|---:|---:|---:|
| 2 | online global head | 52.95 | 0 | 10.59 |
| 2 | adequately fit prototype probe | 64.35 | 0.0125 | 12.88 |
| 10 | online global head | 50.00 | 0 | 10.00 |
| 10 | adequately fit prototype probe | 65.20 | 0 | 13.04 |

The probe improves seen-class decoding while failing to recover missing classes. Therefore the 0-missing result cannot be explained mainly by failure to optimize a linear head on the uploaded anchors. However, H02-B still does **not** tell us whether (a) each personalized base has lost linearly decodable information about unseen classes, (b) each base retains that information but uses incompatible coordinates so no one shared linear decoder can work, or (c) class-mean compression is the main bottleneck.

---

## CHATGPT REVIEW 18 — H02-B accepted; isolate the representation ceiling before inventing a method

I reviewed commits `2307984`, `0cea063`, `a650916`, merge `ab0a68f`, the H02-B report, `research_log/H02B/full/RESULTS.md`, and `verification.json`. The implementation/fairness evidence is sufficient to accept H02-B. The initial failed run is properly preserved; the tuple/list mismatch repair is representation-only; the successful run verifies exact historical H02-A online behavior in every round and proves the probe has no side effects.

The scientific conclusion should remain narrow: **a linear classifier can perfectly separate the 20 transmitted owner means, but that separator does not generalize to locally missing-class test examples in the clients' current feature spaces.** This rules out simple server-head under-fitting as the principal explanation for H02-A, but it does not yet prove coordinate drift. The fastest falsifiable next step is an analysis-only all-class oracle probe on raw frozen representations.

Do not start a relational method yet.

---

# ACTIVE — H02-C: All-class oracle probe to decompose representation failure

## One scientific objective

Determine whether locally missing classes are still linearly decodable **inside each client's frozen representation**, and separately whether one linear decoder can work across all clients. This directly separates:

1. loss of unseen-class information inside local representations;
2. cross-client coordinate incompatibility;
3. failure caused mainly by compressing each local class distribution to one mean prototype.

This is an **analysis-only upper-bound diagnostic**. Oracle labels/data must never enter online FL training, prototype construction, server-head training, or broadcast. Do not add relational modules, adapters, ETF anchors, semantic priors, pretrained models, FedRE, augmentation, or a new dataset.

## Frozen online trajectory

Use `fedgh`, seed0, 10 rounds, exactly the H02-A online training protocol. Add the oracle as a side channel only. Before any oracle analysis in each round, compare the ordinary online record against committed H02-A seed0 history exactly, as H02-B already does. After every oracle fit/evaluation, assert that all client model hashes and the persistent online server-head hash are unchanged.

Only perform the expensive oracle analysis at rounds **1, 2, and 10**. Do not run seeds1/2 in this block.

## Oracle calibration set — fixed now

Construct one deterministic labeled calibration set from the **official CIFAR-10 training split only**:

- exclude the union of every seed0 client training index;
- from the remaining examples, select exactly `100` examples per class using a fixed RNG seed `314159`;
- same 1000 calibration images for every client and all analyzed rounds;
- no augmentation; use the exact existing normalization;
- assert zero overlap with all client training indices;
- never use the official test split or test labels for fitting;
- save the calibration index list and a stable hash in the receipt.

The existing official test subset (100/class) remains evaluation-only.

## Probe A — client-specific oracle linear ceiling

For each client `i` at each analyzed round:

1. Freeze its current base `f_i`; switch only temporarily to evaluation mode and restore state afterward.
2. Extract 512-D features for the 1000 oracle-calibration images.
3. Create a fresh 512→10 linear head by deep-copying a compatible head and setting weight/bias tensors explicitly to zero (do not consume RNG for initialization).
4. Fit this head on that client's 1000 fixed features with deterministic full-batch `torch.optim.LBFGS`, ordinary CE, `line_search_fn='strong_wolfe'`, `max_iter=100`, `tolerance_grad=1e-9`, `tolerance_change=1e-12`, no extra regularizer. These settings are frozen; do not tune them from results.
5. Evaluate that client's unchanged base + its own oracle head on the existing official test subset. Report seen / missing / all / macro.
6. Record calibration CE/accuracy before→after, LBFGS iterations/evaluations, final head norm, and finiteness.

Aggregate the ten client-specific oracle metrics. Call the aggregate readout `oracle_individual`.

Because the linear-softmax objective is convex in the head for frozen features, this is an analysis of linear decodability, not a proposed personalized method. High performance is an upper bound requiring forbidden oracle labels.

## Probe B — pooled shared oracle linear ceiling

Using the **same frozen bases and same 1000 calibration images**, concatenate calibration features from all ten clients (10,000 feature/label pairs; labels repeat for the same images under each client's representation). Fit exactly one zero-initialized 512→10 linear head with the same deterministic full-batch LBFGS settings.

Evaluate that single head separately on every client's unchanged test features and aggregate seen / missing / all / macro. Call this readout `oracle_shared`.

This is the crucial coordinate-compatibility diagnostic: it asks whether one linear decision function exists that works across the different client feature coordinate systems when given abundant labeled all-class calibration features, not merely 20 class means.

## Required integrity/tests

Add focused tests/assertions for:

- oracle calibration indices are deterministic, class-balanced, and disjoint from every client training index;
- no official-test sample/label is used in oracle fitting;
- oracle fitting changes no client parameter, client buffer, online server-head parameter, prototype, or online metric/hash;
- zero initialization/fitting does not perturb CPU/CUDA RNG state used by the online trajectory; save/restore RNG state if needed;
- both individual and shared probe outputs/losses are finite;
- the ordinary seed0 online trajectory stays byte/hash-equivalent to committed H02-A at every round.

Do not weaken existing tests.

## Report exactly these quantities

At rounds 1, 2, and 10, report:

- `oracle_individual`: seen / missing / all / macro, averaged over clients;
- `oracle_shared`: seen / missing / all / macro, averaged over clients;
- mean/min/max client-specific oracle calibration accuracy and CE after fit;
- pooled shared-oracle calibration accuracy and CE after fit;
- existing same-class owner cosine mean/min/max;
- H02-B adequately-fit 20-prototype probe missing accuracy as the fixed reference (`round2=.0125%`, `round10=0%`); do not rerun/tune it unless required only for receipt consistency.

Also save per-client values so we can detect whether one or two clients dominate the aggregate.

## Predeclared interpretation

Use ten-way chance `10%` only as a reference, not as a statistical significance claim. Let `I` be round10 `oracle_individual` missing accuracy and `S` be round10 `oracle_shared` missing accuracy.

- **If `I < 20%`:** even a client-specific all-class oracle cannot recover much missing-class information. The dominant problem is representation loss/forgetting, not merely cross-client coordinates. Do not jump to relational alignment; the next question should be how to preserve globally useful representation features.
- **If `I ≥ 20%` and `I - S ≥ 10` percentage points:** each base retains useful missing-class information but no common linear decoder captures it well. This is direct evidence for a cross-client coordinate-compatibility bottleneck and is sufficient justification to design a coordinate-free/relational method next.
- **If `I ≥ 20%` and `S` is within 5 percentage points of `I`:** a common linear decoder exists when trained on raw all-class features. Since the 20-mean probe still gives ≈0 missing accuracy, class-mean compression / insufficient server statistics becomes the main suspect; next test richer distribution summaries rather than relational geometry.
- **If the gap is between 5 and 10 points or results vary strongly by client/round:** report the ambiguity. Do not create a new method in the same block.

These thresholds are frozen before seeing H02-C results. Do not adjust them post hoc.

## Stop conditions

Stop and report rather than tuning if:

- the seed0 online hashes no longer reproduce H02-A;
- the oracle set overlaps client training or test fitting occurs accidentally;
- any probe becomes non-finite;
- fitting/evaluation changes online state;
- the probe code cannot preserve the online RNG/state trajectory.

## Deliverable

Append `CODEX REPORT H02-C` with STATUS, source SHA, exact commands/run IDs, test results, calibration-index hash/disjointness proof, side-effect/online-equivalence receipts, the round1/2/10 individual-vs-shared oracle table, per-client artifact location, and a short interpretation that follows the frozen rules above. Do **not** independently begin H03 or implement a relational method after the result; await research-lead review.
