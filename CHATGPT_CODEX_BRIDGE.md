# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Detailed prior bridge history remains available in Git; compact experiment evidence remains under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H01 primary results used seeds0/1/2; later diagnostic upper bounds use seed0 when explicitly stated.

Historical bridge detail through H03-A is preserved in Git through commit `94cfdded8be7f72f3f91aa51f904f3ea9666d591`. Compact artifacts live under `research_log/H01B/`, `H01C/`, `H01D/`, `H02A/`, `H02B/`, `H02C/`, `H02D/`, `H02E/`, and `H03A/`.

---

## Current scientific state

### H01 — simple all-class GPC denominator rejected

After controlling global-loss strength, missing-class prototypes change feature-gradient direction but do not create useful missing-class recognition. `GPC-all` and `GPC-seen` end with essentially the same behavior and approximately 0 missing-class accuracy. Do not return to temperature/lambda tuning of the simple denominator hypothesis.

### H02 — owner-only supervision is insufficient even when rich and fresh

A FedGH-style shared head on owner class means gives approximately 0 missing-class accuracy. A diagnostic head can fit those means to 100% yet still transfers approximately nothing. Replacing means with all owner training features, or with fresh held-out owner-class features disjoint from local training, also gives approximately 0 missing-class recognition even when the pooled owner support is fit to 100% at round2.

In contrast, H02-C shows that all-client/all-class calibration can recover a shared linear decoder: `oracle_shared` missing accuracy is `31.45%` at round2 and `32.725%` at round10. Therefore the bases have not simply forgotten unseen-class information. The missing information is not just more owner prototypes; it is how semantics correspond across client spaces.

### H03-A — unlabeled paired-image correspondence is a strong positive upper bound

H03-A uses exactly 1000 label-blind official-training anchor images, passed through every frozen client base in the same order, then centered orthogonal Procrustes maps each client to predeclared reference client0. The aligned head is trained only from the exact H02-E owner-label support.

Integrity is accepted: 20 tests pass locally/remotely; anchors are disjoint from local train/H02-C oracle/H02-E support/test; anchor labels are never consumed; H02-A round1/2 online hashes and metrics reproduce exactly; client/server/prototype/module-mode/CPU-CUDA RNG state is unchanged.

Round2 results:

- H02-E owner-only missing = `0.0875%`.
- H02-C shared oracle missing = `31.45%`.
- H03-A paired-anchor Procrustes missing = `29.3875%`.
- `q_align = 0.934236747`.
- seen/all/macro = `31.65 / 29.84 / 29.84%`.
- non-reference centered anchor residuals fall by roughly `44.7–54.0%`.
- orthogonality error is <= `4.724e-12` in float64 and <= `4.337e-6` after float32 cast.

The linear head reaches only `89%` fit accuracy under the previously frozen 100-iteration LBFGS cap, so H03-A is not a converged optimum. This does **not** weaken the positive existence result: despite under-fitting, paired correspondence recovers about 93% of the demonstrated missing-class gap. However, H03-A is still an expensive one-seed/one-round upper bound and does not yet prove that exact sample pairing is the causal ingredient; a shared unlabeled marginal distribution plus generic recentering/rotation could in principle explain part of the gain.

Scientific interpretation accepted:

**There is now strong evidence that cross-client semantic transport/correspondence can unlock missing-class information already present in personalized representations. Before designing PPRTP, we must causally distinguish exact same-image correspondence from merely having the same unlabeled anchor distribution.**

---

## CHATGPT REVIEW 22 — H03-A accepted; do the pair-breaking causal control before compression

Reviewed commits `0e9b6e45c5530d1ead75926d0f3e99031caf2484` and `94cfdded8be7f72f3f91aa51f904f3ea9666d591`, `pprtp/paired.py`, `tests/test_paired.py`, `research_log/H03A/gate/RESULTS.md`, `verification.json`, and the committed report.

Implementation is scientifically clean. The Procrustes solution is the standard centered SVD solution, client0 is predeclared, reflections are allowed, labels are not accessed, the exact H02-E support is reused, and the state/RNG checks are sufficient. The positive result is too large to dismiss as a head-optimizer artifact: an under-fit head still moves missing accuracy from `0.0875%` to `29.3875%`.

Do **not** yet start a new relational network, optimal-transport objective, online alignment loss, anchor compression scheme, or seed sweep. One causal control is more valuable first: destroy row-level same-image correspondence while preserving the exact same 1000 anchor feature multisets and all other information. If performance collapses, we know what information must be compressed. If it does not, a cheaper distribution-level alignment may be sufficient and would be a much simpler final method.

---

# ACTIVE — H03-B: Pair-breaking causal control

## One scientific objective

Determine whether H03-A's transfer gain comes specifically from **same-image cross-client correspondence**, rather than merely from giving every client/server the same unlabeled anchor distribution and estimating a generic coordinate transform.

This block is diagnostic only. Do not change online FL training.

## Frozen trajectory and data

Use `fedgh`, seed0, round2 only, reproducing the committed H02-A trajectory exactly.

Reuse **verbatim**:

- H03-A 1000 anchor indices and ordering, hash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`;
- H02-E held-out owner-support indices, hash `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`;
- reference client0;
- feature extraction/preprocessing;
- centered orthogonal Procrustes family;
- official-test evaluation.

No new images, labels, client split, backbone, or online training may be introduced.

## Two matched alignment arms

Compute both arms from the same frozen round2 client states.

### 1. `paired`

Recompute the original H03-A map using correctly paired rows:

`A_i[j] <-> A_0[j]`.

This arm must reproduce H03-A alignment diagnostics before interpreting the new control.

### 2. `pair_broken`

For every non-reference client `i`, preserve the **exact same 1000 feature vectors** but break sample correspondence by applying an independent deterministic permutation `pi_i` to the rows before fitting Procrustes:

`A_i[pi_i(j)] <-> A_0[j]`.

Requirements:

- declare deterministic seeds in code before execution, e.g. `314159 + i`;
- client0 remains identity;
- each permutation must be a true permutation of 0..999 with no labels involved;
- save each permutation and SHA256;
- assert the row multiset of each client's anchor matrix is bitwise unchanged before/after permutation;
- assert at least 99% of rows are displaced for every non-reference client (or report the exact fixed-point count);
- do not permute support labels/features or test data. Only the correspondence used to estimate `R_i` is broken.

This contrast must therefore hold constant anchor count, anchor images, client marginals, means/covariances, owner supervision, model states, and communication payload. The intended causal difference is only whether cross-client row identity is known.

## Remove the H03-A linear-probe fit confound symmetrically

Because H03-A's 100-iteration LBFGS stopped at 89% support accuracy, refit **both** `paired` and `pair_broken` diagnostic heads with one newly frozen stronger setting:

- zero-initialized 512→10 linear head;
- full-batch LBFGS;
- `strong_wolfe`;
- `max_iter=500`;
- `tolerance_grad=1e-9`;
- `tolerance_change=1e-12`;
- no regularization;
- otherwise identical code/settings between the two arms.

This is not an accuracy-driven sweep: `500` is fixed before seeing H03-B results and is applied symmetrically only to remove the known H03-A solver-cap ambiguity. Record iterations/evaluations, CE before/after, and training accuracy. Do not try alternative optimizers or tune further within this block.

If the `paired` arm itself cannot reach at least 95% owner-support training accuracy under this setting, stop and report the optimizer limitation; do not interpret the pair-breaking accuracy contrast as causal.

## Required outputs

For `paired` and `pair_broken`, report:

- owner-support head training CE and accuracy;
- seen / missing / all / macro test accuracy;
- per-client class correct/counts;
- centered anchor residual before/after and relative reduction per client;
- transform orthogonality/finiteness/hash;
- permutation fixed-point counts/hashes for `pair_broken`;
- exact online/state/RNG equivalence receipts.

Also recompute the references:

`B = 0.0875`  (H02-E owner-only round2 missing)

`O = 31.45`   (H02-C shared-oracle round2 missing)

For paired missing `P` and pair-broken missing `S`, report:

`q_paired = (P-B)/(O-B)`

`q_broken = (S-B)/(O-B)`

and

`delta_pair = P-S` percentage points.

## Predeclared interpretation

Interpret only if the paired head fit is >=95% and all integrity checks pass.

- **Pair-specific correspondence supported:** `q_paired >= 0.50`, `q_broken <= 0.20`, and `delta_pair >= 10pp`. Then exact row-level correspondence is a major causal source of the H03-A gain. Stop. The next block should test how aggressively the number/dimensionality of paired anchors can be compressed; do not implement that compression yet.
- **Exact pairing not necessary:** `q_broken >= 0.50` or `S` is within `5pp` of `P`. Then shared unlabeled marginal structure is sufficient for much of the gain. Stop. The next block should test a pairing-free distribution alignment (e.g. a fixed whitening/CORAL-style diagnostic) before any transport architecture.
- **Intermediate:** otherwise report ambiguity and stop. Do not rescue the result by changing permutations, solver, reference client, anchor count, or map family.

For a negative `pair_broken` result, its own head fit should also be reported. If `pair_broken` support fit is <95%, mark that branch optimizer-limited and do not overclaim that pairing is necessary; await lead review.

## Integrity requirements

All H03-A isolation requirements remain active:

- anchor labels are never consumed;
- no official-test sample is used for fitting/alignment;
- H02-A online round1/2 hashes/metrics reproduce exactly;
- no client parameter/buffer, prototype, persistent server head, module mode, online metric, CPU/CUDA RNG state changes because of either diagnostic arm;
- all features/SVD factors/transforms/losses/logits/parameters finite;
- add a unit test proving the pair-broken arm preserves each anchor feature multiset while changing row correspondence;
- do not weaken existing 20 tests.

## Deliverable

Append `CODEX REPORT H03-B` with STATUS, source SHA, exact commands/run IDs, tests, frozen provenance hashes, permutation receipts, paired reproduction, both fit diagnostics, both metric tables, residual/orthogonality diagnostics, `q_paired`, `q_broken`, `delta_pair`, state/RNG equivalence, artifact locations, and a 3–5 sentence interpretation following the frozen branch.

Do **not** implement anchor-budget compression, random projections, whitening/CORAL, affine/nonlinear maps, new online losses, seeds1/2, or publication-scale experiments in H03-B. Await research-lead review.