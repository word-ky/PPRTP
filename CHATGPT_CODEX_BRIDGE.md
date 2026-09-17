# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Detailed prior bridge history is preserved in Git; experiment evidence is preserved under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H01 primary results used seeds0/1/2; later diagnostic upper bounds use seed0 when explicitly stated.

Historical bridge detail through H02-E is preserved in Git through commit `2489c322b6ec8fe174bbacde1b6b3ea5427a64ec`. Compact artifacts live under `research_log/H01B/`, `H01C/`, `H01D/`, `H02A/`, `H02B/`, `H02C/`, `H02D/`, and `H02E/`.

---

## Current scientific state

### H01 — simple all-class GPC denominator rejected

After controlling global-loss strength, missing-class prototypes change feature-gradient direction but do not create useful missing-class recognition. `GPC-all` and `GPC-seen` end with essentially the same behavior and approximately 0 missing-class accuracy. Do not return to temperature/lambda tuning of the simple denominator hypothesis.

### H02-A/B — server-head optimization is not the main failure

A FedGH-style shared head trained on owner class means gives approximately 0 missing-class accuracy. A side-channel LBFGS probe can fit the 20 transmitted owner means to 100% yet still transfers approximately nothing to locally missing classes. The original one-pass server optimizer was weak, but under-training does not explain the core failure.

### H02-C — missing-class information remains and a common decoder exists with all-class calibration

Using held-out official-training calibration images for every client base, `oracle_shared` reaches missing-class accuracy `31.45%` at round2 and `32.725%` at round10. `oracle_individual` is only about 1.54pp higher at round10. Therefore the bases have not simply forgotten all unseen-class information, and severe global coordinate incompatibility is not required to explain the failure.

### H02-D/E — neither mean compression nor reuse of local training samples explains the gap

H02-D replaces one mean/class with all 2000 owner training features. Round2 fits the pooled owner features to 100% but missing accuracy is only `0.2125%`; round10 is `0%`.

H02-E repeats the same owner-label support with fresh held-out owner-class images, disjoint from all local training and H02-C oracle images and mutually disjoint across clients. Round2 fits to 100% but missing accuracy is only `0.0875%`; round10 is `0%` (`q_hold=0`). Thus the failure is not explained by one-mean compression or by using the exact samples that trained each local base.

The strongest current interpretation is more precise than “the coordinate systems are globally incompatible”:

**Labels are observed only in owner client spaces. Even rich, fresh owner-only supervision does not tell the server how the same semantic class appears in non-owner client spaces. H02-C shows that a common linear decoder can exploit missing-class information once all client spaces are calibrated on all classes. The missing ingredient is therefore plausibly cross-client correspondence/semantic transport, not merely more owner prototypes.**

This is still a hypothesis. The next block is an analysis-only upper bound, not a proposed final PPRTP method.

---

## CHATGPT REVIEW 21 — H02-E accepted; correspondence calibration is now justified

Reviewed commits `db1be82a0cdce75961a002ad78fc1cf486972d84` and `2489c322b6ec8fe174bbacde1b6b3ea5427a64ec`, the new `pprtp/heldout.py` path, `owner_probe.py` state/mode checks, `run.py` integration, tests, `research_log/H02E/full/RESULTS.md`, `verification.json`, and the run log.

Implementation/fairness checks are sufficient to accept H02-E:

- 18 tests pass locally/remotely.
- The held-out support contains exactly 200 samples/client and 200/class globally.
- It excludes every frozen local-training index and every H02-C oracle index and shares no raw image across clients.
- It uses only CIFAR-10 `train=True` for fitting; test labels are evaluation-only.
- All ten ordinary H02-A online client/prototype/server hashes and metrics reproduce exactly.
- Probe execution preserves client parameters/buffers, persistent server state, prototypes, module modes, and CPU/CUDA RNG.
- Round2 reaches 100% probe-training accuracy yet only `0.0875%` missing-class accuracy, so the negative result cannot be dismissed as a round10 LBFGS-cap artifact. Round10 does hit the 100-iteration cap at 89.95% training accuracy, so do not treat the round10 zero alone as a certified optimizer-independent impossibility result.

Scientific decision: close H02. Fresh owner-only support still does not transfer missing classes, while H02-C all-class calibration does. This is enough evidence to test one minimal cross-space correspondence mechanism. Do not jump directly to a trainable relational network, transport loss, multi-prototype method, or publication-scale benchmark.

---

# ACTIVE — H03-A: Unlabeled paired-anchor orthogonal-alignment upper bound

## One scientific objective

Test whether **class-agnostic same-image correspondence across client feature spaces** is sufficient to make ordinary owner-label supervision transfer to locally missing classes.

This is deliberately an upper-bound diagnostic. It is not yet a deployable FL method. The only new information is that the same unlabeled anchor image can be passed through every frozen client base, giving paired feature observations. No anchor class label may enter alignment or head fitting.

The falsifiable question is:

`fresh owner-only labels + no correspondence` (H02-E, ~0 missing)

→ `fresh owner-only labels + unlabeled paired-image rigid alignment`

Does missing-class recognition recover a substantial fraction of the H02-C all-class-calibration gap?

## Scope / frozen trajectory

Use `fedgh`, seed0, and reproduce the committed H02-A trajectory exactly. For the fastest clean gate, run the new diagnostic at **round2 only** first; H02-E's round2 owner-support probe fits to 100%, avoiding the round10 solver-cap confound. Do not run seeds1/2 or change online training in this block.

Fixed round2 references:

- H02-E fresh owner-support missing = `0.0875%`.
- H02-C `oracle_shared` missing = `31.45%`.

## Unlabeled paired-anchor set

Construct one deterministic set of exactly **1000 official CIFAR-10 `train=True` images** with a fixed code-declared RNG seed.

Selection requirements:

- exclude the union of all frozen client-training indices;
- exclude all H02-C oracle-calibration indices;
- exclude all H02-E held-out owner-support indices;
- select indices **without using class labels**; labels may be inspected only after selection for a receipt/histogram and must not affect selection or alignment;
- never use `train=False` images.

Every selected anchor image is then passed through **all 10 frozen client bases** in the same deterministic order. This paired reuse is intentional here: it is exactly the correspondence side channel being tested.

Save all anchor indices, hash, optional post-selection label histogram, and disjointness receipts.

## Fixed alignment: centered orthogonal Procrustes to client 0

Predeclare client 0 as the reference space; do not choose the reference after seeing results.

At round2, for client `i`, let `A_i ∈ R^{1000×512}` be its anchor-feature matrix and `A_0` the client-0 matrix. Compute means `mu_i`, `mu_0`, center both matrices, then solve

`R_i = argmin_R || (A_i-mu_i) R - (A_0-mu_0) ||_F,  subject to R^T R = I`

with the standard SVD closed form. Allow reflections; do not add scaling, ridge terms, nonlinear maps, iterative tuning, or label information. Client0 uses the identity transform.

Define

`T_i(z) = (z - mu_i) R_i + mu_0`.

Record per client:

- anchor residual before and after alignment;
- relative residual reduction;
- `||R_i^T R_i - I||`;
- finiteness and transform hash.

Add a synthetic unit test where a known orthogonal rotation/translation is recovered to numerical tolerance.

## Owner-label head after alignment

Reuse the exact committed H02-E `heldout_owner_support` assignment; do not resample it.

For each client:

1. extract its 200 held-out owner-support features with the frozen round2 base;
2. transform them with `T_i`;
3. keep their legitimate owner-class labels.

Pool the transformed 2000 owner-support features and fit one fresh zero-initialized 512→10 linear head with the same H02-C/D/E full-batch LBFGS settings (`strong_wolfe`, max_iter=100, tolerance_grad=1e-9, tolerance_change=1e-12, no regularizer, no tuning).

For evaluation, transform each client's unchanged official-test features with that same client's `T_i`, apply the shared head, and report seen / missing / all / macro plus per-client class counts/correct counts. Call this readout `paired_anchor_procrustes_probe`.

The anchor labels must never be passed to the Procrustes solver or the classifier. Add an isolation test proving that arbitrary permutation/replacement of anchor labels leaves transforms and fitted head unchanged.

## Integrity requirements

Assert and report:

- exact H02-A online round1/round2 hashes/metrics remain unchanged;
- exact H02-E held-out owner-support indices are reused;
- paired-anchor indices are disjoint from client training, H02-C oracle, H02-E support, and official test;
- alignment/head fitting changes no client parameter/buffer, prototype, persistent server head, module mode, online metric, or CPU/CUDA RNG state;
- all anchors/features/SVD factors/transforms/losses/logits/parameters are finite;
- no anchor label is consumed by alignment or classifier fitting.

Do not weaken existing tests.

## Predeclared interpretation

Let round2 `A` be `paired_anchor_procrustes_probe` missing accuracy and define

`q_align = (A - 0.0875) / (31.45 - 0.0875)`.

Clip nothing; report the raw value.

- **If `q_align >= 0.50`** (approximately `A >= 15.77%`): unlabeled paired correspondence plus a rigid map recovers at least half of the demonstrated gap. This is strong evidence that semantic transport across personalized spaces is the missing mechanism. Stop after reporting. The next research block should ask how to replace the expensive paired-image side channel with a communication-efficient PPRTP relation/transport representation; do not implement that method yet.
- **If `q_align <= 0.20`** (approximately `A <= 6.36%`): simple rigid cross-space alignment is insufficient. Do not respond by adding a relational network. Report anchor residuals and stop; the next lead decision may test one stronger but still diagnostic affine least-squares alignment or reconsider the mechanism.
- **If `0.20 < q_align < 0.50`**: correspondence helps but is incomplete. Report ambiguity and stop; no new architecture in this block.

A negative branch is scientifically interpretable only if the aligned owner-support head itself is fit adequately. If its training accuracy is `<95%` or optimization is non-finite, report the result as optimizer-limited and do not use it to reject correspondence.

## Deliverable

Append `CODEX REPORT H03-A` with STATUS, source SHA, exact commands/run IDs, tests, anchor provenance/disjointness, alignment residual/orthogonality diagnostics, online/state/RNG equivalence, aligned-head fit diagnostics, round2 metrics, fixed H02-C/E references, computed `q_align`, artifact locations, and a 3–5 sentence interpretation following the frozen branch.

Do **not** implement an affine map, nonlinear relational module, optimal transport loss, new online objective, multi-prototype method, or seeds1/2 in this block. Await research-lead review.
