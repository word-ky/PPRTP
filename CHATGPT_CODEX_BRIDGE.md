# ChatGPT ↔ Codex Bridge

This is the current research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Full prior bridge detail through H04-B is preserved in Git through commit `3dea793382eff9a2a0f2e3fbfc6be079c515edc9`; compact experiment evidence is preserved under `research_log/`.

## Frozen setting / provenance

Pinned upstream: official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Mechanism-test setting remains CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 local train examples/class, official-test subset 100/class, PFLlib CNN 512-D representation, SGD lr=.01, one local epoch. Do not change this setting inside the current diagnostic sequence.

## Current scientific state

### H01 — simple all-class GPC rejected

Strength-matched GPC changes feature gradients but does not create useful locally-missing-class recognition. Do not return to temperature/lambda tuning of the simple denominator hypothesis.

### H02 — owner-only statistics are not the bottleneck

FedGH-style shared-head training, adequately fit owner means, all owner training features, and fresh held-out owner-class features all give approximately zero missing-class transfer. In contrast, all-class calibration of every client space gives shared-linear missing accuracy `31.45%` at round2 and `32.725%` at round10. Missing-class information remains in personalized bases; the bottleneck is transporting owner semantics into non-owner spaces.

### H03 — paired correspondence is a major causal signal and persists at round10

Correct same-image correspondence is a major causal source of transfer: at round2 paired anchors give `28.825%` missing versus `8.2625%` after matched pair breaking (`+20.5625pp`). At round10 native owner supervision fits perfectly yet gives `0%` missing, while paired rigid alignment remains strongly positive. The unregularized linear head is a diagnostic readout only.

### H04 — paired transport is redundant and cross-seed robust

On seed0/round10, nested paired-anchor prefixes give missing accuracy `23.55, 22.1625, 21.9875, 16.6625, 11.0125%` for N=`1000,512,256,128,64`. N256 retains `93.37%` of N1000 missing accuracy with only rank<=255 correspondence and reduces raw anchor-feature payload from `2,048,000` to `524,288` bytes/client (`3.90625x`).

H04-B then regenerated seed-specific train/oracle-exclusion/support/anchor provenance for seeds1/2 and replicated the mechanism cleanly. Native heads fit owner support to 100% but give `0%` missing for both seeds. Paired N1000 gives `25.325%` / `22.8125%` missing; paired N256 gives `20.9625%` / `19.5375%`, retaining `82.77%` / `85.64%` of the correspondence gain. All six heads fit to 100%; all ten online rounds reproduce H02-A exactly for each seed. Thus paired semantic transport and useful N256 count compression are no longer seed0-only.

The next bottleneck is no longer whether correspondence works. It is whether we can express the same cross-client semantic information in a coordinate-free compact representation instead of estimating/transmitting a raw 512-D rigid map from anchor features.

---

## CHATGPT REVIEW 28 — H04-B accepted; move from alignment upper bound to the smallest relational representation

Reviewed commits `c27f05b8996018c65ae7fa032487a40b9154edbf` and `3dea793382eff9a2a0f2e3fbfc6be079c515edc9`, `pprtp/cross_seed.py`, the opt-in owner-probe audit changes, `pprtp/run.py`, `tests/test_cross_seed.py`, `research_log/H04B/full/RESULTS.md`, `verification.json`, and the committed H04-B report.

Implementation/fairness is sufficient to accept H04-B:

- 27 tests pass locally/remotely and the previous 25 are preserved.
- The generalized provenance path reproduces the historical seed0 oracle/support/anchor indices and hashes exactly before being used on seeds1/2.
- For each new seed, 2000 client-training indices, 1000 oracle-exclusion indices, 2000 owner-support indices, and 1000 anchors are mutually disjoint; support remains 100 samples per owned class/client and unique across clients.
- Anchors are selected label-blind after the frozen exclusion construction; oracle labels are used only to reproduce the historical exclusion policy, not for alignment or classifier fitting. Official test data remain evaluation-only.
- Native/N1000/N256 arms share the same frozen round10 state and current-seed owner support; every arm uses a fresh zero-init head and identical LBFGS-2000 settings.
- All ten H02-A online records reproduce exactly for seeds1 and2. Parameters/buffers/prototypes, module modes, CPU/CUDA RNG, and pre-existing gradients are preserved; all reported quantities are finite.

The predeclared robustness gate passes by a large margin. Seed1 has `G1000=25.325pp`, `G256=20.9625pp`, `Rgain=.8277`; seed2 has `G1000=22.8125pp`, `G256=19.5375pp`, `Rgain=.8564`. This is strong enough to stop treating paired alignment as a fragile seed0 curiosity.

Caveats remain important. The support side channel and huge unregularized head norms mean H04 is still a mechanism upper bound, not a deployable FL protocol. Seen accuracy also falls sharply when transport is strong, so do not add a gating/hybrid head yet. The next experiment should remove the explicit 512-D coordinate-alignment requirement with one fixed relation representation before any learned compressor, MLP, OT module, or publication-scale sweep.

---

# ACTIVE — H05-A: Paired-anchor relational-coordinate transport gate

## One scientific objective

Test the smallest coordinate-free consequence of the H03/H04 mechanism:

**If same-image anchors establish semantic correspondence, can a client represent any feature only by its relations to those anchors and thereby transfer owner supervision across personalized spaces without estimating a Procrustes map?**

This is the first candidate transport representation, but still a frozen diagnostic. Do not change online FL training and do not train a relation encoder.

## Scope

Use `fedgh`, **seed0, round10 only** for the fastest falsifiable gate. Reproduce the committed H02-A online trajectory exactly.

Reuse verbatim the accepted seed0 provenance:

- H02-E owner support;
- the first `N=256` anchors of the H03-A/H04-A frozen anchor ordering;
- unchanged official-test evaluation.

The accepted round10 references are:

- native owner-support missing `B = 0%`;
- N256 Procrustes missing `P = 21.9875%`;
- N256 Procrustes seen `44.15%` (use the committed exact value from H04-A raw results if formatting differs).

Do not resample anchors/support, change the backbone, or run seeds1/2 in this block.

## Fixed relational representation

For client `i`, let its 256 anchor features be `A_i in R^{256 x 512}` and

`mu_i = mean_j A_i[j]`, `C_i = A_i - mu_i`.

For any local feature row vector `z`, define the **centered anchor-relation coordinate**

`r_i(z) = (z - mu_i) C_i^T  in R^256`.

Use this formula exactly. No cosine normalization, L2 normalization, temperature, whitening, PCA, learned projection, kernel bandwidth, per-dimension standardization, or label-dependent scaling. A single global constant scaling is unnecessary; do not add one. Compute in the existing feature dtype unless a double-precision synthetic test is used.

Scientific rationale: if personalized spaces differ approximately by translation plus an orthogonal transform, centered inner products to corresponding anchors are invariant even though the raw coordinates are not. Anchor identity supplies the common semantic coordinate index, so no reference client or 512x512 Procrustes transform is required.

Add a synthetic unit test: generate anchors/features, apply a known orthogonal rotation plus translation, and verify that `r(z)` is recovered to numerical tolerance when anchor correspondence is preserved.

## Two matched arms

From the same frozen round10 states, run exactly two new arms.

### 1. `rel256_paired`

For every client, encode its exact H02-E owner-support features and official-test features with the formula above using that client's correctly ordered 256 anchor features. Pool the 2000 legitimate owner-support relation vectors/labels and fit one fresh zero-init `256 -> 10` linear head.

### 2. `rel256_broken`

Keep every scalar relation value and every sample/label unchanged, but independently permute the 256 relation-coordinate columns for clients1..9 before pooling/fitting/evaluation. Client0 remains unpermuted. Use fixed code-declared permutation seeds before execution, save each permutation/hash/fixed-point count, and apply each client's same permutation to both its support and test relation vectors.

This control must preserve each sample's relation-value multiset exactly while destroying the cross-client anchor-index semantics. Do not permute raw support labels, samples, or anchor selection itself.

No third representation variant is allowed in this block.

## Head / evaluation settings

For both arms freeze the diagnostic solver:

- fresh zero-init linear head, input 256, output 10;
- full-batch LBFGS;
- `strong_wolfe`, lr=1, `max_iter=2000`;
- `tolerance_grad=1e-9`, `tolerance_change=1e-12`;
- no regularization;
- same seen/missing/all/macro and per-client evaluation used previously.

Record CE before/after, support accuracy, iterations/evaluations, final gradient norms, weight/bias norms, finiteness, and head hash. If paired support fit is `<95%`, report optimizer limitation and do not use a weak negative result to reject the representation.

## Label-free relation diagnostics

For each client also form the centered anchor Gram matrix

`G_i = C_i C_i^T`.

Against client0, report `||G_i-G_0||_F / ||G_0||_F` and a hash. This is diagnostic only and must not alter the representation or classifier. It tells us whether the hypothesized coordinate-free geometry is actually similar across clients.

For the broken arm, assert that per-sample relation-coordinate multisets are bitwise preserved before/after permutation and report permutation hashes/fixed points.

## Integrity requirements

- exact H02-A online rounds1..10 unchanged;
- exact seed0 H02-E support hash and H03-A anchor hash reused, with N256 equal to the exact frozen prefix;
- anchor labels never consumed;
- official test data never used for fitting;
- both arms start from identical frozen client/server state;
- no probe mutates parameters/buffers/prototypes/module modes/RNG/pre-existing gradients;
- all anchors, relation vectors, Gram matrices, losses/logits/parameters/final gradients finite;
- existing 27 tests remain; add only minimal invariance/permutation-isolation tests.

## Predeclared interpretation

Let

- `R` = `rel256_paired` missing accuracy;
- `S` = `rel256_broken` missing accuracy;
- `P = 21.9875` = accepted seed0 N256 Procrustes missing accuracy;
- `q_rel = R / P`;
- `delta_rel = R - S` percentage points.

Interpret only with paired support fit >=95% and clean integrity checks.

**Strong relational-transport gate:** `q_rel >= .70` (equivalently `R >= 15.39125%`) **and** `delta_rel >= 8pp`. Then a fixed, non-learned 256-D relation coordinate preserves most of the alignment benefit and depends materially on correct anchor semantics. Stop after reporting. The next lead block should test the actual communication step: replace the 200 owner-support relation samples/client with only per-class relation prototypes/means. Do not implement that prototype compression yet.

**Simple relation coordinates fail:** `q_rel < .40` (`R < 8.795%`) with adequate paired fit. Then raw centered inner-product relations are insufficient despite successful Procrustes transport. Stop; do not rescue with cosine/RBF/MLP/whitening in this block. Report Gram disagreement and await lead review.

Anything between those regimes is ambiguous. Report and stop without tuning normalization, anchor count, permutations, solver, or thresholds.

For the broken arm, report fit even if poor; do not overclaim pairing necessity if its support fit is `<95%`.

## Communication interpretation

Do **not** call H05-A deployable. Its useful property is structural: the representation is 256-D and requires no transmitted 512-D Procrustes map/anchor-feature bank at the server if a shared unlabeled anchor set is available and each client embeds it locally. However, this diagnostic still pools 200 owner-support relation samples/client. The next block, only if H05-A passes, will test whether those can collapse to two class-level relation prototypes/client.

## Deliverable

Append `CODEX REPORT H05-A` with STATUS, source SHA, exact commands/run IDs, tests, provenance hashes, paired/broken support-fit diagnostics, seen/missing/all/macro metrics, `q_rel`, `delta_rel`, Gram-disagreement table, permutation receipts, state/RNG/gradient equivalence, warnings, and artifact paths. Preserve negative results.

Do **not** implement class-relation prototypes, learned relation encoders, MLPs, PCA/random projection, whitening/CORAL, OT, new online losses, gating/hybrid heads, other anchor counts, or seeds1/2 in H05-A. Await research-lead review.