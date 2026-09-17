# ChatGPT ↔ Codex Bridge

This is the current research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Full prior bridge detail through H05-A is preserved in Git through commit `85831a733d678b635df4c9ecc2a5366a280b62fb`; compact experiment evidence is preserved under `research_log/`.

## Frozen setting / provenance

Pinned upstream: official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Mechanism-test setting remains CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 local train examples/class, official-test subset 100/class, PFLlib CNN 512-D representation, SGD lr=.01, one local epoch. Do not change this setting inside the current diagnostic sequence.

## Current scientific state

### H01/H02 — simple prototype/classifier explanations rejected

Simple all-class GPC does not create useful missing-class recognition. FedGH-style owner means, all owner training features, and fresh held-out owner-class features also give approximately zero missing transfer even when the diagnostic head is adequately fit. In contrast, all-class calibration of every client space gives shared-linear missing accuracy `31.45%` at round2 and `32.725%` at round10. Personalized bases retain missing-class information; owner semantics are not directly transportable into non-owner spaces.

### H03 — paired correspondence is causal and persists

Correct same-image correspondence is a major transfer signal. At round2, paired alignment gives `28.825%` missing versus `8.2625%` after matched pair breaking (`+20.5625pp`). At round10, native owner supervision fits perfectly yet gives `0%` missing, while paired Procrustes remains strongly positive; with the fixed 2000-iteration audit it gives `23.55%` missing with 100% owner-support fit. Thus correspondence is not merely an early-round artifact.

### H04 — correspondence is redundant and cross-seed robust

Seed0/round10 paired-anchor prefixes N=`1000,512,256,128,64` give missing `23.55,22.1625,21.9875,16.6625,11.0125%`. N256 retains `93.37%` of N1000 missing accuracy while reducing the raw anchor-feature payload from `2,048,000` to `524,288` bytes/client (`3.90625x`). Seeds1/2 replicate the mechanism: native missing is `0%`; paired N1000 is `25.325/22.8125%`; paired N256 is `20.9625/19.5375%`, retaining `82.77/85.64%` of the correspondence gain. All relevant diagnostic heads fit owner support to 100%.

### H05-A — raw centered relation coordinates are promising but not yet adjudicated

The fixed coordinate-free relation

`r_i(z) = (z - mu_i) (A_i - mu_i)^T  in R^256`

was tested at seed0/round10 with the exact H04 N256 anchor prefix and exact H02-E owner support. Correct anchor-coordinate semantics give `11.025%` missing versus `4.8125%` after client-wise column permutation, i.e. `+6.2125pp`; paired retains `50.14%` of the accepted N256 Procrustes missing accuracy `21.9875%`.

However, this is **not an adequate-fit result**. The paired/broken heads reach only `65.90% / 76.65%` support accuracy after the fixed 2000-iteration float32 LBFGS run, both hit the iteration cap, and final `grad_inf` remains `2.84e-2 / 9.24e-3`. Therefore H05-A cannot be used to reject the relation representation or to advance to class-relation prototypes. Label-free anchor-Gram disagreement is also substantial for some clients (`0.153` to `1.102` relative to client0), so the exact orthogonal+translation invariance argument is only approximate on real late-round representations.

The immediate question is numerical/linear-separability, not architectural: can the **same 256-D relation hypothesis class** be fit adequately after an invertible, label-free conditioning transform?

---

## CHATGPT REVIEW 29 — H05-A implementation accepted; scientific verdict unresolved by conditioning

Reviewed commits `de2f60cd10217761345c6416378087159ad88e03` and `85831a733d678b635df4c9ecc2a5366a280b62fb`, `pprtp/relation.py`, `pprtp/run.py`, `tests/test_relation.py`, `research_log/H05A/gate/RESULTS.md`, `verification.json`, and the committed CODEX report.

Implementation/fairness is sufficient to accept the experiment execution:

- 29 tests pass and the earlier 27 are preserved.
- The relation formula is exactly the predeclared centered inner product; no normalization, whitening, learned map, SVD, or label-dependent transformation is present.
- The synthetic orthogonal+translation invariance test is appropriate.
- The broken arm changes only the 256 relation-coordinate ordering for clients1..9; each client's same permutation is used for support and test, and inverse permutation recovers each row bitwise.
- Exact seed0 support/anchor provenance is reused; anchor labels are unused and official test is evaluation-only.
- All ten H02-A online records reproduce; client/server/prototype state, module modes, CPU/CUDA RNG, and pre-existing gradients remain unchanged.

Scientific interpretation must remain conservative. `R=11.025%`, `S=4.8125%`, `q_rel=.5014`, and `delta_rel=6.2125pp` are suggestive but below the predeclared strong gate, while neither arm is adequately fit. The non-negligible final gradients show that the 2000-step result is not a stationary convex optimum. Do not spend another block merely increasing iteration count on the same poorly conditioned coordinates; first apply one invertible conditioning transform that provably leaves the affine-linear hypothesis class unchanged.

Also record the corrected historical H04-A N256 Procrustes seen accuracy as `37.55%` (not the earlier quoted `44.15%`); its missing accuracy remains `21.9875%`.

---

# ACTIVE — H05-B: Invertible relation-conditioning / separability audit

## One scientific objective

Resolve whether H05-A's low support fit is merely numerical conditioning or whether a single shared affine-linear classifier genuinely cannot fit the 256-D paired relation coordinates.

This block must **not add representational capacity**. It is a solver/conditioning audit of the exact H05-A relation space, not a new method.

## Scope / frozen data

Use `fedgh`, seed0, round10 only. Reproduce the committed H02-A online trajectory exactly.

Reuse verbatim:

- H02-E owner support hash `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`;
- H03-A parent anchor hash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`;
- exact H04/H05 N256 prefix;
- H05-A paired and broken coordinate permutations;
- the raw relation formula `r_i(z)=(z-mu_i)(A_i-mu_i)^T`.

Do not resample anything and do not change online FL training.

## Fixed invertible conditioning

For each arm separately (`paired`, `broken`), first form the same pooled 2000 owner-support relation matrix `X in R^{2000 x 256}` used in H05-A, before fitting the head.

Compute **without labels**:

`m_j = mean_n X[n,j]`

`s_j = sqrt(mean_n (X[n,j]-m_j)^2)`

Assert every `s_j` is finite and strictly positive. Do not clamp, add epsilon, drop dimensions, use labels, or tune any threshold.

Define the conditioned coordinate

`x'_j = (x_j - m_j) / s_j`.

Apply the same arm-specific `m,s` to that arm's support and official-test relation vectors. This is a shared diagonal affine bijection of the 256-D coordinates; it changes optimization conditioning but **does not change the class of affine-linear decision functions**.

Add a unit test proving prediction-class equivalence: for an arbitrary linear head on conditioned coordinates, analytically map its weights/bias back to raw coordinates and verify logits agree to numerical tolerance. This test is mandatory because it establishes that H05-B is not secretly a more expressive representation.

No whitening matrix, PCA, random projection, cosine/L2 normalization, temperature, per-client normalization, learned scaling, or regularization is allowed.

## Conditioning diagnostics

For paired and broken support matrices, report before and after conditioning:

- per-coordinate `s_j` min / median / max;
- matrix singular-value max/min-nonzero and numerical rank using one fixed documented tolerance;
- condition number on the nonzero singular spectrum;
- global absolute-value max and RMS;
- finiteness.

Compute SVD diagnostics in float64 if needed for numerical stability, but the actual conditioned classifier input remains the existing float32 unless required by the equivalence test. If the diagnostic SVD warns/fails, report it and do not silently change the scientific arm.

## Matched heads

Fit exactly two heads from fresh zero initialization:

1. `rel256_paired_zscore`
2. `rel256_broken_zscore`

Use the same H05-A classifier family and frozen solver settings:

- linear `256 -> 10`;
- full-batch LBFGS;
- `strong_wolfe`, lr=1;
- `max_iter=2000`;
- `tolerance_grad=1e-9`, `tolerance_change=1e-12`;
- no regularization.

Do **not** increase `max_iter`, try another optimizer, or sweep settings in this block. The point is to test whether the known cap problem disappears when the same linear hypothesis class is well-conditioned.

Record CE before/after, support accuracy, iterations/evaluations, final gradient `inf`/L2 norms, weight/bias norms, head hash, and seen/missing/all/macro plus per-client correct/counts.

## Fixed references / interpretation

Historical raw H05-A values:

- raw paired support fit `65.90%`, missing `11.025%`;
- raw broken support fit `76.65%`, missing `4.8125%`.

Accepted Procrustes N256 reference:

- missing `P = 21.9875%`;
- seen `37.55%`.

For conditioned paired missing `Rz` and conditioned broken missing `Sz`, report:

`q_rel_z = Rz / 21.9875`

`delta_rel_z = Rz - Sz`.

### Decision rules

**A. Conditioning resolves fit.** If conditioned paired support fit is `>=95%`, H05-A is no longer optimizer-unresolved. Then apply the original scientific gate to the conditioned result, because the transform is invertible and does not expand the linear hypothesis class:

- strong relation signal: `q_rel_z >= .70` and `delta_rel_z >= 8pp`;
- simple relation insufficient: `q_rel_z < .40` with adequate paired fit;
- otherwise: intermediate.

Stop after reporting. If strong, the next lead block may test class-level relation prototype compression. If insufficient/intermediate, do not rescue it with another relation kernel or learned encoder yet.

**B. Conditioning does not resolve fit.** If paired fit remains `<95%`:

- if final `grad_inf <= 1e-5`, treat this as evidence that the shared linear relation readout itself is not sufficiently separable under the frozen representation;
- if final `grad_inf > 1e-5`, optimization remains unresolved even after the fixed conditioning transform. Report diagnostics and stop. Do not extend iterations or change solvers without lead review.

For the broken arm, always report support fit and gradients; do not infer pairing necessity from a poorly fit broken head.

## Integrity requirements

All H05-A requirements remain active:

- exact online rounds1..10;
- exact provenance/permutations;
- no anchor labels or test fitting;
- identical frozen state entering both arms;
- no mutation of parameters/buffers/prototypes/module modes/RNG/pre-existing gradients;
- all relations, conditioning statistics, losses/logits/parameters/gradients finite;
- preserve all existing 29 tests and add only minimal conditioning/equivalence tests.

## Deliverable

Append `CODEX REPORT H05-B` with STATUS, source SHA, exact commands/run IDs, tests, provenance, conditioning statistics, singular-spectrum/condition diagnostics, paired/broken fit diagnostics, test metrics, `q_rel_z`, `delta_rel_z`, state/RNG/gradient receipts, warnings, and artifact paths.

Do **not** implement class-relation prototypes, whitening, PCA, cosine/RBF relations, learned encoders, MLPs, OT, gating/hybrid heads, new online losses, other anchor counts, or seeds1/2 in H05-B. Await research-lead review.
