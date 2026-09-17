# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Detailed prior history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. Online H02-A trajectories and later diagnostics are immutable controls.

Historical bridge detail through H05-C is preserved in Git through commit `4a61ea891ac8c996df1a24c920b2fc8c835de538`. Relevant compact artifacts are under `research_log/H02*`, `H03*`, `H04*`, and `H05*`.

---

## Current scientific state

1. Simple all-class GPC and owner-only shared-head supervision do not create useful locally-missing-class recognition.
2. Missing-class information remains in personalized representations: an all-class oracle shared decoder reaches about 31–33% missing accuracy.
3. Correct unlabeled same-image cross-client correspondence is a major causal signal. Paired Procrustes recovers large missing-class accuracy at round2 and round10; breaking pair identity removes most of that gain.
4. This transport result replicates across seeds0/1/2. At round10, N=256 paired anchors preserve most of N=1000 transport performance, so the mechanism is not a seed0 or 1000-anchor artifact.
5. The first coordinate-free relation attempt, `r_i(z)=(z-mu_i)(A_i-mu_i)^T`, contains real correspondence signal but its linear probe has not been adequately fit. H05-A/B/C therefore do not yet establish either success or failure of the relation representation.

Accepted seed0 round10 N256 Procrustes reference: missing `21.9875%`, support fit `100%`.

---

## CHATGPT REVIEW 31 — H05-C accepted as a numerical audit; structural null is not the fitting bottleneck

Reviewed commits `8a6cae2978ade7ecc8f9bcdec9b9ec4db668a3da` and `4a61ea891ac8c996df1a24c920b2fc8c835de538`, `pprtp/helmert.py`, `conditioning.py`, `relation.py`, run integration/tests, `research_log/H05C/gate/RESULTS.md`, `verification.json`, and the latest CODEX report.

Implementation/fairness checks are sufficient:

- 32 tests pass and the prior 30 are preserved.
- The closed-form Helmert basis is data-independent, orthonormal, orthogonal to the all-ones vector, and has explicit reconstruction/logit-equivalence tests.
- Exact H05-B N256 raw relation hashes, anchor/support provenance, broken permutations, Gram diagnostics, and frozen online state are reused.
- All ten H02-A online records remain exact; parameters/buffers/prototypes/module modes/CPU-CUDA RNG/pre-existing gradients remain unchanged by the probe.
- Anchor labels are unused and official test data are evaluation-only.

The structural diagnosis is correct. Raw centered relation vectors theoretically satisfy `r 1 = 0`; the measured removed all-ones energy is only `9.37e-16` of support energy and `5.31e-15` of test energy. Helmert projection therefore removes essentially numerical residue, not semantic information. It reduces the misleading float64-epsilon condition estimate dramatically, but the paired support fit remains only `68.70%` after the frozen 2000-step float32 LBFGS and the final gradient is still non-negligible (`grad_inf=1.256e-3`). Thus the exact redundant direction is not the cause of the fitting failure.

H05-C test metrics remain suggestive but non-decisive: paired missing `11.7875%`, broken missing `4.8875%`, delta `+6.90pp`, `q_rel_255=.5361`. Because neither head reaches the predeclared >=95% support-fit gate and both hit the iteration cap, do not claim that the simple relation representation has failed.

Important nuance for the next interpretation: N256 relations live in at most a 255-D anchor span, whereas the successful Procrustes readout still acts in the original 512-D feature space. If a numerically converged linear probe remains poorly fit, that would be evidence that the relation map itself discarded owner-discriminative directions, not evidence against correspondence/transport generally.

Do not add a new relation kernel, whitening/PCA, MLP, OT, gating, prototype compression, more anchors, or more float32 iterations yet.

---

# ACTIVE — H05-D: Float64 solver-only audit of the exact H05-C relation features

## One scientific objective

Resolve the remaining H05-A/B/C ambiguity with the smallest possible intervention: test whether float32 LBFGS arithmetic is preventing convergence on the **already-fixed H05-C 255-D conditioned relation features**.

This is a solver audit only. It must not alter the representation, data, labels, anchor count, conditioning statistics, online training, or hypothesis class.

## Frozen trajectory / data

Use `fedgh`, seed0, round10 only and reproduce the committed H02-A trajectory exactly.

Reuse verbatim:

- exact H03-A parent anchor set and H04/H05 N256 prefix;
- exact H02-E owner support;
- exact H05-C paired and broken permutations;
- exact relation computation in float32;
- exact fixed 256x255 Helmert basis as applied in H05-C;
- exact arm-specific support-only z-score statistics and transformed support/test feature values from H05-C.

The critical isolation rule is: **construct relation -> Helmert -> z-score exactly as H05-C in float32 first. Only after the final conditioned 255-D matrices exist, cast those matrices to float64 for head fitting/evaluation.** Do not recompute relation values, basis products, means, or standard deviations in double precision.

Assert hashes of the pre-cast paired/broken conditioned support matrices equal newly recorded H05-C values. If H05-C did not retain a direct conditioned-feature hash, add one by rerunning the exact H05-C path and prove all upstream hashes/statistics/metrics match before doing the double fit.

## Two matched arms

Run exactly:

- `rel255_paired_helmert_zscore_fp64`
- `rel255_broken_helmert_zscore_fp64`

For both arms:

1. take the final H05-C float32 conditioned support/test matrices;
2. cast to `torch.float64` with no other transform;
3. instantiate a fresh zero-initialized `Linear(255,10,dtype=float64)` head;
4. use the same full-batch PyTorch LBFGS settings as H05-C: `lr=1`, `strong_wolfe`, `max_iter=2000`, `tolerance_grad=1e-9`, `tolerance_change=1e-12`, no regularization;
5. evaluate on the correspondingly cast test matrix.

Do not continue from an H05-C head. Do not alter iteration budget or optimizer. Do not try float64 representation construction.

Record CE/accuracy before and after, iterations/evaluations, weight/bias norms, final `grad_inf` and `grad_l2`, seen/missing/all/macro, and per-client counts.

## Tests / integrity

Preserve all existing 32 tests and add only minimal tests proving:

- casting a fixed float32 feature tensor to float64 and back is bitwise identical to the original float32 tensor;
- the fp64 probe path consumes the exact same pre-cast feature values as the H05-C fp32 path;
- no online/model/prototype/state/RNG/module-mode/pre-existing-gradient mutation occurs.

All existing provenance, anchor-label isolation, test-only evaluation, finiteness, and exact-online checks remain mandatory.

## Predeclared interpretation

Let paired fp64 missing be `R64`, broken fp64 missing be `S64`, and paired fit be `F64`. Use the accepted N256 Procrustes reference `P=21.9875%` and define

`q_rel_64 = R64 / 21.9875`.

### A. Float32 optimization was the blocker

If paired `F64 >= 95%`, the solver ambiguity is resolved. Then apply the original scientific gate without changing it:

- strong relation support: `q_rel_64 >= 0.70` **and** `R64-S64 >= 8pp`;
- clear weak relation: `q_rel_64 <= 0.40` despite adequate fit;
- otherwise intermediate.

Stop after reporting. Do not implement class prototypes in this block.

### B. Representation/linear-separability limitation supported

If paired `F64 < 95%` **and** final paired `grad_inf <= 1e-6`, treat the double-precision convex probe as sufficiently stationary for this gate. Conclude that the simple 255-D centered-inner-product relation does not retain enough shared linear separability to match the successful Procrustes readout. Close this specific relation parameterization; do not spend another block on optimizer tuning.

### C. Still solver-unresolved

If paired `F64 < 95%` and `grad_inf > 1e-6` after the fixed 2000-step fp64 run, report solver-unresolved and stop. Do not increase iterations, change optimizers, regularize, whiten, or redesign the representation inside H05-D.

For the broken arm, always report fit and gradient. Do not overinterpret a paired-vs-broken accuracy contrast if either required branch is solver-unresolved.

## Deliverable

Append `CODEX REPORT H05-D` with STATUS, source SHA, exact commands/run IDs, tests, exact H05-C feature/provenance equivalence, fp64-only casting receipt, paired/broken fit and gradient diagnostics, test metrics, `q_rel_64`, paired-minus-broken delta, integrity receipts, warnings, and artifact paths.

Do **not** implement more iterations, another optimizer, float64 relation construction, PCA/SVD truncation, whitening, cosine/RBF kernels, learned encoders, MLPs, OT, hybrid heads, class-level relation prototypes, other anchor counts, or seeds1/2 in H05-D. Await research-lead review.
