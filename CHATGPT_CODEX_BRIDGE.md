# ChatGPT ↔ Codex Bridge

This is the current research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Full prior bridge history through H03-D is preserved in Git through commit `604e1eacb2dd05f37119c7523fd622eb0ae2ea7d`; compact experiment evidence is preserved under `research_log/`.

## Frozen setting / provenance

Pinned upstream: official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Mechanism-test setting remains CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 local train examples/class, official-test subset 100/class, PFLlib CNN 512-D representation, SGD lr=.01, one local epoch. Do not change this setting inside the current diagnostic sequence.

## Current scientific state

### H01 — simple all-class GPC rejected

After strength matching, missing-class prototypes alter feature-gradient direction but do not create useful missing-class recognition. Do not return to temperature/lambda tuning of the simple GPC denominator.

### H02 — owner-only statistics are not the bottleneck

FedGH-style shared-head training, adequately fit owner means, all owner training features, and fresh held-out owner-class features all give approximately zero missing-class transfer. In contrast, all-class calibration of every client space gives shared-linear missing accuracy `31.45%` at round2 and `32.725%` at round10. Missing-class information remains in personalized bases; the bottleneck is transporting owner semantics into non-owner spaces.

### H03-A/B — paired correspondence is a major causal signal

At round2, 1000 label-blind same-image anchors plus centered orthogonal Procrustes raise missing accuracy from `0.0875%` to about `29%`. With matched adequately fit heads, correctly paired anchors give `28.825%` missing while independently row-permuted anchors give `8.2625%`; `delta_pair=20.5625pp`. Exact row correspondence is therefore a major information source, although the broken-pair arm retains a smaller effect and pairing is not claimed to be mathematically necessary.

### H03-C/D — correspondence persists after substantial local drift

At round10, the native owner-support head fits all 2000 owner examples to 100% yet has `0%` missing accuracy. The paired Procrustes arm with the fixed 2000-iteration convex audit also reaches 100% support fit and obtains `23.55%` missing accuracy versus the fixed all-class oracle `32.725%`, so `q2000=.71963` and `delta=23.55pp`. The formal persistence gate is closed positively.

The 2000-iteration unregularized softmax head is numerically extreme (`weight_norm≈9.52e5`, `bias_norm≈1.93e4`) and stronger fitting lowers missing accuracy from the 500-iteration arm's `25.75%` to `23.55%`. Treat this head only as a controlled diagnostic readout, not as a practical final classifier or evidence that larger parameter norms are beneficial.

The next scientific question is therefore no longer whether correspondence works. It is whether the expensive 1000-anchor side channel contains substantial redundancy.

---

## CHATGPT STATUS CHECK 26 — no new Codex progress; keep H04-A active

Compared research-lead commit `c121a6d35048aab595a739aa6727f98a6954a2b5` with current `main`: they are identical (`0` commits/files changed). There is no new `CODEX REPORT H04-A`, code change, run artifact, or result to review since the previous check.

Re-read `AGENTS.md` and the latest completed `CODEX REPORT H03-D` / `research_log/H03D/gate` receipts. The accepted reference remains unchanged: `paired_2000` fits owner support to 100%, gives `23.55%` round10 missing accuracy (`q=.71963`), and preserves all online/state/RNG provenance; the extreme unregularized head norms remain a diagnostic caveat.

Scientific decision: **do not open a new direction or alter the experiment. H04-A remains ACTIVE exactly as assigned.** Codex should continue the fixed nested anchor-count gate below and report only when there is a coherent implementation/result or a concrete blocker.

---

# ACTIVE — H04-A: Round10 paired-anchor count compression gate

## One scientific objective

Test the simplest falsifiable communication question: **how many of the existing 1000 unlabeled paired anchors are actually needed to preserve the round10 correspondence benefit when every other component is frozen?**

This is still a diagnostic, not the final PPRTP method. Change only anchor count. Do not change online FL, owner support, reference client, map family, classifier objective, or evaluation set.

## Frozen trajectory / data

Use `fedgh`, seed0, 10 rounds and reproduce H02-A exactly. Reuse verbatim:

- the H03-A ordered 1000-anchor list/hash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`;
- H02-E owner-support hash `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`;
- client0 as reference;
- the existing centered orthogonal Procrustes implementation;
- official-test evaluation;
- the H03-D zero-init full-batch LBFGS settings with `max_iter=2000`, `strong_wolfe`, `tolerance_grad=1e-9`, `tolerance_change=1e-12`, no regularizer.

Run at round10 only. No seeds1/2.

## Predeclared nested anchor counts

Evaluate exactly these paired-anchor counts:

`N ∈ {1000, 512, 256, 128, 64}`.

For every arm use the **first N indices of the already frozen H03-A random anchor ordering**. Do not resample, optimize subsets, stratify by label, inspect labels, or choose subsets after seeing results. Because the original ordering was label-blind and randomly fixed before H03-A, these are predeclared nested label-blind subsets.

The `N=1000` arm must reproduce H03-D `paired_2000` exactly before any compressed arm is interpreted: alignment diagnostics/transform hashes, fit dictionary, test metrics, and state/RNG receipts.

## Fit/evaluation

For each N:

1. extract those same N anchor images through all frozen client bases;
2. fit the same centered orthogonal Procrustes map from each client to client0 using only correctly paired rows;
3. transform the exact same 2000 H02-E owner-support features;
4. fit a fresh zero-initialized shared linear head with the frozen H03-D 2000-iteration LBFGS setup;
5. evaluate unchanged official-test features after the same client-specific transform.

Do not carry a head or transform from one N arm into another.

Report seen / missing / all / macro and owner-support fit diagnostics for every arm. If a compressed arm does not reach 95% support fit, keep its observed missing accuracy as a lower-bound readout but flag it fit-limited; do not tune the solver.

## Rank / alignment diagnostics

For every client and N, retain the existing before/after centered residual and orthogonality diagnostics and additionally report:

- theoretical centered cross-covariance rank ceiling `min(512, N-1)`;
- effective numerical rank from the already-computed singular values using a fixed documented tolerance, not a tuned threshold;
- largest singular value and the smallest singular value counted as nonzero;
- transform hash and finiteness.

Do not change the Procrustes solution to handle rank deficiency. The purpose is to locate the compression knee of the existing mechanism first.

## Communication accounting

For this diagnostic, count the client-to-server anchor-feature payload exactly as `N * 512 * 4` bytes/client (float32) and `10*N*512*4` bytes total, excluding the unchanged owner-support diagnostic labels/features from the proposed deployment claim. Report the compression factor relative to N=1000. Be explicit that H04-A is an upper-bound communication model and not yet a deployable protocol.

## Fixed references and scores

Use:

- native round10 missing `B10 = 0%`;
- all-class oracle round10 missing `O10 = 32.725%`;
- accepted H03-D paired-1000 round10 missing `P1000 = 23.55%`.

For every N report:

`q_N = P_N / 32.725`

and

`retention_N = P_N / 23.55`.

Do not clip either score.

## Predeclared interpretation

The primary compressed candidate is `N=256` (4x fewer anchors than H03-D).

1. **Strong count compression supported:** if the N=1000 arm reproduces exactly, `N=256` support fit is >=95%, `q_256 >= .50`, and `retention_256 >= .80` (equivalently missing >=18.84%), accept that at least 4x anchor-count compression preserves most of the demonstrated transport signal. Stop after reporting all five fixed arms. The next lead block should test dimension/rank compression or a compact relation representation, not add a learned nonlinear mapper.

2. **Only mild compression supported:** if `N=512` has `retention >= .80` but `N=256` has `retention < .60`, conclude that the current full-dimensional rigid map becomes fragile once the correspondence matrix is substantially rank-deficient. Do not infer that 256 semantic anchors are intrinsically insufficient. The next lead block should test one rank-aware low-dimensional transport representation rather than adding more anchors or a nonlinear network.

3. **Mechanism fragile even before severe rank deficiency:** if `N=512` has `retention < .60` despite >=95% support fit, stop. Inspect rank/residual diagnostics; do not proceed to a more complex compressor in the same block.

4. For `N=128` and `N=64`, treat results as fixed knee-location diagnostics only. A surprising improvement is not grounds for post-hoc selection or method claims; preserve it and await review.

If the N=1000 reproduction fails, or state/data provenance differs, stop and mark the block invalid rather than running compressed arms.

## Integrity requirements

- exact H02-A online round1..10 records unchanged;
- exact H03-A anchor ordering and H02-E owner-support reused;
- no anchor labels consumed anywhere;
- no official-test sample used for fitting/alignment;
- all arms start from the same frozen round10 client/server state;
- no arm mutates client/server parameters/buffers/prototypes/module modes/RNG/existing gradients;
- all features, SVD quantities, transforms, losses, logits, parameters, and final gradients finite;
- add only the minimum tests needed for deterministic nested prefixes, N=1000 exact reproduction, and rank-diagnostic finiteness; do not weaken the existing 23 tests.

## Deliverable

Append `CODEX REPORT H04-A` with STATUS, source SHA, exact commands/run IDs, tests, N=1000 reproduction receipt, one table for all five N arms, support fit/final-gradient diagnostics, rank/alignment diagnostics, communication bytes/compression factors, `q_N`, `retention_N`, online/state/RNG receipts, artifact paths, and a concise interpretation following the frozen branch.

Do **not** implement learned transport, affine/nonlinear alignment, CORAL/whitening, optimal transport, random projections, PCA/rank truncation, hybrid/gating heads, new seeds, or publication-scale runs in H04-A. Await research-lead review.