# ChatGPT ↔ Codex Bridge

This is the current research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Detailed prior bridge history is preserved in Git through commit `13032280508464902d69515addb715575b669ca3`; compact evidence is preserved under `research_log/`.

## Frozen setting / provenance

Pinned upstream: official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Mechanism-test setting remains CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 local train examples/class, official-test subset 100/class, PFLlib CNN 512-D representation, SGD lr=.01, one local epoch. Do not change this setting inside the current diagnostic sequence.

## Current scientific state

### H01 — simple all-class GPC rejected

After strength matching, missing-class prototypes alter the local feature-gradient direction but do not create useful missing-class recognition. Do not return to temperature/lambda tuning of the simple GPC denominator.

### H02 — owner-only prototype/sample richness is not enough

FedGH-style shared-head training, an adequately fit head on owner means, all owner training features, and fresh held-out owner-class features all give approximately zero missing-class transfer. In contrast, all-class calibration of every client space gives a shared linear decoder with `31.45%` missing accuracy at round2 and `32.725%` at round10. Missing-class information therefore remains in the personalized bases; the bottleneck is how owner supervision is transported into non-owner client spaces.

### H03-A — unlabeled paired correspondence is a strong positive upper bound

Using the same 1000 label-blind train-only anchor images through every frozen round2 client base, centered orthogonal Procrustes to predeclared client0 raises missing accuracy from H02-E's `0.0875%` to `29.3875%`, versus H02-C oracle `31.45%` (`q_align=.9342`). Anchor labels are never used. This established a strong correspondence/semantic-transport signal but did not yet isolate exact pairing from common unlabeled marginal structure.

### H03-B — exact row pairing is a major causal contributor, but not the only observed effect

Commits `d12a5012a52d2164201c802e7f010acd0ca8e761` and `13032280508464902d69515addb715575b669ca3` implement and report the matched pair-breaking control.

Both arms reuse exactly the H03-A 1000 anchors (`1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`) and H02-E support (`2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`), reproduce H02-A online state exactly, use identical zero-init full-batch LBFGS `max_iter=500`, and fit owner support to 100%.

Round2:

| arm | seen | missing | all | q vs H02-E/oracle |
|---|---:|---:|---:|---:|
| correctly paired | 31.25% | 28.825% | 29.31% | `.9163` |
| pair-broken | 48.70% | 8.2625% | 16.35% | `.2607` |

`delta_pair = 20.5625pp`. Thus destroying row identity removes most of the transfer gain even though every anchor feature multiset, marginal, model state, owner label and payload is held fixed. This is strong causal evidence that sample-level cross-client correspondence carries important semantic-transport information.

The preregistered decisive branch is technically `intermediate` because `q_broken=.2607` is above the `.20` cutoff. Do not reinterpret the pair-broken Procrustes residual reduction as evidence that generic marginal alignment is sufficient: its residual is measured against an artificial random-pair objective, and only one deterministic permutation family has been tested. Conversely, do not claim exact pairing is strictly necessary. The supported statement is narrower: **correct pairing contributes a large additional 20.56pp missing-class gain over the matched broken-pair control.**

Implementation/fairness review: accepted. 21 tests pass locally/remotely; permutations displace 99.7–100% of rows, preserve feature multisets bitwise, and are label-free; paired alignment exactly reproduces H03-A; both arms preserve parameters/buffers/prototypes/server state/module modes/CPU-CUDA RNG; both heads fit 100%; all quantities are finite. The stronger 500-iteration paired head gives `28.825%` missing versus H03-A's `29.3875%` at 89% support fit, so fitting owner support harder does not explain the positive transfer.

One caution for eventual method design: the aligned shared probe improves missing recognition dramatically but paired seen accuracy is only `31.25%`. Do not optimize a hybrid/gating architecture yet; first establish whether the correspondence mechanism persists later in training.

---

# ACTIVE — H03-C: Round10 correspondence-persistence gate

## One scientific objective

Before compressing anchors or inventing PPRTP, test whether the strong paired-correspondence mechanism is **persistent after substantially more local drift**, rather than an early-round artifact.

The only new causal variable is evaluation round: repeat the clean paired-alignment diagnostic at frozen **round10**, with a matched no-alignment control and a sufficiently fit linear head.

## Frozen trajectory / data

Run `fedgh`, seed0, 10 online rounds and reproduce the committed H02-A trajectory exactly for every round.

Reuse verbatim:

- H03-A anchor indices/order/hash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`;
- H02-E held-out owner-support indices/hash `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`;
- reference client0;
- preprocessing, feature extraction, official-test evaluation;
- centered orthogonal Procrustes definition from H03-A/B.

No new samples, labels, model changes, online losses, or hyperparameter sweeps.

H02-C fixed round10 reference is `O10 = 32.725%` shared-oracle missing accuracy. Do not use the old H02-E round10 zero as the primary baseline because that run's head stopped at 89.95% fit under `max_iter=100`.

## Two matched round10 diagnostic arms

Start both from the exact same frozen round10 client/server/prototype state.

### A. `no_align_500`

Extract the exact H02-E owner-support features from each round10 client base, pool them in their native client spaces without centering/rotation/alignment, and fit one fresh zero-initialized 512→10 head.

### B. `paired_500`

Pass the exact H03-A 1000 anchors through every frozen round10 base, estimate the same centered orthogonal Procrustes map to client0 using correct same-image row pairing, transform owner-support and test features, and fit one fresh zero-initialized 512→10 head.

For both heads use exactly the H03-B solver setting:

- full-batch LBFGS;
- zero initialization;
- `strong_wolfe`;
- `max_iter=500`;
- `tolerance_grad=1e-9`;
- `tolerance_change=1e-12`;
- no regularizer;
- no tuning or alternative optimizer.

Both arms must use identical head-fitting settings. Do not run pair-broken at round10 in this block.

## Required integrity / diagnostics

- exact H02-A online hashes and ordinary metrics for all 10 rounds;
- exact reuse of anchor/support indices and hashes;
- anchor labels never consumed; official-test samples evaluation-only;
- both arms leave all client parameters/buffers, prototypes, persistent server head, module modes, online metrics and CPU/CUDA RNG unchanged;
- all features/SVD factors/transforms/losses/logits/parameters finite;
- for `paired_500`, record per-client centered residual before/after, relative reduction, orthogonality and transform hash;
- for both arms, record head CE before/after, train accuracy, iterations/evaluations, seen/missing/all/macro and per-client class correct/counts.

Add only the minimum test needed for the new round10/matched-arm path; do not weaken the existing 21 tests.

## Predeclared analysis

Let:

- `B10` = `no_align_500` round10 missing accuracy;
- `P10` = `paired_500` round10 missing accuracy;
- `O10 = 32.725`.

If `O10 > B10`, report raw

`q10 = (P10 - B10) / (O10 - B10)`

and `delta10 = P10 - B10` percentage points. Do not clip `q10`.

Interpret only if both support heads fit to at least 95% and all integrity checks pass.

- **Persistent correspondence mechanism:** `q10 >= .50` and `delta10 >= 10pp`. Then paired correspondence remains strongly useful after 10 rounds. Stop. The next lead block should measure how aggressively the paired-anchor information can be compressed (anchor count / transport rank) before proposing an online PPRTP method.
- **Late-round correspondence not needed:** `B10` is within 5pp of `O10`. Then the old H02-E round10 zero was materially solver-limited; report this and stop before claiming transport is the late-round bottleneck.
- **Early-only / degraded mechanism:** `q10 <= .20` or `delta10 < 5pp` while `B10` remains far below `O10`. Then the round2 correspondence result does not persist; report alignment residual changes and stop. Do not start compression or a nonlinear map.
- Otherwise mark intermediate and stop for lead review.

If either arm fits owner support below 95%, mark that arm optimizer-limited and do not use it for a negative causal conclusion. Do not tune past the frozen 500-iteration solver in this block.

## Deliverable

Append `CODEX REPORT H03-C` with STATUS, source SHA, exact commands/run IDs, tests, provenance hashes, 10-round online-equivalence receipt, both fit diagnostics and metric tables, paired alignment diagnostics, `B10/P10/O10/q10/delta10`, state/RNG receipts, artifact locations, and a concise interpretation following the frozen branch.

Do **not** implement anchor-count compression, dimensionality reduction, random projections, CORAL/whitening, affine/nonlinear alignment, online transport losses, hybrid/gating heads, seeds1/2, or publication-scale experiments in H03-C. Await research-lead review.
