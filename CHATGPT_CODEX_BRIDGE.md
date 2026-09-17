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
## CODEX REPORT H05-A — DONE (2026-09-18 01:14 +08)

STATUS: DONE; **fit-limited / optimization unresolved**,no negative representation verdict. Source `de2f60cd10217761345c6416378087159ad88e03`;release `20260918-011059-h05a`;run `20260918-011118-h05a-relation`,exit0. Exactly paired/broken relation256 arms,seed0round10. No normalization,scaling,SVD,map,prototype compression or solver extension.

Files: `pprtp/relation.py`,opt-in `pprtp/run.py --relation-probe`,`tests/test_relation.py`,`scripts/report_h05a.py`.29 tests pass locally/remotely,previous27 preserved. New double-precision synthetic test verifies orthogonal rotation+translation invariance;column permutation test verifies deterministic client0identity and per-row bitwise multiset preservation.

Commands (AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h05a
./scripts/autodl-run.ps1 -Name h05a-relation -Cmd 'PPRTP_SOURCE_SHA=de2f60cd10217761345c6416378087159ad88e03 bash scripts/run_h01.sh --modes fedgh --seeds 0 --rounds 10 --relation-probe'
D:\anaconda3\python.exe scripts/report_h05a.py research_log/H05A/gate
```

Formula exactly `(z-mu)@(A-mu).T`,computed in existing float32;mu is the mean of the256 anchors,not support/test. No normalization,temperature,whitening,scaling or learned encoder. Gram is `(A-mu)@(A-mu).T`,client0 used only for reporting disagreement. Each arm independently creates a fresh zero-init256->10 head,fullbatch LBFGS lr1,max_iter2000,strong_wolfe,tolerance_grad1e-9,tolerance_change1e-12,no regularizer. Random head construction occurs within fork_rng and initial parameters are zeroed before fitting.

All10 H02-A client/prototype/server/ordinary metric records exactly reproduce. Both arms start identical and preserve client/server parameters/buffers/prototypes,module modes,CPU/CUDA RNG and pre-existing gradients. All anchorfeatures/relations/Grams/losses/logits/parameters/final gradients finite. Exact parent anchor hash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`,support hash `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`;prefix indices/hash equal H04-A N256 verbatim. Anchorlabels unused,test evaluation-only. Parent/prefix receipts saved.


| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE before/after | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| rel256_paired | 56.550001 | 11.025000 | 20.130000 | 20.130000 | 65.900004 | 2.30258393/1.16467738 | 0.0284156315 | 0.255756944 | 68.8054428/1.03762555 | 2000/2116 |
| rel256_broken | 60.099999 | 4.812500 | 15.870000 | 15.869999 | 76.650006 | 2.30258393/0.855615675 | 0.00924307108 | 0.109551854 | 66.7388916/1.2084285 | 2000/2083 |

R=11.025000%,S=4.812500%,q_rel=0.501421258,delta_rel=6.212500pp. Frozen branch: fit-limited; do not reject representation from weak negative result.
Historical H04-A N256 Procrustes seen=37.550000% (corrects44.15 in assignment);missing21.9875%.

| Client | Gram relative disagreement | Gram SHA256 |
|---|---:|---|
| 0 | 0 | 77ceec3253c61bece612fa02d9b51dbb07a66baefa7baf7b4b551a71b340cc06 |
| 1 | 0.153230652 | ba1c91208e7f2bb4f01c6203f852745dd1c0b02d0b63cfe3ba1e3a547cef6b21 |
| 2 | 0.239965186 | a4b7451338a4452ae45705727e2c5dc70c35af8512366f297e62f7591b3e312c |
| 3 | 0.252810091 | 32eead46577a5ad7458e7ad69a4cdc28928ffd7124b9aea6c7181ef48b96ff79 |
| 4 | 0.446717262 | 2ae3165605610059c21c45a6cf0d9c72b38a4484a8d31ce445dc4340d9464fa1 |
| 5 | 0.432853967 | eb85cf17b57605d21221e124b767a5867e55041432bc76fcc55a527a64721a0d |
| 6 | 0.929122806 | e3aa0b96769a366a850aa6528cb277457e38fea33fc2504a6832d54beac37b5f |
| 7 | 1.10199749 | 032482f582064e2ea818a6e6a4042a677e498c2066e98edf0e758ab9febe6d76 |
| 8 | 0.826244652 | c5f3652a1e4dfb7fb55259cead55ffb69532c6fea9a6f34d9bd776a4206c2b98 |
| 9 | 0.475106955 | 574fa0ea4b20c777c4d35c203227f4ecc9a854022148540ee5d4e4ae1c2bc5b7 |

| Client | Seed | Fixed points | Permutation SHA256 |
|---|---|---:|---|
| 0 | None | 256 | be66c2867e103fa00c88539675e5cd95f52ec5e38d9038e6d3710539f6c0aec1 |
| 1 | 314160 | 1 | 7895f8a7b8620d25b2b6a81e99687710923cf2b4f48f6b61511fe5d05ae0db8d |
| 2 | 314161 | 0 | 7a1e35d9a2cf823ea7eafacb340e4d8bba42be2ef0516c6e67390109cbcf61e2 |
| 3 | 314162 | 1 | a6a914cc77eaf8cd986a4616f3a0d36c2cc0de09d5b6995c00acb771eecc1954 |
| 4 | 314163 | 2 | 62d25aef2bf949197fa7029539d9ea81137c19bfa39226d05bbbb69c4ff8ef6b |
| 5 | 314164 | 1 | d8d4c6c995c4da41324d0cf9d4b5da25641374228c6a1f320d3f697d55e99947 |
| 6 | 314165 | 0 | 136cd1c938ff040a04f13703f717e4356eff5177108e1766e9fac9ce5502771f |
| 7 | 314166 | 2 | 1ac58997a7ab0c004b1466cd62e43a9d7845bd62cb4c96abd9c261ac1c7ff5c1 |
| 8 | 314167 | 3 | 10e019895cc3e963d13e8d5da9879d378a709b48d0b7d16c6a4da88c36910e46 |
| 9 | 314168 | 1 | ddc83efc6063357edde23d9887de924e56f8a6ca1300b524fdecc6c8da877683 |

Permutations fixed before execution as314159+i for clients1..9;client0identity. Full permutations saved in rawJSON. Each client's SAME permutation applies to support and test columns;inverse permutation recovers each row bitwise. Raw sample order/labels unchanged. Gram diagnostics identical across arms. Nonreference fixedpoints1,0,1,2,1,0,2,3,1 (all at least253/256 coordinates displaced).

Interpretation: observed paired missing11.025% exceeds broken4.8125% by6.2125pp and retains50.14% of the accepted Procrustes missing accuracy. However paired supportfit65.90% and broken76.65% both fail95%,both hit2000iterations,and final grad_inf .02842/.00924 remain non-negligible. The formal relational gate is therefore unresolved by fitting,not a valid rejection of centered inner-product relations or proof of pairing necessity. Gram disagreements range.1532–1.1020,so actual client spaces do not share identical centered anchor geometry despite the synthetic invariance property. Stop and await lead;no solver extension,normalization,alternative representation or class-relation prototypes implemented.

Historical reference correction: the assignment quoted N256 Procrustes seen44.15%;committed H04-A raw value is37.55%,used as explicitly instructed;missing21.9875% unchanged. Known NVML initialization warnings retained verbatim in train.log;CUDA run completed;no SVD used in this diagnostic and no SVD warning. Ddrive8192bytes shortage repaired by eviction of one verified duplicate ignored checkpoint;remote original and hash receipt retained in progress.md.

Artifacts: `research_log/H05A/gate/` RESULTS.md,verification.json,full per-client support/test correct/counts,head/Gram hashes,permutations,state/RNG/gradient receipts,meta/run/log/tests. Remote originals/checkpoints `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-011118-h05a-relation`. No failed run or nonfinite results.
