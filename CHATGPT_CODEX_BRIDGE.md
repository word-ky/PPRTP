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

## CODEX REPORT H05-B — DONE (2026-09-18 02:22 +08)

STATUS: DONE; **optimization remains unresolved after fixed conditioning**. Source `c8c5c2ec405f4134a709f45df63c7cdf854c6d9a`;release `20260918-021930-h05b`;run `20260918-021953-h05b-conditioning`,exit0. Exactly two zscore arms,seed0round10. No additional iterations,solver,whitening,feature family or online change.

Files: `pprtp/conditioning.py`,opt-in paths in `pprtp/relation.py`/`pprtp/run.py`,`tests/test_conditioning.py`,`scripts/report_h05b.py`.30 tests pass locally/remotely;previous29 preserved. Mandatory test maps arbitrary conditioned-head W,b back to raw coordinates as Wraw=W/s,braw=b-Wraw@m and verifies logits to1e-12 in double. It also checks unit population variance/zero mean and rejection of zero std without clamping. Thus the diagonal affine transform preserves the affine-linear hypothesis class;it does not imply identical optimizer implicit bias or identical finite-precision solutions.

Commands (AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h05b
./scripts/autodl-run.ps1 -Name h05b-conditioning -Cmd 'PPRTP_SOURCE_SHA=c8c5c2ec405f4134a709f45df63c7cdf854c6d9a bash scripts/run_h01.sh --modes fedgh --seeds 0 --rounds 10 --conditioning-probe'
D:\anaconda3\python.exe scripts/report_h05b.py research_log/H05B/gate
```

For each arm,mean/std are computed from its pooled2000 owner-support relation rows only,in float32 using population std=sqrt(mean squared centered values). Every std strictlypositive/finite;noepsilon/clamp/dimensiondrop. Same mean/std applied to all clients' support/test relation rows for that arm. No labels,test or perclient statistics used for conditioning. SVD diagnostics use float64 on the actual before/after matrices,while head/input stay float32. Fresh zero-init256->10/fullbatch LBFGS lr1,strong_wolfe,max_iter2000,tolerance_grad1e-9,tolerance_change1e-12,no regularizer unchanged.

Exact parent anchor hash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`;support hash `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`;N256 prefix exactly H05-A. Historical Gram hashes/disagreements,permutations and frozenstate exactly match H05-A. Permutations still314159+i,client0identity;support/test relation multisets bitwise preserved before conditioning. Both arms same startingstate;all10 H02-A client/prototype/server/ordinary metrics exact;parameters/buffers/prototypes,module modes,CPU/CUDA RNG and existing gradients unchanged. All features,statistics,SVD,losses/logits/headparameters/finalgradients finite.


| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE before/after | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| rel256_paired_zscore | 56.149999 | 12.012500 | 20.840000 | 20.840000 | 68.650001 | 2.30258393/1.07599366 | 0.00222687563 | 0.0183848478 | 774.708557/0.930271804 | 2000/2133 |
| rel256_broken_zscore | 59.850000 | 4.925000 | 15.910000 | 15.910000 | 76.450002 | 2.30258393/0.855554402 | 0.00147384172 | 0.0186458286 | 810.236938/0.686846137 | 2000/2097 |

Rz=12.012500%,Sz=4.925000%,q_rel_z=0.546333148,delta_rel_z=7.087500pp. Frozen branch: optimization remains unresolved after fixed conditioning.

| Arm | Matrix | Std min/median/max | Singular max/min-nonzero | Rank | Tolerance | Condition nonzero | Abs max | RMS |
|---|---|---|---|---:|---:|---:|---:|---:|
| rel256_paired_zscore | before | 1.57717165/9.1773968/35.8514184 | 8651.9041/7.56336555e-05 | 256 | 3.84221726e-09 | 114392251 | 176.526276 | 12.6599894 |
| rel256_paired_zscore | after | 0.999999883/0.999999973/1.00000008 | 618.458245/5.648753e-06 | 256 | 2.74650634e-10 | 109485801 | 7.08438349 | 0.999999971 |
| rel256_broken_zscore | before | 5.65783097/12.1932904/23.4394305 | 4421.63347/7.60112406e-05 | 256 | 1.96359971e-09 | 58170784.1 | 176.526276 | 12.6599894 |
| rel256_broken_zscore | after | 0.999999905/0.999999975/1.00000007 | 325.865825/5.41460697e-06 | 256 | 1.44713497e-10 | 60182729.2 | 9.34680557 | 0.999999977 |

rel256_paired_zscore conditioning statistics SHA256: 9844754bca15db5922bb7ae2d96c0f7e978f890d7894e033e80f26dcf6385651; raw support SHA256: d05243ebd179d5e2acf8b319b71557d643348495b7ca4ba49bee7998428d0273.


rel256_broken_zscore conditioning statistics SHA256: 12cc7f9c35da131a7d6297b6761cc2fcc9dd81ffbb6bc68a24e4f97445445611; raw support SHA256: cdee85fe25a34a1f353494907908c4ecb19712ab4fdde527b95ccd9ec342508a.

SVD diagnostic tolerance: max(matrix_shape)*eps64*smax. SVD uses the actual uncentered matrix before/after zscore, float64 diagnostics only. Std diagnostics are recomputed in double (median=0.5 quantile); actual mean/std and inputs remain float32. Full mean/std vectors in raw JSON; no labels or test samples used for statistics.

Both heads hit2000cap with paired/broken supportfit68.65/76.45% (raw H05-A65.90/76.65%). Neither95% threshold is reached. Paired grad_inf .0022269 >1e-5 selects the frozen unresolved branch;broken .0014738 also remains non-negligible. Accepted N256 Procrustes reference remains missing21.9875%,seen37.55%.

Interpretation: shared zscore improves paired observed supportfit modestly and missing accuracy to12.0125%,but does not resolve the optimization gate. Despite nearunit marginal variances,the nonzero-spectrum condition numbers remain about1.095e8 paired/6.018e7 broken;diagonal scaling has not eliminated the strong matrix ill-conditioning. The prescribed float64 rank diagnostic counts256 directions in the computed float32 relation matrices;this numerical observation is not a claim that the mathematically centered relation has no near-null direction. With supportfit below95% and gradients above1e-5,neither a representation failure nor a stationary separability ceiling is established. Stop and await lead without more iterations,whitening,another kernel or class-prototype compression.

Warnings: known Can't initialize NVML warnings preserved verbatim;CUDA run successful. No SVD warning/failure or nonfinite result in this run. Full mean/std vectors and hashes,rawsupport hashes,spectrum statistics,per-client support/test counts and all isolation receipts are in rawJSON. Evidence `research_log/H05B/gate/` includes RESULTS.md,verification.json,artifacts,meta/run/log/tests. Remote originals/checkpoints `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-021953-h05b-conditioning`.
