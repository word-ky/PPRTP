# ChatGPT ↔ Codex Bridge

This is the current research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Full prior bridge detail through H05-B is preserved in Git through commit `3b7e9c2ef78ce2127c345089dbeb0402c40da119`; compact experiment evidence is preserved under `research_log/`.

## Frozen setting / provenance

Pinned upstream: official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Mechanism-test setting remains CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 local train examples/class, official-test subset 100/class, PFLlib CNN 512-D representation, SGD lr=.01, one local epoch. Do not change this setting inside the current diagnostic sequence.

## Current scientific state

### H01/H02 — simple prototype / owner-only explanations rejected

Simple all-class GPC, FedGH owner means, all owner training features, and fresh held-out owner-class features do not create useful missing-class recognition even when the diagnostic classifier is adequately fit. All-class calibration of every client space reaches shared-linear missing accuracy `31.45%` at round2 and `32.725%` at round10. Personalized bases retain missing-class information, but owner semantics are not directly transportable into non-owner spaces.

### H03/H04 — paired correspondence is causal, persistent, compressible, and cross-seed robust

Correct same-image correspondence is a major transfer signal. At round2, paired Procrustes gives `28.825%` missing versus `8.2625%` after matched pair breaking (`+20.5625pp`). At round10, native owner supervision fits 100% yet gives `0%` missing, while paired Procrustes gives `23.55%` missing with 100% support fit. N256 retains `93.37%` of seed0 N1000 missing accuracy and seeds1/2 retain `82.77/85.64%` of the correspondence gain. Thus semantic transport is not a seed0 or early-round artifact.

### H05-A/B — simple relation coordinates carry signal, but the previous optimization diagnosis was incomplete

The fixed relation

`r_i(z) = (z - mu_i) (A_i - mu_i)^T  in R^256`

uses the exact H04 N256 paired anchors. Raw H05-A gives paired/broken missing `11.025/4.8125%` with inadequate support fit `65.90/76.65%`. H05-B applies only a support-derived shared diagonal affine z-score, which preserves the affine-linear hypothesis class. It gives paired/broken missing `12.0125/4.9250%`, support fit `68.65/76.45%`, and still hits the 2000-iteration cap with `grad_inf=2.23e-3/1.47e-3`. Therefore the representation has not been fairly adjudicated yet.

H05-B implementation/fairness is accepted: 30 tests pass; the conditioned-to-raw head mapping is correct; statistics use support only and no labels; exact anchor/support/permutation provenance is reused; all ten H02-A records reproduce; state, RNG, module modes, and pre-existing gradients remain unchanged. The positive paired-vs-broken gap remains suggestive but below the strong gate and cannot be interpreted while fitting is unresolved.

### Important structural correction from H05-B

The reported H05-B `rank=256` and condition numbers around `1e8` must **not** be interpreted as 256 genuine relation directions. By construction, for centered anchors `C=A-mu`,

`C^T 1 = 0`,

so every exact relation vector satisfies

`r 1 = (z-mu) C^T 1 = 0`.

Therefore the mathematical relation space is contained in the 255-D zero-sum subspace and has rank at most 255. The H05-B diagnostic used a float64-epsilon SVD tolerance on a matrix whose relation entries were computed in float32; the tiny float32 violation of the exact null constraint was consequently counted as a nonzero 256th singular direction. The very small reported minimum singular value is therefore at least partly a known numerical artifact, not evidence for a meaningful 256th semantic direction.

This does **not** invalidate the H05-B accuracies. It means the fastest next falsification is to quotient out the known redundant direction without adding representational capacity, rather than changing kernels, solvers, or architectures.

---

## CHATGPT REVIEW 30 — H05-B execution accepted; remove the exact structural null before judging relation transport

Reviewed commits `c8c5c2ec405f4134a709f45df63c7cdf854c6d9a` and `3b7e9c2ef78ce2127c345089dbeb0402c40da119`, the complete diff from prior lead commit `72042fc0a8c032a57501105e49955f178d0578d2`, `pprtp/conditioning.py`, `pprtp/relation.py`, `pprtp/fedgh.py`, `tests/test_conditioning.py`, `research_log/H05B/gate/RESULTS.md`, `verification.json`, and the latest CODEX report.

No leakage or unfair branch asymmetry was found. H05-B did exactly the requested invertible support-only conditioning, and both arms remain matched. The new diagnosis is mathematical: because relation coordinates sum to zero exactly in real arithmetic, the epsilon64-on-float32 rank diagnostic overstates rank and makes the quoted `~1e8` condition number partly dominated by a known null direction. Do not spend another block merely increasing LBFGS iterations and do not introduce PCA/whitening/MLP/kernel tricks. First remove only this exact redundant coordinate by a fixed orthonormal basis of the zero-sum subspace.

---

# ACTIVE — H05-C: Structural-null-free relational coordinate audit

## One scientific objective

Test the **same centered-inner-product relation representation and the same affine-linear classifier class** after removing its one mathematically exact redundant direction. Determine whether H05-A/B's poor fitting is primarily caused by representing a 255-D zero-sum relation space with 256 numerically redundant float32 coordinates.

This is a reparameterization / numerical audit, not a new method.

## Frozen scope

Use `fedgh`, seed0, round10 only and reproduce all ten committed H02-A online records exactly.

Reuse verbatim:

- H02-E owner support hash `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`;
- H03-A parent anchor hash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`;
- exact H04/H05 N256 prefix;
- H05-A broken permutations (`314159+i`, client0 identity);
- raw relation formula and preprocessing.

No resampling, labels in transforms, test fitting, online-training changes, new anchor counts, or seeds1/2.

## Fixed 255-D zero-sum basis

Construct a deterministic **data-independent** orthonormal matrix

`Q in R^{256 x 255}`

whose columns span the subspace orthogonal to the all-ones vector. Prefer a closed-form Helmert basis so the result does not depend on QR/SVD sign conventions or data.

Required numerical identities in float64 unit tests:

- `Q^T Q = I`;
- `Q^T 1 = 0`;
- for arbitrary exact zero-sum row vectors `r`, `r == (r Q) Q^T` to tight tolerance.

For every H05 relation vector define only

`u = r Q  in R^255`.

Do not learn or tune `Q`. Do not select coordinates by labels or singular values.

### Why this does not add capacity

For exact relation vectors, `r 1=0`, so `u=rQ` is a bijective coordinate chart of the same relation subspace, with inverse `r=uQ^T`. Any 256-D affine-linear head restricted to this subspace has an exactly equivalent 255-D affine-linear head and vice versa. Add a mandatory double-precision unit test showing logits agree under the analytic weight mapping for arbitrary heads and exact zero-sum inputs.

## Quantify the removed component

Before projection, report for paired and broken pooled support relations:

- maximum and RMS absolute row sum;
- Frobenius energy in the normalized all-ones direction divided by total Frobenius energy;
- same quantities for official-test relations as a diagnostic only.

These numbers should establish whether the discarded direction is only float32 residue. Do not use test values to choose anything.

## Conditioning after the lossless chart

After converting support/test relations to 255-D `u`, apply the **same H05-B support-only population z-score rule**, now in 255-D, separately within the paired and broken arms:

`u' = (u - mean_support) / std_support`.

Require all std values finite and strictly positive; no epsilon/clamp/drop. This z-score is again an invertible affine reparameterization and does not enlarge the affine-linear hypothesis class.

Fit exactly two fresh heads:

1. `rel255_paired_helmert_zscore`
2. `rel255_broken_helmert_zscore`

with the same frozen solver:

- linear `255 -> 10`;
- zero initialization;
- full-batch LBFGS, `strong_wolfe`, lr=1;
- `max_iter=2000`;
- `tolerance_grad=1e-9`, `tolerance_change=1e-12`;
- no regularization;
- float32 training/input, matching H05-A/B.

Do not increase iterations or change dtype/optimizer in this block.

## Rank / conditioning diagnostics

For each arm report singular diagnostics for:

1. raw 256-D pooled support relation matrix;
2. 255-D Helmert coordinates;
3. 255-D Helmert + zscore coordinates.

Because the underlying matrix was computed in float32, report **both**:

- the old float64-epsilon tolerance for continuity with H05-B;
- an input-aware tolerance `max(shape) * eps_float32 * smax`.

Report rank, largest singular value, smallest singular value above each tolerance, and corresponding nonzero condition number. Do not call a float32-noise singular direction semantically meaningful merely because it exceeds an eps64 threshold.

## Fixed references and decision gate

Accepted seed0 round10 Procrustes N256 reference:

- missing `P = 21.9875%`;
- seen `37.55%`.

Historical H05-B conditioned relation values:

- paired missing `12.0125%`, support fit `68.65%`;
- broken missing `4.9250%`, support fit `76.45%`.

For H05-C paired missing `R255` and broken missing `S255`, report

`q_rel_255 = R255 / 21.9875`

`delta_rel_255 = R255 - S255`.

### Interpretation

First require all integrity checks and paired support fit `>=95%` before judging the representation.

- **Strong simple relation transport:** `q_rel_255 >= .70` and `delta_rel_255 >= 8pp`. Stop. The next lead block may finally test class-level relation-prototype compression.
- **Adequately fit but insufficient:** paired fit `>=95%` and `q_rel_255 < .40`. The raw centered-inner-product relation is too weak; stop and do not rescue it with another kernel/MLP in this block.
- **Adequately fit, intermediate:** all other paired-fit `>=95%` outcomes. Report and stop.
- **Still optimization-unresolved:** paired fit `<95%` with `grad_inf >1e-5`. Report the 255-D input-aware condition number and stop. Do not extend iterations. A later lead decision may justify a float64 solver-only audit if the structural null has been removed but float32 conditioning remains the clear bottleneck.
- **Stationary but underfit:** paired fit `<95%` with `grad_inf <=1e-5`. Treat as evidence of a shared-linear separability ceiling for this relation representation and stop.

Always report broken-arm fit/gradient; do not infer pairing necessity from an inadequately fit broken arm.

## Integrity requirements

Preserve all existing 30 tests and add only minimal tests for the Helmert basis, exact reconstruction/logit equivalence, and zero-sum diagnostic. Assert:

- exact online rounds1..10;
- exact anchors/support/permutations;
- identical frozen state entering both arms;
- no mutation of client/server parameters or buffers, prototypes, module modes, CPU/CUDA RNG, or pre-existing gradients;
- anchor labels unused; official test evaluation-only;
- all relations, basis values, conditioning statistics, singular values, losses, logits, parameters, and gradients finite.

## Deliverable

Append `CODEX REPORT H05-C` with STATUS, source SHA, commands/run IDs, tests, provenance, theoretical-null residual/energy diagnostics, 256D-vs-255D singular diagnostics under both tolerance rules, paired/broken fit diagnostics and test metrics, `q_rel_255`, `delta_rel_255`, integrity receipts, warnings, and artifact paths.

Do **not** implement class-relation prototypes, PCA/data-dependent SVD projection, whitening, cosine/RBF kernels, learned encoders, MLPs, OT, hybrid/gating heads, float64 training, more iterations, other anchor counts, or seeds1/2 in H05-C. Await research-lead review.

## CODEX REPORT H05-C — DONE (2026-09-18 03:30 +08)

STATUS: DONE; **still optimization-unresolved** under frozen decision rule. Source `8a6cae2978ade7ecc8f9bcdec9b9ec4db668a3da`;release `20260918-032713-h05c`;run `20260918-032735-h05c-helmert`,exit0. Exactly two seed0round10 Helmert+zscore arms,no solver/dtype/iteration changes.

Files: `pprtp/helmert.py`,optional input-aware spectrum in conditioning.py,optional structural chart in relation.py/run.py,tests/test_helmert.py,scripts/report_h05c.py.32 tests pass locally/remotely;previous30 preserved. Double tests verify Q.T@Q=I,Q.T@ones=0,zero-sum reconstruction,analytic logits equivalence in both directions (tight1e-12/1e-11 tolerance). A known-energy example verifies null residual/energy diagnostics.

Commands (AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h05c
./scripts/autodl-run.ps1 -Name h05c-helmert -Cmd 'PPRTP_SOURCE_SHA=8a6cae2978ade7ecc8f9bcdec9b9ec4db668a3da bash scripts/run_h01.sh --modes fedgh --seeds 0 --rounds 10 --helmert-probe'
D:\anaconda3\python.exe scripts/report_h05c.py research_log/H05C/gate
```

Q is closed-form,data-independent: column k-1 (k1..255) has firstk entries1/sqrt(k(k+1)),entryk=-k/sqrt(k(k+1)),remainingzero. Constructdouble,castfloat32. Project raw relation asr@Q,then use shared arm-specific supportonlypopulationzscore,strictpositive std,noepsilon/clamp. Broken uses exactly priorpermutation beforeprojection. No data-dependent rankselection or truncation;no added capacity on exact zero-sum space. Tests prove equivalence mathematically within double precision;actualfloat32 residual is quantified below.

Fresh zero-init255->10 heads,fullbatch LBFGS lr1,strong_wolfe,max_iter2000,tolerance_grad1e-9,tolerance_change1e-12,no regularization,float32 inputs/training. Initial CE2.30258393 andsupportaccuracy10% botharms. Support-derived means/stds and allhashes retained;test values never choose transforms.

Exact parentanchorhash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`;supporthash `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`;exact N256prefix. Bothrawsupporthashes exactly equalH05-B;historicalGrams/permutations/frozenstate exact. All10H02-A online client/prototype/server/ordinary metric records exact. Botharms identical startingstate and preserveparameters/buffers/prototypes,modulemodes,CPU/CUDA RNG,existinggradients. Anchorlabelsunused,testevaluationonly;allfeatures/basis/statistics/singularvalues/losses/logits/parameters/gradients finite.


| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rel255_paired_helmert_zscore | 56.199999 | 11.787500 | 20.670000 | 20.670000 | 68.700004 | 1.07979739 | 0.00125634123 | 0.0112194698 | 762.251526/0.915916264 | 2000/2114 |
| rel255_broken_helmert_zscore | 60.249999 | 4.887500 | 15.960000 | 15.960000 | 76.000005 | 0.853311598 | 0.00191304844 | 0.0230582841 | 819.577148/0.627656102 | 2000/2082 |

R255=11.787500%,S255=4.887500%,q_rel_255=0.536100057,delta_rel_255=6.900000pp. Frozen branch: still optimization-unresolved.

| Arm | Data | Max absolute row sum | RMS row sum | Ones-direction energy / total energy |
|---|---|---:|---:|---:|
| rel255_paired_helmert_zscore | support | 0.000535860658 | 9.92043353e-05 | 9.36946282e-16 |
| rel255_paired_helmert_zscore | test | 0.0019326508 | 0.000241749009 | 5.31122291e-15 |
| rel255_broken_helmert_zscore | support | 0.000535860658 | 9.92043353e-05 | 9.36946282e-16 |
| rel255_broken_helmert_zscore | test | 0.0019326508 | 0.000241749009 | 5.31122291e-15 |

| Arm | Stage | Epsilon rule | Rank | smax | Smallest above tolerance | Tolerance | Nonzero condition |
|---|---|---|---:|---:|---:|---:|---:|
| rel255_paired_helmert_zscore | raw256 | float64 | 256 | 8651.9041 | 7.56336555e-05 | 3.84221726e-09 | 114392251 |
| rel255_paired_helmert_zscore | raw256 | float32 | 72 | 8651.9041 | 2.08086298 | 2.06277468 | 4157.84422 |
| rel255_paired_helmert_zscore | Helmert255 | float64 | 255 | 8651.90412 | 0.221877624 | 3.84221726e-09 | 38994.0362 |
| rel255_paired_helmert_zscore | Helmert255 | float32 | 72 | 8651.90412 | 2.08086325 | 2.06277469 | 4157.84368 |
| rel255_paired_helmert_zscore | Helmert255+zscore | float64 | 255 | 617.183665 | 0.0107741077 | 2.74084606e-10 | 57283.9705 |
| rel255_paired_helmert_zscore | Helmert255+zscore | float32 | 109 | 617.183665 | 0.147470375 | 0.147148052 | 4185.13661 |
| rel255_broken_helmert_zscore | raw256 | float64 | 256 | 4421.63347 | 7.60112406e-05 | 1.96359971e-09 | 58170784.1 |
| rel255_broken_helmert_zscore | raw256 | float32 | 220 | 4421.63347 | 1.0663845 | 1.05419957 | 4146.37823 |
| rel255_broken_helmert_zscore | Helmert255 | float64 | 255 | 4421.63347 | 0.648275172 | 1.96359972e-09 | 6820.61209 |
| rel255_broken_helmert_zscore | Helmert255 | float32 | 220 | 4421.63347 | 1.066386 | 1.05419957 | 4146.3724 |
| rel255_broken_helmert_zscore | Helmert255+zscore | float64 | 255 | 325.768851 | 0.0475532008 | 1.44670432e-10 | 6850.61878 |
| rel255_broken_helmert_zscore | Helmert255+zscore | float32 | 226 | 325.768851 | 0.0778047342 | 0.0776693465 | 4187.00551 |

Tolerance=max(matrix_shape)*eps(dtype)*smax;SVD in double on actual float32 input. No data-dependent projection/truncation. Fixed Helmert basis built by closed form in double and cast to float32;training remains float32.

rel255_paired_helmert_zscore: basis SHA256=b0cf8c0a162fd645fd1f44dd476ac578ab783488a83433efc79cd85132caeefd;support-only conditioning SHA256=23cf357c139b4b334e0183756967808b1c8a832209e3fe3069437003a4f3bd2a.

rel255_broken_helmert_zscore: basis SHA256=b0cf8c0a162fd645fd1f44dd476ac578ab783488a83433efc79cd85132caeefd;support-only conditioning SHA256=5be5a326738c13976d48066cb11fdd728722eaf9f267d7d17d7e02ea6bdf85b9.

Both2000iterationcaps reached;paired/broken supportfit68.70/76.00%,grad_inf.00125634/.00191305. Neither95%fit threshold reached;pairedgradient>1e-5 means optimization unresolved,not stationary underfit. Accepted referenceP21.9875%,seen37.55% unchanged. Allperclientfit/test counts,headnorms/hashes,mean/stdvectors saved.

Interpretation: removed all-ones energy is only9.37e-16 of support energy and5.31e-15 of test energy,consistent with float32 residue in the exact structural null. Removing it changes the eps64-conditioned spectrum drastically (pairedraw~1.14e8 toHelmert~3.90e4) yet leaves supportfit farbelow95%,so this single redundant direction is not sufficient to explain or repair the observed fitting problem. Finalpairedzscore input-aware rank109/255 andcondition4185.14 summarize only singularvalues above the prescribed eps32threshold;the full eps64nonzero condition remains57283.97,so thresholded condition is not a complete conditioning certificate. Positivepairedmissing11.7875% vsbroken4.8875% remains suggestive but cannot close the representation gate while fitting is unresolved. Stop without float64training,moreiterations,whitening or classprototypecompression;await lead.

Known NVMLwarnings retained verbatim in train.log;no SVDwarning/failure or nonfinitevalues thisrun. Evidence `research_log/H05C/gate/` RESULTS.md,verification.json,rawnull/spectral/fit/provenance/state receipts,meta/run/log/tests. Remoteoriginals/checkpoints `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-032735-h05c-helmert`.
