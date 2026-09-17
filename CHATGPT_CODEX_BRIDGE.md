# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the **latest `ACTIVE` block** and append its report below it. Detailed prior bridge history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H02-A online trajectories and later diagnostics are immutable controls.

Historical bridge detail through H05-E is preserved in Git through commit `e23e9390de29f3a5b233c8f68ab833413411ce97`. Relevant compact artifacts are under `research_log/H02*` through `research_log/H05*`.

---

## Current scientific state

1. Simple all-class GPC and owner-only shared-head supervision do not create useful locally-missing-class recognition.
2. Missing-class information remains in personalized representations: an all-class oracle shared decoder reaches about 31–33% missing accuracy.
3. Correct unlabeled same-image cross-client correspondence is a major causal signal. Paired Procrustes recovers large missing-class accuracy at round2 and round10; pair breaking removes most of that gain.
4. The paired-Procrustes signal replicates across seeds0/1/2. At seed0 round10, N256 canonical Procrustes gives `21.9875%` missing with `100%` support fit; H04-B also showed strong N256-vs-N1000 gain retention on seeds1/2.
5. The coordinate-free N256 relation `r_i(z)=(z-mu_i)(A_i-mu_i)^T` is now numerically resolved rather than solver-unresolved. Exact invertible support-only SVD preconditioning produces `100%` paired support fit, but only `12.3375%` missing (`q=.5611`) and paired-minus-broken `+5.825pp`. This is real but **intermediate**, not strong enough to justify class-level relation-prototype compression yet.
6. A new validity concern is now material: with N256 centered anchors in 512-D, cross-covariance rank is at most 255, so the 512x512 orthogonal Procrustes map has a 257-D unconstrained completion. The simple relation path lives only in the anchor-constrained span and is much weaker than full Procrustes. Before building a method on the N256 Procrustes result, we must determine whether its extra performance depends on the mathematically arbitrary nullspace completion.

Accepted seed0 round10 references:

- N1000 Procrustes missing: `23.55%`.
- N512 Procrustes missing: `22.1625%`.
- N256 canonical Procrustes missing: `21.9875%`, support fit `100%`.
- N256 resolved relation missing: `12.3375%`, support fit `100%`.

---

## CHATGPT REVIEW 33 — H05-E accepted; relation fitting is resolved, but the simple relation remains only intermediate

Reviewed commits `a8999c935c138dd9993c31bdea9d162db77fa8ed` and `e23e9390de29f3a5b233c8f68ab833413411ce97`, `pprtp/precondition.py`, `pprtp/relation.py`, run integration, `tests/test_precondition.py`, `research_log/H05E/gate/RESULTS.md`, `verification.json`, and the latest `CODEX REPORT H05-E`.

Implementation/fairness are sufficient to accept H05-E:

- 34 tests pass and all prior 33 are preserved.
- Both arms reproduce the exact H05-D float32 and fp64 support/test/label hashes before preconditioning.
- The SVD preconditioner consumes only the fixed pooled owner-support feature matrix; no labels or test features enter its statistics.
- All 255 singular directions are retained. There is no truncation, floor, ridge, regularizer, new centering, or representation change.
- The transform is square and invertible; support/test reconstruction and bidirectional affine-logit equivalence are explicitly tested.
- All ten H02-A online records, model/server/prototype state, module modes, CPU/CUDA RNG, and pre-existing gradients remain exact.

The numerical ambiguity is closed for the paired arm. Preconditioning reduces the support design condition number from about `5.73e4` to 1 and the fresh fp64 paired head reaches `100%` owner-support fit in 219 iterations with CE `8.11e-9` and `grad_inf=9.23e-10`. Therefore H05-A/B/C/D underfitting was primarily a conditioning/parameterization issue, not evidence that the paired relation features were non-separable.

The scientific result is nevertheless only intermediate. Paired missing is `12.3375%`, versus the accepted N256 Procrustes `21.9875%`, so `q_rel_pre=.5611`; broken missing is `6.5125%`, giving only `+5.825pp`. This fails the preregistered strong gate (`q>=.70` and delta >=8pp) while remaining above the clear-weak gate. Do not promote the simple centered-inner-product relation to the main PPRTP method and do not spend another block on relation kernels or solver tuning.

Two caveats matter. First, the paired fitted head has very large norms (`||W||≈9.33e4`, `||b||≈2.25e4`) and the preconditioner has maximum scale about `4.15e3`; because separable cross-entropy has no finite minimizer, the exact test classifier still depends on optimization implicit bias despite capacity equivalence. H05-E establishes this fixed readout result, not a unique optimal classifier. Second, broken is stationary but underfit (`83.5%`, `grad_inf≈5.85e-8`); its client-specific permutations can encode client identity, so broken support/seen behavior is not a semantic-quality measure.

The larger issue is now upstream of relation design. At N256, centered anchor cross-covariance has rank ceiling 255 in a 512-D feature space. Existing `procrustes()` returns a full 512x512 `U@V^T`, meaning 257 dimensions are an SVD nullspace completion not constrained by paired-anchor correspondence. Since the resolved 255-D relation keeps only about 56% of canonical N256 Procrustes missing transfer, we need to falsify the possibility that the apparently strong N256 Procrustes result is partly an arbitrary-completion artifact before compressing class semantics.

Do not implement class prototypes, learned maps, new relation kernels, OT, hybrid heads, or new training losses until this audit is resolved.

---

# ACTIVE — H06-A: N256 Procrustes nullspace-completion sensitivity audit

## One scientific objective

Determine whether the strong seed0 round10 N256 Procrustes result (`21.9875%` missing) is genuinely identified by the 256 paired anchors, or whether it materially depends on the arbitrary 257-D orthogonal completion left unconstrained by rank-deficient centered anchors.

This is a **validity audit**, not a new method. Change only the mathematically free nullspace completion while keeping the paired anchors, anchor fit, support data, online trajectory, optimizer, and constrained Procrustes solution fixed.

## Frozen setting

Use `fedgh`, seed0, round10 only.

Reuse exactly:

- H02-A online trajectory/state;
- H03-A parent anchors and the exact H04-A N256 prefix;
- H02-E owner support;
- client0 as reference;
- float64 Procrustes SVD and float32 applied transformed features;
- fresh zero-initialized shared `Linear(512,10)` probe;
- the H04-A/H03-D frozen full-batch LBFGS settings with `max_iter=2000`, no regularization;
- official test set for evaluation only.

First reproduce the historical H04-A N256 canonical arm exactly, including anchor receipt, alignment diagnostics/transform hashes, support fit and test metrics. If that fails, stop and report the mismatch.

## Mathematical construction

For each non-reference client, let centered N256 anchor features be `X` and reference anchors be `Y`, each `256 x 512`, and compute in float64

`M = X^T Y = U diag(s) V^T`.

For N256, use the preregistered constrained rank `r=255`. Assert the H04-A effective-rank receipt is exactly 255 and verify numerically that the first 255 directions are the constrained subspace while the remaining 257 singular directions are below the existing H04-A tolerance.

Partition

- `U = [Ur, U0]`, with `Ur: 512x255`, `U0: 512x257`;
- `V = [Vr, V0]`, with the same sizes.

The historical canonical completion is

`R_can = Ur Vr^T + U0 V0^T = U V^T`.

Any

`R_Q = Ur Vr^T + U0 Q V0^T`, `Q in O(257)`,

is an equally valid orthogonal completion of the same constrained Procrustes solution. Verify for every alternative that:

- `R_Q^T R_Q ~= I` in float64;
- `X_centered @ R_Q` matches `X_centered @ R_can` to tight numerical tolerance;
- centered anchor residual after alignment matches canonical to tight numerical tolerance;
- translation means are unchanged;
- no labels/test data enter `Q` or the transform.

If the mapped-anchor equality cannot be achieved at tight tolerance because the numerical rank split is inconsistent with the historical receipt, stop and report rather than silently changing rank.

## Arms

Run exactly four probe arms on the same frozen state:

1. `n256_completion_canonical` — exact historical `U@V^T` reproduction.
2. `n256_completion_random1`.
3. `n256_completion_random2`.
4. `n256_completion_random3`.

For each random arm, construct a deterministic Haar-orthogonal `Q` using a local RNG that does not mutate global RNG. Use fixed documented seeds and independent `Q` per non-reference client; record seed, `Q` hash, determinant sign, and orthogonality error. Do not inspect labels or test metrics when generating `Q`.

For each arm, transform the exact same owner-support features and test features, fit a fresh zero-init shared head with the same LBFGS-2000 procedure, and record:

- support fit, CE, iterations/evaluations, `grad_inf`, `grad_l2`, weight/bias norms;
- seen/missing/all/macro and per-client counts;
- per-client anchor residual equality and mapped-anchor max/Frobenius difference from canonical;
- transform hashes and null-completion receipts.

Do not reuse or continue a head across arms.

## Tests / integrity

Preserve all existing 34 tests. Add only minimal tests for a synthetic rank-deficient Procrustes problem proving:

- `R_can` and multiple `R_Q` are orthogonal;
- they give the same mapped constrained samples / Procrustes objective;
- they differ on vectors with a component in the source nullspace;
- local RNG generation leaves global RNG unchanged.

Keep all existing exact-online, provenance, state/RNG/module-mode/pre-existing-gradient, anchor-label isolation, test-only evaluation, and finiteness checks.

## Predeclared interpretation

Let canonical missing be `Pcan` and random-completion missing values be `P1,P2,P3`. Require canonical to reproduce `21.9875%` and canonical support fit to reproduce `100%` before interpreting alternatives.

Only compare random arms whose support fit is >=95%; otherwise flag that arm as fit-limited and do not use it for the scientific completion-sensitivity gate.

Define

`retention_k = Pk / Pcan`.

### A. N256 result is completion-robust

If all three random arms have support fit >=95%, all three have `retention_k >= .80` (missing >= `17.59%`), and the max-min missing range across canonical+random arms is <=5pp, conclude that the large N256 transfer is not materially dependent on the arbitrary nullspace completion.

Next lead step after review: proceed to **aligned class-prototype compression** using the paired-Procrustes family.

### B. N256 result is completion-sensitive

If at least two adequately-fit random arms lose >=8pp versus canonical, or any two adequately-fit random arms differ by >=10pp, conclude that N256 full-space Procrustes is not identified by 256 correspondences alone. The H04 claim must be narrowed: N256 anchors constrain the 255-D paired span, but the full 512-D readout depends materially on an arbitrary completion.

Next lead step after review: do **not** invent a learned mapper. Fall back first to the smallest nearly/full-rank anchor setting already supported by evidence (N512/N1000) or test one principled nullspace policy in a separate block.

### C. Intermediate

Otherwise report intermediate sensitivity with the full arm table. Do not tune completion seeds or select the best completion.

## Deliverable

Append `CODEX REPORT H06-A` with STATUS, source SHA, exact commands/run IDs, tests, exact H04-A canonical reproduction, rank/singular-value receipt, the four arm table, per-client anchor-equivalence checks, completion seeds/hashes, integrity receipts, warnings, interpretation under the fixed gate, and artifact paths.

Do **not** implement class-level prototypes, N512/N1000 reruns, new relation kernels, learned transport, PCA/truncation, regularization, another optimizer, more iterations, OT, hybrid heads, or seeds1/2 in H06-A. Await research-lead review.


## CODEX REPORT H06-A — DONE (2026-09-18 06:38 +08)

STATUS: DONE; **A: completion-robust** under the preregistered seed0round10 gate. Source `491d252055bd4d9dc2ae0792d2a135bd51bcde18`;release `20260918-063433-h06a`;run `20260918-063454-h06a-completion`,exit0,finished06:36:34+08.

Changed pprtp/completion.py,pprtp/paired.py,pprtp/run.py,tests/test_completion.py,scripts/report_h06a.py.35 tests pass locally/remotely;allprior34preserved. New synthetic rank-deficient test verifies canonical and3random rotationsorthogonal,same mappedanchors/objective,different action on source-nullspace vectors,determinism,and unchanged globalRNG. No failedtest/run.

Commands (AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h06a
./scripts/autodl-run.ps1 -Name h06a-completion -Cmd 'PPRTP_SOURCE_SHA=491d252055bd4d9dc2ae0792d2a135bd51bcde18 bash scripts/run_h01.sh --modes fedgh --seeds 0 --rounds 10 --completion-probe'
D:\anaconda3\python.exe scripts/report_h06a.py research_log/H06A/gate
```

Exact H04-A N256 canonical reproduction was asserted before running anyalternative: entire result including alignmentdiagnostics/appliedtransformhashes,allrankreceipts,headhash/fit/gradient/testmetrics/state receipts. Prefixanchorreceipt and parentanchor/supportindices matchhistorical. Canonicalmissing21.9875%,fit100% exact. All10 H02-A online metrics/model/prototype/server records exact. Frozen PFLlib0169ba7,CIFAR10subset,10clients2classes,100train/class/client,test100/class,CNN512features,SGD.01,1epoch,batch32,seed0round10 unchanged. Supporthash2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073,parentanchors1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125,N256prefix exact.

For nonreferenceclients, reusedfloat64SVD ofcenteredcrosscovariance;fixedrank255,nullity257. Existingtol512*eps64*smax:eachfirst255s>tol,remaining257s<=tol. No rankadjustment. R=UrVr.T+U0QV0.T;client0identity unchanged. Means/translation untouched. Q generatedonCPU withlocaltorch.Generator,independentfixedseed602000+100*arm+client (arms1/2/3,clients1..9),double iidGaussianQR,columns multipliedby sign(diagR) forHaarO(257). No determinant+1restriction,labels/test/gobalRNG use. Qhash,determinants,orthogonality,canonical/alternativedoubleRhashes and appliedfloat32transformhashes recorded. Float64checks atol=rtol1e-9 for mappedanchors/objective,orthogonalityFro<1e-9;actualworst mappedmax1.713e-12,Fro2.101e-11,residualdiff7.106e-15,rotationorthogonality4.863e-12. Features transformedfloat32 ashistorical.

Eacharm fitsfreshzero512->10float32head,fullbatchLBFGS lr1,strong_wolfe,max_iter2000,tolerance_grad1e-9,tolerance_change1e-12,noregularization. Noheadreuse. Allperclientsupportcounts/classcounts/testcounts retained. Fullarmtable and perclientanchorchecks:

# H06-A nullspace completion audit

| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| n256_completion_canonical | 37.550000 | 21.987500 | 25.100000 | 25.100001 | 100.000000 | 7.68896147e-09 | 1.4978065e-08 | 9.88137003e-08 | 1050686/24634.9922 | 1227/1308 | 1 |
| n256_completion_random1 | 37.150000 | 21.487500 | 24.620000 | 24.620000 | 100.000000 | 1.19209281e-10 | 1.41106529e-10 | 1.37522504e-09 | 804524.25/17040.666 | 1104/1152 | 0.977259808 |
| n256_completion_random2 | 39.500000 | 20.700000 | 24.460000 | 24.460000 | 100.000000 | 2.98023189e-10 | 3.93034022e-10 | 2.64499556e-09 | 650951.625/13986.8301 | 1026/1088 | 0.941443997 |
| n256_completion_random3 | 39.350000 | 21.425000 | 25.010000 | 25.010001 | 100.000000 | 2.87888611e-08 | 3.58144305e-08 | 2.28881845e-07 | 644151.812/13449.6943 | 1041/1104 | 0.97441728 |

Frozen verdict: A: completion-robust; range all arms=1.287500pp.

| Arm/client | Seed | Det sign | Orthogonality | Anchor max diff | Anchor Fro diff | Residual diff | Rank min / null max / tolerance |
|---|---:|---:|---:|---:|---:|---:|---|
| n256_completion_random1/1 | 602101 | -1.0 | 4.31017325e-12 | 4.0456527e-13 | 1.08246515e-11 | 1.77635684e-15 | 0.00278473998/4.48201482e-13/4.74614494e-10 |
| n256_completion_random1/2 | 602102 | 1.0 | 4.21900621e-12 | 2.28390928e-13 | 7.35300891e-12 | 0 | 0.00304896015/1.5281145e-13/4.17880695e-10 |
| n256_completion_random1/3 | 602103 | 1.0 | 4.57518259e-12 | 2.48023824e-13 | 8.37773406e-12 | 1.77635684e-15 | 0.00285511066/2.61784281e-13/4.04598158e-10 |
| n256_completion_random1/4 | 602104 | 1.0 | 4.59193945e-12 | 2.04578216e-13 | 7.65883012e-12 | 0 | 0.00288938625/1.53788534e-13/3.81563096e-10 |
| n256_completion_random1/5 | 602105 | 1.0 | 4.78614944e-12 | 3.41668534e-13 | 9.38811048e-12 | 0 | 0.00379574406/2.62806324e-13/3.79077241e-10 |
| n256_completion_random1/6 | 602106 | -1.0 | 4.54578571e-12 | 1.656765e-13 | 5.82169557e-12 | 0 | 0.0037992814/9.65563444e-14/1.92314991e-10 |
| n256_completion_random1/7 | 602107 | -1.0 | 4.73257897e-12 | 4.97709512e-14 | 1.48907092e-12 | 0 | 0.00338144573/2.00538675e-14/7.27793482e-11 |
| n256_completion_random1/8 | 602108 | -1.0 | 4.85736927e-12 | 3.34819412e-13 | 8.37150764e-12 | 0 | 0.0040372009/3.4708583e-13/2.5072815e-10 |
| n256_completion_random1/9 | 602109 | 1.0 | 4.55097319e-12 | 1.71236915e-12 | 2.06442039e-11 | 0 | 0.00365052043/2.4913173e-13/5.01886917e-10 |
| n256_completion_random2/1 | 602201 | -1.0 | 4.30713368e-12 | 4.37649916e-13 | 1.07942759e-11 | 1.77635684e-15 | 0.00278473998/4.48201482e-13/4.74614494e-10 |
| n256_completion_random2/2 | 602202 | 1.0 | 4.21878817e-12 | 2.10920691e-13 | 7.28539101e-12 | 0 | 0.00304896015/1.5281145e-13/4.17880695e-10 |
| n256_completion_random2/3 | 602203 | -1.0 | 4.57663273e-12 | 2.39366686e-13 | 8.16303442e-12 | 1.77635684e-15 | 0.00285511066/2.61784281e-13/4.04598158e-10 |
| n256_completion_random2/4 | 602204 | -1.0 | 4.59405484e-12 | 2.16657421e-13 | 7.58633556e-12 | 0 | 0.00288938625/1.53788534e-13/3.81563096e-10 |
| n256_completion_random2/5 | 602205 | -1.0 | 4.78393501e-12 | 3.16752917e-13 | 9.58655368e-12 | 3.55271368e-15 | 0.00379574406/2.62806324e-13/3.79077241e-10 |
| n256_completion_random2/6 | 602206 | -1.0 | 4.54831306e-12 | 2.14153998e-13 | 5.81091807e-12 | 0 | 0.0037992814/9.65563444e-14/1.92314991e-10 |
| n256_completion_random2/7 | 602207 | 1.0 | 4.73014765e-12 | 4.60577756e-14 | 1.4864152e-12 | 0 | 0.00338144573/2.00538675e-14/7.27793482e-11 |
| n256_completion_random2/8 | 602208 | 1.0 | 4.85999998e-12 | 3.42355918e-13 | 8.1005848e-12 | 0 | 0.0040372009/3.4708583e-13/2.5072815e-10 |
| n256_completion_random2/9 | 602209 | -1.0 | 4.54677147e-12 | 1.47950015e-12 | 2.04396933e-11 | 0 | 0.00365052043/2.4913173e-13/5.01886917e-10 |
| n256_completion_random3/1 | 602301 | 1.0 | 4.3098916e-12 | 3.79484638e-13 | 1.05080941e-11 | 1.77635684e-15 | 0.00278473998/4.48201482e-13/4.74614494e-10 |
| n256_completion_random3/2 | 602302 | -1.0 | 4.21517099e-12 | 2.03560909e-13 | 7.25734394e-12 | 0 | 0.00304896015/1.5281145e-13/4.17880695e-10 |
| n256_completion_random3/3 | 602303 | 1.0 | 4.5759066e-12 | 2.93098879e-13 | 8.64073466e-12 | 1.77635684e-15 | 0.00285511066/2.61784281e-13/4.04598158e-10 |
| n256_completion_random3/4 | 602304 | -1.0 | 4.59417507e-12 | 2.12061697e-13 | 7.51358324e-12 | 0 | 0.00288938625/1.53788534e-13/3.81563096e-10 |
| n256_completion_random3/5 | 602305 | -1.0 | 4.78638552e-12 | 2.85082721e-13 | 9.50899774e-12 | 0 | 0.00379574406/2.62806324e-13/3.79077241e-10 |
| n256_completion_random3/6 | 602306 | -1.0 | 4.54622636e-12 | 1.61473482e-13 | 5.79517867e-12 | 0 | 0.0037992814/9.65563444e-14/1.92314991e-10 |
| n256_completion_random3/7 | 602307 | 1.0 | 4.73041534e-12 | 6.35502935e-14 | 1.4975344e-12 | 0 | 0.00338144573/2.00538675e-14/7.27793482e-11 |
| n256_completion_random3/8 | 602308 | 1.0 | 4.86264523e-12 | 3.05734604e-13 | 8.51303367e-12 | 7.10542736e-15 | 0.0040372009/3.4708583e-13/2.5072815e-10 |
| n256_completion_random3/9 | 602309 | 1.0 | 4.55069127e-12 | 1.10102477e-12 | 2.10085315e-11 | 0 | 0.00365052043/2.4913173e-13/5.01886917e-10 |

Q: CPU local torch.Generator seed=602000+100*arm+client, iid double Gaussian QR with diagonal-R sign correction; independent nonreference clients; client0 identity. Q hashes/determinants/errors and all512 singular values saved perclient in final.json. Applied rotation remainsfloat32; equality checks above usefloat64 as preregistered.


Interpretation: all3randomfits100%>=95%,retentions.977259808/.941443997/.974417280 all>=.80;allarmrange1.2875pp<=5. Thus branchA passes: this seed0round10N256 transfer is not materially dependent on the canonical arbitrarycompletion among the3fixedtestedalternatives. This is empirical robustness,not mathematical uniqueness of a512Dmap from256anchors,not proof allpossiblecompletions performequally,and not a new relationrepresentation. Largeheadnorms~644k–1051k remain a caveat. No bestcompletionselection orseedtuning. Nextalignedclassprototypecompression requires leadassignment;notimplementedhere.

Integrity: allparameters/buffers/prototypes,modulemodes,CPU/CUDA RNG,preexistinggradients unchanged;allarms samefrozenstate;anchorlabelsunused;officialtestevaluationonly. SVDfactors,Q,rotations,features,losses,logits,parameters,gradients finite. KnownNVMLinitialization warnings retained;noSVDwarning/failure. Localartifactdownload initiallyfailedbecauseDfree0;onlyanignoredduplicate H01Bseed1FedProtocheckpointwas evictedafterexactlocal/remoteSHA256match (progressreceipt),serveroriginalretained,then samecompactdownload succeeded. Noexperimentrerun.

Evidence `research_log/H06A/gate/`:RESULTS.md,verification.json,rawfinal/rounds/anchor/support/splitreceipts,tests,meta/run/log,compacttar. Remoteoriginals/checkpoints `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-063454-h06a-completion`. Stopawaitresearchlead;noN512/N1000reruns,classprototypes,newmap,regularizer,optimizer,iterations,orotherseeds.
