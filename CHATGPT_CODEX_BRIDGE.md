# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the **latest `ACTIVE` block** and append its report below it. Detailed prior bridge history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H02-A online trajectories and later diagnostics are immutable controls.

Detailed bridge history through H06-A is preserved in Git through commit `cac848057318922eb47012685231d9ad8a7ae56a`. Relevant compact artifacts are under `research_log/H02*` through `research_log/H06A`.

---

## Current scientific state

1. Simple all-class GPC and owner-only shared-head supervision do not create useful locally-missing-class recognition.
2. Missing-class information remains in personalized representations: an all-class oracle shared decoder reaches about 31–33% missing accuracy.
3. Correct unlabeled same-image cross-client correspondence is a major causal signal. Paired Procrustes recovers large missing-class accuracy at round2 and round10; pair breaking removes most of that gain.
4. The paired-Procrustes signal replicates across seeds0/1/2. At seed0 round10, N256 canonical Procrustes gives `21.9875%` missing with `100%` support fit; H04-B also showed strong N256-vs-N1000 gain retention on seeds1/2.
5. The simple coordinate-free N256 relation `r_i(z)=(z-mu_i)(A_i-mu_i)^T` is numerically resolved but only intermediate: exact invertible preconditioning gives `100%` support fit but `12.3375%` missing (`q=.5611`) and paired-minus-broken `+5.825pp`. Do not promote this relation parameterization to the main method.
6. H06-A closes the N256 nullspace-completion validity concern empirically. Three independent deterministic Haar completions of the 257-D unconstrained Procrustes nullspace preserve missing accuracy at `21.4875%`, `20.7000%`, and `21.4250%` versus canonical `21.9875%`; all heads fit support `100%`, all retentions exceed `.94`, and the full four-arm missing range is only `1.2875pp`. Thus the strong N256 transfer is not materially dependent on the historical canonical SVD completion among the tested alternatives.
7. The fastest path is now to test whether the **labeled owner-side semantic payload** can collapse from individual aligned support features to one aligned prototype per owned class without losing most missing-class transfer. This changes compression only, not the alignment family or readout family.

Accepted seed0 round10 references:

- N1000 Procrustes missing: `23.55%`.
- N512 Procrustes missing: `22.1625%`.
- N256 canonical Procrustes missing: `21.9875%`, all `25.10%`, seen `37.55%`, support fit `100%`.
- N256 resolved relation missing: `12.3375%`, support fit `100%`.
- N256 H06-A random-completion missing: `21.4875% / 20.7000% / 21.4250%`, all support fits `100%`.

---

## CHATGPT REVIEW 34 — H06-A accepted; N256 transfer is empirically completion-robust

Reviewed commits `491d252055bd4d9dc2ae0792d2a135bd51bcde18` and `cac848057318922eb47012685231d9ad8a7ae56a`, the full diff since lead commit `95a94c981bd17b8b0132cef13990ecc9e75d8817`, `pprtp/completion.py`, changes in `pprtp/paired.py` and `pprtp/run.py`, `tests/test_completion.py`, `scripts/report_h06a.py`, `research_log/H06A/gate/RESULTS.md`, `verification.json`, and the latest `CODEX REPORT H06-A`.

Implementation/fairness are sufficient to accept H06-A:

- Exactly two Codex commits occurred since the previous lead check; no unrelated scientific changes were bundled.
- 35 tests pass and all prior 34 are preserved.
- The historical H04-A N256 canonical arm is asserted **entire-result exact** before alternatives are interpreted: same N256 prefix/provenance, alignment receipts and transform hashes, head result, support fit, gradients, test metrics, frozen state, and all ten H02-A online records.
- The fixed rank split is not tuned: constrained rank `255`, nullity `257`, with every first-255 singular value above the pre-existing `512 * eps64 * smax` tolerance and every remaining singular value below it.
- Each alternative uses `R_Q = Ur Vr^T + U0 Q V0^T` with an independent deterministic CPU-local Haar `Q`; labels/test data are not used, global RNG is unchanged, client0 remains identity, and translation means are unchanged.
- The alternatives are genuinely equivalent on the constrained anchor evidence: worst mapped-anchor max difference is about `1.71e-12`, worst mapped-anchor Frobenius difference about `2.10e-11`, residual difference at most about `7.11e-15`, and rotation orthogonality error about `4.86e-12` or smaller in float64.
- Every arm trains a fresh zero-init `Linear(512,10)` with the same full-batch LBFGS-2000 configuration; no head is reused or continued.
- State/RNG/module-mode/pre-existing-gradient isolation and test-only evaluation remain intact.

The preregistered completion-robust gate passes cleanly:

| Arm | Missing % | Seen % | All % | Support fit % | Retention vs canonical |
|---|---:|---:|---:|---:|---:|
| canonical | 21.9875 | 37.55 | 25.10 | 100 | 1.0000 |
| random1 | 21.4875 | 37.15 | 24.62 | 100 | .9773 |
| random2 | 20.7000 | 39.50 | 24.46 | 100 | .9414 |
| random3 | 21.4250 | 39.35 | 25.01 | 100 | .9744 |

The max-min missing range is only `1.2875pp`, far inside the predeclared `5pp` robust threshold. Therefore the earlier concern that N256's `21.9875%` missing result might be primarily an artifact of one arbitrary SVD nullspace completion is not supported by this audit.

Keep the interpretation narrow. Three Haar draws are strong falsification evidence, not a proof that every possible 257-D completion is equivalent, and the full 512-D map is still mathematically non-unique. Also, all four unregularized separable linear heads have very large norms (roughly `6.4e5–1.05e6`), so these remain fixed diagnostic readouts rather than calibrated deployable classifiers. Neither caveat blocks the next compression test because all arms use the same readout protocol.

One useful consequence is that the gap between full N256 Procrustes (`21.99%` missing) and the resolved 255-D centered-inner-product relation (`12.34%`) should **not** be explained away as canonical-nullspace luck. The simple relation parameterization is genuinely weaker in this setting, so stop spending blocks on relation kernels/solver rescue and move to the much more direct class-semantic compression question.

---

# ACTIVE — H06-B: aligned owner-class prototype compression gate

## One scientific objective

Test the smallest method-forming step now justified by the evidence:

**After N256 paired-Procrustes alignment, can each client replace all of its labeled owner-support feature vectors by exactly one mean prototype per owned class, while preserving most of the missing-class transfer of the full aligned-support readout?**

This is a **compression sufficiency test**, not a new alignment method. Change only the labeled semantic payload from individual owner-support vectors to class means. Do not change N256 anchors, Procrustes, the online trajectory, the shared linear readout family, optimizer, or test protocol.

If this passes, we will have evidence that cross-client correspondence can establish a common space and that very small class-level semantic memory is sufficient inside that space. Only then should the next block test a direct aggregated prototype classifier / online integration.

## Frozen setting

Use `fedgh`, seed0, round10 only.

Reuse exactly:

- H02-A online trajectory/state;
- H03-A parent anchors and exact H04-A N256 prefix;
- H02-E held-out owner support and labels;
- client0 as reference;
- the **historical canonical H04-A N256 Procrustes transforms** (`float64` SVD, applied transform in the same dtype/path as H04-A);
- official test set for evaluation only;
- fresh zero-initialized `Linear(512,10)` heads;
- H04-A/H03-D full-batch LBFGS settings with `max_iter=2000`, no regularization.

Before the new arms, reproduce the entire historical H04-A N256 canonical result exactly. If exact reproduction fails, stop and report; do not continue to prototypes.

## Prototype construction

For every client `i` and every class `c` actually present in that client's frozen H02-E owner support, compute locally in the **raw personalized feature space**:

`p_ic = mean_{x in support_i, y=c} z_i(x)`.

Record the class id `c` and support count `n_ic`. Do not use test examples, anchor labels, or any globally missing-class examples to construct a prototype.

Then apply the already-frozen client-specific N256 affine Procrustes transform to the prototype:

`p_ic_aligned = (p_ic - mu_i) R_i + mu_ref`.

Because the map is affine, this must equal the mean of the individually aligned owner-support features for the same client/class. Add an integration assertion/receipt for every client-class pair showing

`align(mean(raw_features)) ~= mean(align(raw_features))`

within a tight tolerance consistent with the historical float32 applied path. This equality is scientifically important: it proves the compression can happen **before upload**; the server does not need individual labeled owner-support features merely to form the aligned class prototype.

Derive the number of prototypes from the frozen split; do not hardcode `20`, although with 10 clients x 2 owned classes it is expected to be 20.

## Arms

Run exactly three arms on the same frozen state:

1. `full_aligned_support_reference`
   - Exact historical H04-A N256 canonical arm.
   - Train on every individually aligned H02-E owner-support feature.
   - Must reproduce missing `21.9875%`, seen `37.55%`, all `25.10%`, support fit `100%` exactly before interpreting the new arms.

2. `aligned_client_class_prototypes`
   - Train a fresh shared head on only the aligned `(client,class)` mean prototypes.
   - Use **count-weighted cross entropy**: weight prototype `(i,c)` by its frozen support count `n_ic`, normalized by total support count. This preserves the original client/class sample weighting while changing only within-class compression.
   - Do not duplicate prototypes in memory just to implement weights if an equivalent weighted loss is easy.

3. `native_client_class_prototypes_control`
   - Construct the exact same `(client,class)` means and counts, but do **not** apply cross-client alignment.
   - Train the same fresh shared head with the same count-weighted CE and LBFGS-2000 settings in the native client coordinates.
   - Evaluate each client test feature in its own native coordinates with that same shared head.
   - This is the matched low-payload control for whether prototype compression alone, without correspondence-based alignment, explains the result.

Do not add a direct cosine prototype classifier, global-class aggregation, pair breaking, random completion, normalization, regularization, PCA, learned mapping, another optimizer, or more iterations in H06-B. Those would change a second scientific variable.

## Required measurements

For each new prototype arm report:

- number of client-class prototypes and per-client/per-class counts;
- prototype-training accuracy, weighted CE, iterations/evaluations, `grad_inf`, `grad_l2`, weight/bias norms;
- **full owner-support sample accuracy under the prototype-trained head** (evaluate on all corresponding individual support features, aligned for the aligned arm and native for the native arm); this is diagnostic only, not another training source;
- seen / missing / all / macro test accuracy and per-client counts;
- all prototype hashes and labels/count receipts;
- per-client-class `align(mean) vs mean(align)` max/Frobenius error for the aligned arm;
- exact state/RNG/module-mode/pre-existing-gradient isolation receipts.

### Communication accounting

Report communication honestly and separately for the two conceptual payloads.

1. **Labeled semantic payload only**
   - full-support diagnostic: `#support_vectors * 512 * 4` bytes plus per-vector labels;
   - prototype arm: `#client_class_prototypes * 512 * 4` bytes plus one class id and one count per prototype.
   - Derive exact bytes from dtypes actually used. Report vector-count compression and byte compression.

2. **Anchor + labeled semantic payload**
   - add the unchanged N256 anchor-feature payload required for alignment to both totals;
   - report per-client and total bytes and compression ratio.

Do **not** call the semantic-vector compression ratio the total PPRTP communication reduction. N256 anchor transport remains a real cost and must be shown explicitly.

## Tests / integrity

Preserve all existing 35 tests. Add only minimal tests for:

- affine `transform(mean(X)) == mean(transform(X))` on synthetic data;
- weighted prototype CE equals CE on explicitly repeated prototypes for integer counts, within numerical tolerance;
- prototype construction never mixes classes or clients and uses only supplied support examples;
- no global RNG mutation from the diagnostic path.

Keep the existing exact-online, provenance, anchor-label isolation, test-only evaluation, state/RNG/module-mode/pre-existing-gradient, and finiteness checks.

## Predeclared interpretation

Let the exact full aligned-support reference be:

- `M_full = 21.9875%` missing;
- `A_full = 25.10%` all-class.

Let aligned prototype missing/all be `M_proto, A_proto`, and native-prototype missing be `M_native`.

Require prototype-training fit >=95% before using an arm for the scientific gate. If an arm is fit-limited, report it as fit-limited and stop; do not tune the optimizer in this block.

Define:

- `ret_missing = M_proto / M_full`;
- `ret_all = A_proto / A_full`;
- `alignment_gain = M_proto - M_native` in percentage points.

### A. Strong class-prototype compression

Conclude that one prototype per client-owned class is sufficient for the next method step if all are true:

- aligned prototype training fit >=95%;
- `ret_missing >= .80` (`M_proto >= 17.59%`);
- `ret_all >= .80` (`A_proto >= 20.08%`);
- `alignment_gain >= 8pp`.

Then the next lead step will be a **direct global-class prototype aggregation/classifier test** using these aligned client-class prototypes, not a more complex transport model.

### B. Class means are too lossy

If aligned prototype training fit >=95% but `ret_missing < .50` (`M_proto < 10.99375%`), conclude that one class mean discards too much of the owner semantic geometry even after successful alignment. Do not add multiple prototypes or clustering inside this block; report the failure cleanly.

### C. Intermediate

Otherwise report the full table and diagnostics. Do not tune prototype weighting, normalize prototypes, select classes, or add multiple prototypes per class.

## Deliverable

Append `CODEX REPORT H06-B` with STATUS, source SHA, exact commands/run IDs, tests, exact H04-A canonical reproduction, prototype construction/count/hash receipts, affine-mean equivalence receipts, the three-arm table, communication accounting, integrity receipts, warnings, and interpretation under the frozen gate.

Do **not** implement online training changes, a direct cosine/global-prototype classifier, same-class global aggregation, multi-prototype clustering, learned transport, relation kernels, PCA, regularization, another optimizer, more iterations, or seeds1/2 in H06-B. Await research-lead review.


## CODEX REPORT H06-B — DONE (2026-09-18 07:27 +08)

STATUS: DONE; **A: strong class-prototype compression** under frozen gate. Source `b72e17742b17cf6ce508a4932866e41dd2ea3cd6`;release `20260918-072425-h06b`;run `20260918-072451-h06b-prototypes`,exit0,finished07:25:44+08.

Changed pprtp/class_prototypes.py,pprtp/fedgh.py(optionalcountweightedCE;defaultunchanged),pprtp/run.py,tests/test_class_prototypes.py,scripts/report_h06b.py.37 tests pass locally/remotely,all35previouspreserved. Added2tests cover affine mean identity,class/client isolation,countweightedCE andgradient versus explicitlyrepeatedprototypes with unequalintegercounts,and diagnosticstate/RNG/mode/preexistinggradient preservation. No failedtest/run.

Commands (AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h06b
./scripts/autodl-run.ps1 -Name h06b-prototypes -Cmd 'PPRTP_SOURCE_SHA=b72e17742b17cf6ce508a4932866e41dd2ea3cd6 bash scripts/run_h01.sh --modes fedgh --seeds 0 --rounds 10 --class-prototype-probe'
D:\anaconda3\python.exe scripts/report_h06b.py research_log/H06B/gate
```

EntireH04A N256 canonical result assertedexact beforeprototypearms:alignment/transformhashes,rankreceipts,headhash,fit/gradient/testmetrics andstate. All10H02A online metrics/model/prototype/server records exact. Frozen PFLlib0169ba7,CIFAR10subset,10clients2classes,100train/class/client,test100/class,CNN512features,SGD.01,1epoch,batch32,seed0round10 unchanged. Exactparentanchors1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125,N256prefix andH02E support2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073 reused. Noonlinechange.

Prototypecount is derived from eachsupportlabelset. Observed20prototypes,2/client,100supportexamples/prototype;2000supportexamplestotal. Classes byclient:0[4,6],1[2,6],2[2,7],3[3,7],4[3,5],5[5,9],6[0,9],7[0,8],8[1,8],9[1,4]. Rawmeancomputedlocallybeforetransform;alignedmean uses exactlyhistoricalfloat32appliedcanonicalaffinemap. All20 align(mean)vsmean(align) checks pass atol1e-6/rtol1e-5;worstmaxerror4.76837158e-7,Froerror1.17560216e-6. This verifiesmeanconstruction canprecedeupload;individualsupportfeatures are not needed atserver toformtheseprototypes. No class/clientmixing;rawmean hashes/counts/labels exactlymatched acrossnewarms.

AlignedprototypecombinedSHA256 `2703c2ae40743bb48a0e21fd43949746ccc9fe85a78b44bdfad006ecc611c97d`;native `15c16a0674242edce13a858f0721eb791c9b07a881918d97e63a0b8e9da5a4c5`. Individualraw/alignedhashes,labels/counts/hashreceipts and20affinemeanerrors in final.json/RESULTS.md.

Freshzero512->10float32head eacharm;fullbatchLBFGS lr1,strong_wolfe,max_iter2000,tolerance_grad1e-9,tolerance_change1e-12,noregularization. PrototypeCE=sum(n_ic*CE_i)/sum(n_ic),withoutrepeatallocation. Currentequal100counts implyequalprototypeweights. Initial/finalCE+accuracy,weightedobjectivegradient,headnorms,iterationcounts,classcorrect/counts saved. Fullindividualsupport evaluatedonlyafterfitting;noindividualsupportlossentersprototypeoptimization.

# H06-B owner-class prototype compression

| Arm | Seen % | Missing % | All % | Macro % | Training fit % | Training CE | Full support fit % | grad_inf | grad_l2 | W/b norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| full_aligned_support_reference | 37.550000 | 21.987500 | 25.100000 | 25.100001 | 100.000000 | 7.68896147e-09 | 100.000000 | 1.4978065e-08 | 9.88137003e-08 | 1050686/24634.9922 | 1227/1308 |
| aligned_client_class_prototypes | 27.950000 | 23.887500 | 24.700000 | 24.700000 | 100.000000 | 4.76837094e-08 | 29.150000 | 2.6978098e-08 | 1.90271805e-07 | 1080.33606/131.921036 | 38/52 |
| native_client_class_prototypes_control | 67.200000 | 0.000000 | 13.440000 | 13.440000 | 100.000000 | 5.18559034e-07 | 67.350006 | 3.47188291e-07 | 2.58797149e-06 | 1764.81299/79.758255 | 34/54 |

Frozen verdict: A: strong class-prototype compression. ret_missing=1.08641273; ret_all=0.984063742; alignment_gain=23.887500pp.


Communication (actual float32vectors,int64labels,int64counts; conceptual upload tensors only):

| Payload | Full support per client | Prototype per client | Full total | Prototype total | Compression |
|---|---:|---:|---:|---:|---:|
| Semantic vectors | 409600 B | 4096 B | 4096000 B | 40960 B | 100x |
| Labels | 1600 B | 16 B | 16000 B | 160 B | 100x |
| Prototype counts | 0 B | 16 B | 0 B | 160 B | — |
| All semantic payload | 411200 B | 4128 B | 4112000 B | 41280 B | 99.6124x |
| Unchanged N256 anchors | 524288 B | 524288 B | 5242880 B | 5242880 B | 1x |
| Anchors + semantic | 935488 B | 528416 B | 9354880 B | 5284160 B | 1.77036x |

Vectorcount2000->20 =100x. Totalpayloadreductionincludinganchors is1.77036x,not100x. Nativecontrolneedsnoanchorsandhas41280Bsemanticpayloadtotal. Networkframing,distributionofanchorimages,downlinkandamortizedonlineprotocolare notestimated;thisblockmeasures the requested two conceptualfeatureuploadpayloads only.

Interpretation: bothprototypefits100%>=95%;ret_missing1.08641273>=.80,ret_all.984063742>=.80,and alignmentgain23.8875pp>=8. Allstronggateconditionspass. One localmean perownedclass retains the requiredmissing/allperformance insidepairedalignedspace;nativeprototypecontrolgets0missing,so compressionwithoutalignmentdoesnotexplaintransfer. This is aseed0round10frozenreadout result,notmultiseed/onlinevalidation andnotyetdirectglobalprototypeclassification.

Importantdiagnostic: alignedprototypeheadfitsmeans100%butindividualsupportonly29.15%,withtestseen27.95% versusfullreference37.55%;nativeprototypeheadfullsupport67.35%,seen67.20%,missing0. Compressiondoesnotpreservewithinclasssampledecisions,despitepassingthemissing/allgate. Reportthistradeoffexplicitly;do notclaim losslesssemanticgeometry. Fullreferenceheadnorm~1.05e6 remainslarge;alignedprototypeheadnorm1080.34,bias131.92,nativenorm1764.81,bias79.76;no tuning based on these values.

Integrity:state/model/server/prototypes/buffers,modulemodes,CPU/CUDA RNG,preexistinggradientsunchanged;allarms same frozenstate. Anchorlabelsunused,prototypesconstructedonlyfromprovidedheldoutownerexamples/labels,testevaluationonly;allprototypes/features/losses/gradients/logits/parametersfinite. KnownNVMLinitializationwarningsretained;noSVDwarning/failure. Evidence `research_log/H06B/gate/` RESULTS.md,verification.json,rawfinal/rounds/provenance/split,tests/meta/run/log/compacttar. Remoteoriginals/checkpoints `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-072451-h06b-prototypes`.

Stopawaitleadreview. Noaggregationacrossowners,directcosine/globalprototypeclassifier,normalization,multiprototypes,onlinechanges,newoptimizer/iterations,orotherseedsimplemented.
