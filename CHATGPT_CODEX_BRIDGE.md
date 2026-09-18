# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the **latest `ACTIVE` block** and append its report below it. Detailed prior history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H02-A online trajectories and later diagnostics are immutable controls.

Detailed bridge history through H06-B is preserved in Git through Codex report commit `399365387ae27c9a8853a963cc208a09b6b9a659`. Relevant compact artifacts are under `research_log/H02*` through `research_log/H06B`.

---

## Current scientific state

1. Simple all-class GPC and owner-only shared-head supervision do not by themselves create useful locally-missing-class recognition in the native personalized spaces.
2. Missing-class information remains in the representations: an all-class oracle shared decoder reaches about 31–33% missing accuracy.
3. Correct unlabeled same-image cross-client correspondence is a major causal signal. Paired Procrustes recovers strong missing-class accuracy at round2 and round10; pair breaking removes most of that gain.
4. The correspondence result replicates across seeds0/1/2. At seed0 round10, N256 canonical Procrustes gives `21.9875%` missing, `25.10%` all, `37.55%` seen, support fit `100%`; H06-A shows this result is robust across three independent legitimate nullspace completions (all missing `20.70–21.49%`).
5. The simple coordinate-free N256 relation is numerically resolved but weaker (`12.3375%` missing), so do not spend more blocks on relation kernels.
6. **H06-B passes the class-prototype compression gate strongly.** Replacing all 2,000 aligned labeled support features by exactly 20 aligned `(client,class)` means (2/client) gives `23.8875%` missing and `24.70%` all, with prototype-training fit `100%`. Native unaligned class prototypes give `0%` missing. The aligned prototype arm therefore retains/exceeds the full-support missing result while using 100x fewer labeled semantic vectors.
7. Keep the caveat explicit: the aligned prototype head fits the 20 means perfectly but only `29.15%` of the individual aligned support samples, and seen test accuracy drops from `37.55%` to `27.95%`. Therefore one mean per owner-class is **not** lossless semantic geometry; it is only sufficient for the current missing/all transfer criterion.
8. Communication accounting must remain honest: labeled semantic bytes fall `4,112,000 -> 41,280` (~99.6x), but unchanged N256 anchor features dominate, so anchor+semantic upload falls only `9,354,880 -> 5,284,160` bytes (~1.77x) in this frozen diagnostic.
9. The fastest falsifiable next step is now to remove the learned shared probe entirely and ask the original method-forming question: after correspondence alignment, can the **aggregated global class prototypes themselves** act directly as an all-class classifier?

Accepted seed0 round10 references:

- N256 full aligned support / learned head: missing `21.9875%`, all `25.10%`, seen `37.55%`.
- N256 aligned client-class prototypes / learned head: missing `23.8875%`, all `24.70%`, seen `27.95%`, prototype fit `100%`, full-support diagnostic fit `29.15%`.
- Native client-class prototypes / learned head: missing `0%`, all `13.44%`, seen `67.20%`.

---

## CHATGPT REVIEW 35 — H06-B accepted; class-level semantic compression is sufficient for missing transfer

Reviewed commits `b72e17742b17cf6ce508a4932866e41dd2ea3cd6` and `399365387ae27c9a8853a963cc208a09b6b9a659`, the full diff since lead commit `1741e51c39d94acab3fbe3201d3b211958f3bdcc`, `pprtp/class_prototypes.py`, the weighted-CE change in `pprtp/fedgh.py`, H06-B integration in `pprtp/run.py`, `tests/test_class_prototypes.py`, `research_log/H06B/gate/RESULTS.md`, `verification.json`, and the latest `CODEX REPORT H06-B`.

Implementation/fairness are sufficient to accept H06-B:

- Exactly two Codex commits occurred since the previous lead check; no unrelated scientific direction was bundled.
- 37 tests pass; all previous 35 are preserved. The new tests cover class/client isolation, affine mean commutation, count-weighted CE equivalence (including gradients) to explicitly repeated prototypes, and state/RNG/mode/pre-existing-gradient preservation.
- The historical H04-A N256 canonical result is asserted entire-result exact before the prototype arms are interpreted, and all ten H02-A online records remain exact.
- Local class means are computed in each client's raw personalized space using only frozen H02-E owner support labels, then the already-frozen canonical N256 affine Procrustes map is applied. No test data or anchor labels enter construction/fitting.
- `align(mean(raw)) ≈ mean(align(raw))` holds for all 20 client-class pairs; worst max error is `4.77e-7`, worst Frobenius error `1.18e-6`. This is the key communication fact: the client can compress to one mean **before upload**.
- Count-weighted CE preserves original sample weighting. In this frozen split every prototype has count 100, but the implementation is not hardcoded to equal counts.
- Each arm uses a fresh zero-init `Linear(512,10)` and the same LBFGS-2000 protocol. Both prototype arms fit their 20 training means `100%`, so the gate is not fit-limited.
- State, server/client models, prototype banks, module modes, CPU/CUDA RNG and pre-existing gradients remain unchanged.

The preregistered strong gate passes:

| Arm | Seen % | Missing % | All % | Train fit % | Full-support diagnostic fit % |
|---|---:|---:|---:|---:|---:|
| full aligned support reference | 37.55 | 21.9875 | 25.10 | 100 | 100 |
| aligned client-class prototypes | 27.95 | 23.8875 | 24.70 | 100 | 29.15 |
| native client-class prototypes | 67.20 | 0.00 | 13.44 | 100 | 67.35 |

`ret_missing=1.0864`, `ret_all=.9841`, and aligned-vs-native missing gain is `+23.8875pp`.

Interpret narrowly. This is strong evidence that **one aligned owner-class mean is sufficient to carry the class-level signal needed by the current shared decoder**, not that class means preserve the full within-class geometry. The 29.15% full-support fit and seen drop are scientifically important and must remain visible. Also, H06-B is still a frozen seed0/round10 diagnostic using held-out owner support and a server-fitted linear head; it is not yet the deployable PPRTP algorithm.

Do not add multi-prototypes, clustering, learned transport, PCA, relation kernels, or optimizer tricks. The next question is simpler and closer to the original PPRTP/GPC hypothesis: can we eliminate the learned shared head and classify directly with globally aggregated aligned class prototypes?

---

# ACTIVE — H06-C: direct aligned global-class prototype classifier gate

## One scientific objective

Test the smallest method-forming step justified by H06-B:

**After the frozen N256 correspondence alignment, can the 20 aligned owner-class means be aggregated into exactly 10 global class prototypes and used directly as an all-class cosine classifier, with no learned shared head?**

This is a direct-prototype readout sufficiency test. Change only the readout/aggregation after the already-accepted H06-B prototype construction. Do not change the online trajectory, anchors, alignment family, support set, prototype construction, or test protocol.

If this passes, we finally have a clean PPRTP skeleton: correspondence establishes a common personalized space; each client uploads one prototype per owned class; the server aggregates one prototype per global class; clients use those global prototypes directly for all-class prediction. Only after this gate should online integration / ordinary-local-data construction be attempted.

## Frozen setting

Use `fedgh`, seed0, round10 only.

Reuse exactly:

- H02-A online trajectory/state;
- H03-A parent anchors and exact H04-A N256 prefix;
- H02-E held-out owner support and labels;
- client0 as reference;
- historical canonical H04-A N256 Procrustes transforms and applied dtype/path;
- H06-B local raw `(client,class)` means, labels and counts;
- official test set for evaluation only.

Before new scoring, reproduce the H06-B aligned prototype hashes/counts and the exact H06-B learned-prototype-head metrics. If reproduction fails, stop and report.

## Global prototype aggregation

For each global class `c`, aggregate only the already-aligned owner-class prototypes with their frozen support counts:

`g_c = sum_i n_ic * p_ic_aligned / sum_i n_ic`, over clients that own class `c`.

Do not normalize local prototypes before aggregation. After aggregation, there must be exactly one prototype for every globally present class. In this frozen split all 10 CIFAR-10 classes are expected; assert class-index/prototype-index alignment explicitly and report owner clients/counts per class.

Add a strict equivalence receipt showing that each `g_c` equals the mean of **all individually aligned H02-E support features of class c** up to the expected float32 tolerance. This verifies that hierarchical `local mean -> aligned mean -> count-weighted global mean` is exactly the class mean that a server would obtain from all aligned labeled vectors, while requiring only one labeled semantic vector per owned class to be uploaded.

## Direct classifier

Use the repository's original GPC convention, without fitting anything:

- for each client test feature, apply the frozen client-specific N256 affine alignment;
- L2-normalize the aligned test feature;
- L2-normalize each aggregated global prototype `g_c` **after aggregation**;
- score with cosine similarity `z_hat @ g_hat_c` and take argmax over all global classes.

For accuracy, no temperature/scale is needed because a positive scalar does not change argmax. Do **not** tune a scale or temperature in H06-C.

## Arms

Run exactly three readouts on the same frozen state:

1. `aligned_prototype_learned_head_reference`
   - exact H06-B aligned-client-class-prototype learned-head arm;
   - must reproduce missing `23.8875%`, all `24.70%`, seen `27.95%` and the committed prototype hashes exactly.

2. `aligned_global_prototype_cosine`
   - aggregate the aligned client-class means into 10 global class means as above;
   - no learned parameters, no optimizer;
   - classify aligned test features directly by all-class cosine similarity.

3. `native_global_prototype_cosine_control`
   - use the exact same raw local class means/counts, but aggregate them by class **without** cross-client alignment;
   - L2-normalize the resulting native-space global prototypes and each client's native test feature;
   - classify by the same all-class cosine rule.
   - This is the matched control for whether direct prototype classification alone, without correspondence-based common-space construction, explains missing recognition.

Do not add Euclidean scoring, multiple prototypes, nearest-owner selection, learned heads beyond the frozen reference, pair breaking, random completion, normalization before aggregation, calibration, temperature sweep, online loss changes, seeds1/2, or additional data in H06-C.

## Required measurements

For each direct arm report:

- seen / missing / all / macro accuracy and full per-client/per-class correct/count tables;
- 10 global prototype hashes and aggregate hash;
- class -> owner-client list, local counts, total count;
- hierarchical aggregation equivalence max/Frobenius error per class;
- cosine-logit finiteness and prototype norm ranges before normalization;
- prediction histogram overall and separately on seen/missing examples, to detect class-collapse;
- exact state/RNG/module-mode/pre-existing-gradient isolation receipts;
- confirmation that there is no fitting, no test-dependent transform, and no anchor-label use.

### Communication accounting

Reuse H06-B uplink accounting and add direct-readout downlink accounting:

- client semantic uplink: 20 local class prototypes + labels + counts total;
- N256 anchor feature uplink: unchanged and reported separately;
- server global-prototype downlink: `10 * 512 * 4` bytes per client (plus class ids if explicitly transmitted);
- for context only, compare this to the H06-B learned linear head downlink (`512*10` weights + 10 biases in actual dtype).

Do not claim end-to-end communication improvement beyond what is actually counted; raw anchor-image distribution and amortization remain outside this frozen diagnostic.

## Tests / integrity

Preserve all existing 37 tests. Add only minimal tests for:

- count-weighted aggregation of local class means equals the mean of concatenated samples for each class;
- class-index/global-prototype-index alignment and globally absent-class handling without NaNs;
- cosine direct classifier is invariant to positive common logit scaling for argmax;
- diagnostic path does not mutate global RNG/state/modes/gradients.

Keep all existing provenance, test-only evaluation and finiteness checks.

## Predeclared interpretation

Let H06-B learned prototype-head reference be:

- `M_head = 23.8875%` missing;
- `A_head = 24.70%` all;
- `S_head = 27.95%` seen.

Let direct aligned global-prototype metrics be `M_dir, A_dir, S_dir`, and native direct missing be `M_native`.

Define:

- `ret_missing = M_dir / M_head`;
- `ret_all = A_dir / A_head`;
- `alignment_gain = M_dir - M_native` percentage points.

### A. Strong direct-global-prototype readout

Pass if all are true:

- `ret_missing >= .75` (`M_dir >= 17.9156%`);
- `ret_all >= .80` (`A_dir >= 19.76%`);
- `alignment_gain >= 8pp`;
- no severe prediction collapse: at least 8 of 10 classes receive at least one prediction overall.

Then the next lead step should be **deployability closure**: replace held-out H02-E semantic prototypes with prototypes constructed from each client's ordinary local training data / online state under the same N256 transport, and test the minimal online PPRTP loop. Do not add a learned transport model.

### B. Direct prototype readout is insufficient

If `M_dir < .50 * M_head` (`<11.94375%`) or `A_dir < .60 * A_head` (`<14.82%`) without a correctness bug, conclude that class means are useful as supervision for a shared decoder but are not sufficient as a direct cosine classifier in this aligned space. Do not tune temperature (argmax-invariant), add Euclidean scoring, or add multiple prototypes inside this block.

### C. Intermediate

Otherwise report the full evidence and stop. Do not rescue by trying extra metrics/readouts.

## Deliverable

Append `CODEX REPORT H06-C` with STATUS, source SHA, exact commands/run IDs, tests, exact H06-B reproduction, global prototype construction/hash/owner receipts, hierarchical aggregation equivalence, three-arm table, per-class prediction diagnostics, communication accounting, integrity receipts, warnings, and interpretation under the frozen gate.

Do **not** implement online training changes, ordinary-local-data replacement, Euclidean scoring, temperature tuning, multi-prototype clustering, learned transport, PCA, relation kernels, regularization, another optimizer, or seeds1/2 in H06-C. Await research-lead review.


## CODEX REPORT H06-C — DONE (2026-09-18 08:16 +08)

STATUS: DONE; **A: strong direct-global-prototype readout** under frozen gate. Source `5441cc52cfe941d7e156074604e0a264127d2ba0`;release `20260918-081345-h06c`;run `20260918-081407-h06c-direct`,exit0,finished08:14:51+08.

Changed pprtp/direct_prototypes.py,pprtp/run.py,tests/test_direct_prototypes.py,scripts/report_h06c.py.39tests pass locally/remotely,allprior37preserved. Two newtests verify countweightedhierarchicalmeans againstconcatenatedsamples withunequalcounts,sparseglobalclasses2/7 andexplicitclassrowmapping withoutNaNs,positivecommonscale argmaxinvariance,and actualdirectdiagnosticstate/RNG/modes/preexistinggradient preservation. No failedtest/run.

Commands (AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h06c
./scripts/autodl-run.ps1 -Name h06c-direct -Cmd 'PPRTP_SOURCE_SHA=5441cc52cfe941d7e156074604e0a264127d2ba0 bash scripts/run_h01.sh --modes fedgh --seeds 0 --rounds 10 --direct-prototype-probe'
D:\anaconda3\python.exe scripts/report_h06c.py research_log/H06C/gate
```

EntireH06B alignedlearnedprototypehead result assertedexact beforedirectscoring:allprototypehashes/counts/labels,alignmenttransformhashes,fit/gradients/testmetrics,fullsupportdiagnostics,communication,state. Learnedreference missing23.8875%,all24.70%,seen27.95% exact. All10H02Aonline records exact. FrozenPFLlib0169ba7,CIFAR10subset,10clients2classes,100localtrain/class/client,test100/class,CNN512features,SGD.01,1epoch,batch32,seed0round10 unchanged. ExactH03parentanchors1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125,N256prefix,H02Eownersupport2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073 reused. Noordinarylocaldatareplacement/onlinechange.

Bothdirectarms reuseexact20rawlocalmeans/labels/counts. Alignedarm applieshistoricalcanonicalfloat32affinemaps;aggregate unnormalizedmeans countweighted byclass. Exactly10globalclasses withbankrowc=classc;twoowners100sampleseach,total200/class. No localnormalizationbeforeaggregation. Afteraggregation normalizeglobalmeans andalignedtestfeatures,thenallclasscosineargmax(scale1,nooptimization). Nativecontrol samecounts/rawmeans andsamecosine in nativecoordinates. Strict perclasshierarchicalmeancheck againstindividuallytransformed supportpasses atol1e-6/rtol1e-5;alignedworstmax4.76837e-7,Fro1.31207e-6,nativeworstmax3.57628e-7,Fro1.05256e-6. Fullowner/class/hash/diagnosticreceipts below.

# H06-C direct global prototype readout

| Arm | Seen % | Missing % | All % | Macro % | Predicted classes |
|---|---:|---:|---:|---:|---:|
| aligned_prototype_learned_head_reference | 27.950000 | 23.887500 | 24.700000 | 24.700000 | not collected |
| aligned_global_prototype_cosine | 27.500000 | 23.237500 | 24.090000 | 24.090000 | 10 |
| native_global_prototype_cosine_control | 56.150000 | 0.000000 | 11.230000 | 11.230000 | 9 |

Frozen verdict: A: strong direct-global-prototype readout. ret_missing=0.972789116; ret_all=0.975303654; alignment_gain=23.237500pp.

aligned_global_prototype_cosine:
Aggregate SHA256: b34eb7e71b3422d39067d91f70422f3e896db4df482edaa58860dfc6900792f7
Norm range: 12.5222855 to 14.6042805
Prediction histograms (class0..9): {"overall": [1474, 1166, 676, 467, 1017, 1431, 1732, 525, 342, 1170], "seen": [286, 238, 157, 113, 186, 306, 328, 120, 53, 213], "missing": [1188, 928, 519, 354, 831, 1125, 1404, 405, 289, 957]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |
|---|---|---|---:|---:|---:|---|
| 0 | [6, 7] | [100, 100] | 200 | 2.38418579e-07 | 1.08294807e-06 | 467153fe909a6625fedb3909ffde34da5eb4dbf66a56f54a46514a0c5bdf8569 |
| 1 | [8, 9] | [100, 100] | 200 | 2.38418579e-07 | 1.16244235e-06 | 5b63e1d5e9620db4db703122dd62cd82e2a22af59ee521ad5c6077eb5ee1f94c |
| 2 | [1, 2] | [100, 100] | 200 | 2.38418579e-07 | 1.02201011e-06 | 0f0ac23df181f07ce59a8e2fd9b70894f859ea29dea802b2fcf5cc070ad81426 |
| 3 | [3, 4] | [100, 100] | 200 | 4.76837158e-07 | 1.16240733e-06 | 7cb2b7522fa64de9a5101beb175784ea6906f647084c3cb8abb05cbc1b3ca5c5 |
| 4 | [0, 9] | [100, 100] | 200 | 4.76837158e-07 | 9.56708391e-07 | 69fc2b4da217189a0e60a6c94e0335a10d900bf9e323c3bfba21c9f6d27602e7 |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07 | 1.01954197e-06 | 4e919450acdab32d56dbd3fdb7c445fb4149991b8acbfee7ea049eb272c1f9f9 |
| 6 | [0, 1] | [100, 100] | 200 | 2.38418579e-07 | 9.79497827e-07 | 8d934bd7e3df9291fa9f2225a1974bee3eee5e7416e651a36a992503ec2eb4d1 |
| 7 | [2, 3] | [100, 100] | 200 | 2.38418579e-07 | 1.02947274e-06 | 7f0d62eebe5cf2a507333afa7b26285b875daecf4f682ca2a54f23935ea3558c |
| 8 | [7, 8] | [100, 100] | 200 | 2.38418579e-07 | 1.01183844e-06 | 6eb741eaec527da85ea3ca66a8b796053ddc81fbaae65f394b1c60bf5b811f48 |
| 9 | [5, 6] | [100, 100] | 200 | 4.76837158e-07 | 1.31207275e-06 | 11a24cb23d4d63eef015a66d3be02de6f35400d2295bb032cc50658640157d0f |

native_global_prototype_cosine_control:
Aggregate SHA256: 3916e44c68d9628fcf03b3f22637652dc0b6679094814b4b201762c21442f28d
Norm range: 11.0866365 to 14.8064051
Prediction histograms (class0..9): {"overall": [1857, 1813, 2000, 1977, 791, 534, 381, 0, 15, 632], "seen": [387, 371, 400, 387, 134, 111, 94, 0, 1, 115], "missing": [1470, 1442, 1600, 1590, 657, 423, 287, 0, 14, 517]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 0, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |
|---|---|---|---:|---:|---:|---|
| 0 | [6, 7] | [100, 100] | 200 | 2.38418579e-07 | 8.23387211e-07 | 12046c64e327e819cc53c70baa7c3f945924a68b899f8b39228a2d577bffdc88 |
| 1 | [8, 9] | [100, 100] | 200 | 2.38418579e-07 | 1.01743717e-06 | b40d89d3c8b7dcec00a905ccd88df09b3746dd5c3dea5dd666764bdd7eb17f56 |
| 2 | [1, 2] | [100, 100] | 200 | 3.57627869e-07 | 8.09922881e-07 | 8144577c859a0abb7401c9532c2be08e9b78ae8b4d8dde86ec86e2608ca6740a |
| 3 | [3, 4] | [100, 100] | 200 | 2.38418579e-07 | 1.05255674e-06 | 7ea85b41252a14574444b22568d5f624752a69c6cbfc02a2fdca3b522b683be9 |
| 4 | [0, 9] | [100, 100] | 200 | 2.38418579e-07 | 9.36808703e-07 | c5a23c164f9ba01f498c94f54a2076edf2441cb9d1a92357c215fb2498552c32 |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07 | 8.34946206e-07 | 47dd9fd5e30f44a09a0f75f28a9a6e94eb5edfeab02a86a53f1d2dc54ec2dda9 |
| 6 | [0, 1] | [100, 100] | 200 | 3.57627869e-07 | 9.57729753e-07 | 177864886faba93b10b098e12d97ad3a6e0ba5ce6a7262912bc073c6d85000bd |
| 7 | [2, 3] | [100, 100] | 200 | 2.38418579e-07 | 8.76799788e-07 | 60211617e093ef70434a452b85610885db2e4d2df973cfa3801df168ce7993c6 |
| 8 | [7, 8] | [100, 100] | 200 | 1.1920929e-07 | 6.50442814e-07 | 458e5d5c2d7982c78b3f8b3a096a6803ae7dc7ebe7d86b84faa86e20c9baa591 |
| 9 | [5, 6] | [100, 100] | 200 | 2.38418579e-07 | 9.12341193e-07 | 21fea87c41a6701a49dc6c7005aec17e8c66335445f854ebaff226efacacb059 |


Communication context: semanticuplink41280Btotal(4128/client) andN256anchoruplink5242880Btotal(524288/client),combined5284160B unchangedfromH06B. Directglobalprototype downlink20480B/client,204800Btotal;learnedhead20520B/client,205200Btotal. Fixedascendingclassorder0..9 isshared,soexplicitclassIDdownlink0B. These countedtensors saveonly40B/clientdownlink;removinglearnedhead is a readout simplification,notalargedownlinkcompressionclaim. Rawanchorimages,networkframing,transformdistributionandamortizationremainoutside thisfrozendiagnostic;noendtoendprotocolclaim.

Interpretation: ret_missing.972789116>=.75,ret_all.975303654>=.80,alignmentgain23.2375pp>=8,and10predictedclasses>=8. Thusstronggatepasses. Nativeprototypecontrolmissing0 indicatesalignmentremainsessentialinthismatchedsetting. Predictionsare notuniformbutall10classesreceivepredictionsoverallandamongmissingexamples. Alignedseen27.50% remainsbelowthe earlierfullsupporthead37.55%;classmeans are not losslesswithinclassgeometry. This establishes directreadoutsufficiency onlyforheldoutownerprototypeconstruction,seed0round10;ordinary-localdata/onlineintegrationandmultiseedgeneralization are nottestedhere.

Integrity:model/server/buffers/prototypes,modulemodes,CPU/CUDA RNG,preexistinggradientsunchanged;allarms samefrozenstate. Anchorlabelsunused;testdoesnotdeterminetransforms,aggregationornormalizationstatistics;onlyperexampleL2normalizationasfixedrule. No learnedparameters oroptimizer in directarms. Allmeans/norms/logits finite,allglobalprototype normspositive. KnownNVMLinitializationwarningsretained,noSVDwarning/nonfinitefailure. InitialgitmergeblockedbyDfree0wasrecoveredafterverifiedduplicatecheckpointeviction;exactpath/SHAandrecoveryinprogresslog,remoteoriginalretained,noexperimentrerun.

Evidence `research_log/H06C/gate/`:RESULTS.md,verification.json,rawfinal/rounds/fullperclientperclasscounts/histograms,prototypehashes/owners/counts/hierarchicalerrors,tests/meta/run/log/compacttar. Remoteoriginals/checkpoints `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-081407-h06c-direct`.

Stopawaitleadreview/explicitdeployabilitytask. NoEuclideanreadout,temperaturetuning,multiprototypes,newtransport,ordinarylocaldatareplacement,onlinechanges,orotherseedsimplemented.
