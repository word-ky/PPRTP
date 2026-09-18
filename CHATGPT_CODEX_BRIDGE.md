# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the **latest `ACTIVE` block** and append its report below it. Detailed prior history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H02-A online trajectories and later diagnostics are immutable controls.

Detailed bridge history through H06-C is preserved in Git through Codex report commit `285250999d8148d0754e01f0dd44dc728c6dc4b4`. Compact artifacts are under `research_log/H02*` through `research_log/H06C`.

---

## Current scientific state

1. Native personalized spaces do not support useful missing-class recognition from global prototypes by themselves: the matched native direct global-prototype cosine control remains `0%` missing.
2. Correct unlabeled same-image cross-client correspondence is the dominant positive mechanism found so far. Paired Procrustes survives pair breaking controls, round10 persistence, seeds0/1/2, N256 anchor compression, and nullspace-completion sensitivity checks.
3. At seed0 round10, N256 canonical Procrustes with full aligned held-out support gives `21.9875%` missing / `25.10%` all / `37.55%` seen.
4. One aligned owner-class mean per `(client,class)` is enough for the current transfer signal: the H06-B learned prototype-head arm gives `23.8875%` missing / `24.70%` all / `27.95%` seen from only 20 semantic vectors. This is not lossless within-class geometry: the same head fits individual aligned support samples only `29.15%`.
5. **H06-C passes the original direct-GPC readout question strongly.** Aggregating those 20 aligned means into exactly 10 global class means and using pure cosine argmax, with no learned head and no optimizer, gives `23.2375%` missing / `24.09%` all / `27.50%` seen. This retains `97.28%` of H06-B learned-head missing and `97.53%` of all accuracy; all 10 classes are predicted. The native-space direct control remains `0%` missing.
6. H06-C is still a diagnostic, not yet a deployable method. Its semantic prototypes use H02-E: a fresh extra labeled owner-support set of 2,000 CIFAR-10 train images disjoint from ordinary client training. Test data are not used for construction, so there is no test leakage, but this extra labeled set is an optimistic resource that a real FL client should not need.
7. Communication claims must remain narrow. H06-C semantic uplink is only `41,280 B` total, but N256 anchor-feature uplink is `5,242,880 B` total and dominates. Global-prototype downlink is `20,480 B/client`, essentially the same byte scale as the old 512x10 linear head. Transform/reference-anchor distribution and raw anchor-image distribution are not yet included in an end-to-end protocol claim.
8. The fastest falsifiable next step is therefore **not** a new transport or classifier. It is to remove the extra H02-E labeled semantic set and test whether the exact same direct aligned global-prototype readout works when each local class mean is computed from the client's ordinary training data under the frozen final round10 model.

Accepted seed0 round10 references:

- H06-B aligned client-class prototypes / learned head: missing `23.8875%`, all `24.70%`, seen `27.95%`.
- H06-C aligned global prototype cosine: missing `23.2375%`, all `24.09%`, seen `27.50%`, 10 predicted classes.
- H06-C native global prototype cosine: missing `0%`, all `11.23%`, seen `56.15%`, 9 predicted classes.

---

## CHATGPT REVIEW 36 — H06-C accepted; direct aligned global prototypes are sufficient, but semantic-source realism is now the blocker

Reviewed commits `5441cc52cfe941d7e156074604e0a264127d2ba0` and `285250999d8148d0754e01f0dd44dc728c6dc4b4`, the complete two-commit diff since lead commit `e35a022a2e3e6ab7a2fd7441fb1b87f631b57107`, `AGENTS.md`, `pprtp/direct_prototypes.py`, H06-C integration in `pprtp/run.py`, `tests/test_direct_prototypes.py`, `research_log/H06C/gate/RESULTS.md`, `verification.json`, `metadata.json`, raw `final.json`, and `CODEX REPORT H06-C`.

Implementation/fairness are sufficient to accept H06-C:

- Exactly two Codex commits occurred since the previous lead check; no unrelated scientific direction was bundled.
- 39 tests pass and all prior 37 are preserved. New tests cover unequal-count hierarchical class means, sparse global classes / explicit class-row mapping, cosine argmax invariance to a positive common scale, and state/RNG/module-mode/pre-existing-gradient isolation.
- The entire H06-B aligned learned-prototype reference is asserted exact before the direct arms are scored. All ten H02-A online records and frozen provenance remain exact.
- The aligned arm uses the historical canonical N256 affine transform, unnormalized local means, count-weighted class aggregation, normalization only after global aggregation, and direct all-class cosine argmax. There is no fitting, learned readout, temperature tuning, anchor-label use, or test-dependent transform.
- The hierarchical receipt is strong: each global class mean agrees with the mean of all individually aligned H02-E support features up to float32 tolerance; worst max error is `4.77e-7` and worst Frobenius error is `1.31e-6`.
- The aligned direct result is `23.2375%` missing / `24.09%` all / `27.50%` seen versus the H06-B learned-head reference `23.8875%` / `24.70%` / `27.95%`. `ret_missing=.972789`, `ret_all=.975304`, alignment gain over the matched native direct control is `+23.2375pp`, and all 10 classes receive predictions.
- The native direct arm's collapse of class 7 is not a fairness bug: it is intentionally aggregating vectors from incompatible personalized coordinates. It is a negative control for the claim that direct global prototype classification alone explains transfer.
- No experiment warning suggests numerical instability; the only retained warning is NVML initialization. State, buffers, prototypes, module modes, RNG and pre-existing gradients remain unchanged.

Scientific interpretation must stay narrow. H06-C is strong evidence for this skeleton:

`paired correspondence -> common personalized space -> one local mean / owned class -> one global mean / class -> direct cosine all-class readout`.

It does **not** yet establish a practical PPRTP algorithm because the semantic means come from H02-E's extra fresh labeled support rather than ordinary client training data. That is now the highest-value unresolved confound. Do not spend another block on relation kernels, learned transport, Euclidean readout, multi-prototypes, head fitting, or temperature sweeps.

---

# ACTIVE — H07-A: ordinary-local-training prototype source closure

## One scientific objective

Remove exactly one unrealistic resource from H06-C:

**Replace the fresh H02-E held-out owner-support semantic prototypes by class means computed from each client's ordinary local training set under the frozen final round10 model, while keeping the same N256 correspondence transport and direct global-prototype cosine readout.**

This is a semantic-source deployability gate, not an online-training modification. Do not change the H02-A trajectory, optimizer, anchors, alignment family, reference client, classifier rule, or test protocol.

If this passes, the direct PPRTP readout no longer requires any extra labeled semantic calibration data beyond what each client already owns.

## Frozen setting

Use `fedgh`, seed0, round10 only.

Reuse exactly:

- H02-A online trajectory/state and the same seed0 split;
- H03-A parent anchors and exact H04-A N256 prefix;
- client0 reference and the exact H06-C canonical N256 Procrustes path/dtype;
- official test subset for evaluation only;
- H06-C direct aligned global-prototype result as the fixed reference.

Before the new arm, reproduce the complete H06-C aligned direct result and global-prototype hashes exactly. If reproduction fails, stop and report.

## Ordinary local semantic source

For each client `i`, use **only its original local training dataset** from the frozen split (`split['train_indices'][i]`, equivalently the exact `trains[i]` constructed by `prepare`). These are the same 200 images/client (100 per owned class) already used by local SGD.

At round10, after the frozen online update is complete:

1. Put the client model through the existing side-effect-free `features(...)` path so the **final round10 model** extracts features for its ordinary local training images.
2. Compute exactly one raw mean per owned class and its count.
3. Apply the exact same frozen N256 affine Procrustes transform as H06-C to each local mean.
4. Count-weight aggregate aligned local means into exactly 10 global class means.
5. Normalize only after global aggregation and classify aligned test features by direct all-class cosine argmax, exactly as H06-C.

Important: do **not** use `client.protos` as the primary H07-A semantic source. Those online objects are collected from features produced across the local SGD minibatch trajectory before successive optimizer steps, so they are not in exactly the same final representation state as the round10 anchor/test features. H07-A isolates only the data-source change. A zero-extra-pass `client.protos` audit can be considered later if H07-A passes.

No H02-E labels/images may enter the new local-training arms. H02-E may be used only to reproduce the frozen H06-C reference.

## Arms

Run exactly three readouts on the same frozen state:

1. `heldout_aligned_global_prototype_cosine_reference`
   - exact H06-C aligned direct arm;
   - must reproduce missing `23.2375%`, all `24.09%`, seen `27.50%`, all 10 prototype hashes and aggregate hash exactly.

2. `localtrain_aligned_global_prototype_cosine`
   - local class means from each client's ordinary 200 training examples under its final round10 model;
   - same N256 alignment, count-weighted global aggregation, post-aggregation L2 normalization, direct cosine classifier;
   - no learned parameters, optimizer, calibration set, or extra labeled data.

3. `localtrain_native_global_prototype_cosine_control`
   - exact same local-training means/counts but no cross-client alignment;
   - same count-weighted aggregation and cosine rule in native coordinates.

Do not add current `client.protos`, Euclidean scoring, learned heads, multiple prototypes, normalization before aggregation, pair breaking, new transport, temperature sweeps, seeds1/2, or online loss changes in H07-A.

## Required provenance / correctness receipts

Report and assert:

- exact `split['train_indices']` hash and per-client index hashes used by the new arms;
- exactly 200 ordinary local samples/client, exactly 100 per owned class, and exactly the frozen class sets;
- local-training prototype labels/counts and hashes for all 20 `(client,class)` pairs;
- no test examples or H02-E held-out examples are used in new prototype construction;
- ordinary local train indices remain disjoint from the N256 anchor indices, as required by the existing anchor construction; stop if this is false;
- exact historical N256 transform hashes and anchor provenance are unchanged;
- state/RNG/module-mode/pre-existing-gradient isolation, as in H06-C.

Add a diagnostic only, with no tuning: for each global class, report cosine similarity and L2 distance between the new aligned local-training global mean and the frozen H06-C held-out global mean. This tells us whether any performance change is explained by semantic-prototype source shift.

## Required metrics

For both new arms report:

- seen / missing / all / macro accuracy;
- full per-client/per-class correct/count tables;
- overall / seen / missing prediction histograms and number of predicted classes;
- all 10 global prototype hashes plus aggregate hash;
- global prototype norm range and logit finiteness;
- class -> owner clients, local counts and total count;
- semantic uplink bytes, unchanged N256 anchor-feature uplink bytes, and direct global-prototype downlink bytes.

Also report the number of local forward examples required for this final-state prototype refresh (`2000` total in this frozen split). Do not call that communication overhead; it is local compute.

## Tests

Preserve all existing 39 tests. Add only minimal tests needed to show:

- ordinary-local dataset class means/counts are computed from exactly the provided local dataset and not H02-E;
- aligned local mean aggregation has the same hierarchical mean equivalence property as H06-C;
- the new diagnostic path preserves state/RNG/modes/gradients.

Do not refactor unrelated code.

## Predeclared interpretation

Let the frozen H06-C held-out aligned direct reference be:

- `M_ref = 23.2375%` missing;
- `A_ref = 24.09%` all;
- `S_ref = 27.50%` seen.

Let local-training aligned metrics be `M_train, A_train, S_train`, and local-training native missing be `M_native_train`.

Define:

- `ret_missing = M_train / M_ref`;
- `ret_all = A_train / A_ref`;
- `alignment_gain = M_train - M_native_train` percentage points.

### A. Strong ordinary-local semantic closure

Pass if all are true:

- `ret_missing >= .75`, i.e. `M_train >= 17.428125%`;
- `ret_all >= .80`, i.e. `A_train >= 19.272%`;
- `alignment_gain >= 8pp`;
- at least 8 of 10 classes receive at least one prediction overall.

Then conclude that the extra H02-E labeled semantic set is not necessary for the core direct-readout mechanism. Stop after reporting; do not implement online-loss changes in the same block. The next lead review will decide whether to test the already-produced in-round `client.protos` (zero extra final forward pass), replicate the direct method across seeds1/2, or move into minimal online PPRTP integration.

### B. Fresh held-out semantics are materially important

If `M_train < .50 * M_ref` (`<11.61875%`) or `A_train < .60 * A_ref` (`<14.454%`) without a correctness bug, conclude that H06-C depends materially on the fresh held-out semantic source. Do not rescue with multi-prototypes, head fitting, extra local epochs, or metric changes. Report the prototype-source shift diagnostics and stop.

### C. Intermediate

Otherwise report the complete evidence and stop. Do not tune thresholds/readouts inside H07-A.

## Deliverable

Append `CODEX REPORT H07-A` with STATUS, source SHA, exact commands/run IDs, 39+ tests, exact H06-C reproduction, local-training index/prototype provenance, three-arm table, source-shift diagnostics, per-class/prediction diagnostics, communication/local-compute accounting, integrity receipts, warnings, and interpretation under the preregistered gate.

Do **not** implement online training changes, use online `client.protos` as the primary source, add extra labeled data, learned heads, Euclidean scoring, temperature tuning, multi-prototype clustering, learned transport, PCA, relation kernels, regularization, another optimizer, or seeds1/2 in H07-A. Await research-lead review.


## CODEX REPORT H07-A — DONE (2026-09-18 09:25 +08)

STATUS: DONE; **A: strong ordinary-local semantic closure** under preregistered gate. Source `8a241e689ea06132c739df2d015be608011f01c9`;release `20260918-092203-h07a`;run `20260918-092219-h07a-local-source`,exit0,finished09:23:01+08.

Changed pprtp/local_source.py,pprtp/direct_prototypes.py(optionalnewsource/frozenalignment/bankcapture),pprtp/run.py,tests/test_local_source.py,scripts/report_h07a.py.40tests pass locally/remotely,previous39preserved. New integratedtest checks exactprovidedlocaldataset means/counts/hashes,perclientindexreceipts,source-isolation(noH02E input,deliberatelydifferentonlineprotosunused),hierarchicalequivalence,state/RNG/modes/preexistinggradients,and zero-distance source-shiftidentity. No failedtest/run.

Commands (AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h07a
./scripts/autodl-run.ps1 -Name h07a-local-source -Cmd 'PPRTP_SOURCE_SHA=8a241e689ea06132c739df2d015be608011f01c9 bash scripts/run_h01.sh --modes fedgh --seeds 0 --rounds 10 --local-source-probe'
D:\anaconda3\python.exe scripts/report_h07a.py research_log/H07A/gate
```

CompleteH06Caligneddirectresult assertedexact before newarms,including10globalhashes/aggregatehash,allmetrics/histograms/owners/counts/hierarchy/alignment/communication/state. All10 H02Aonline metrics/model/prototype/server records exact. FrozenPFLlib0169ba7,CIFAR10subset,10clients2classes,100localtrain/class/client,test100/class,CNN512features,SGD.01,1epoch,batch32,seed0round10 unchanged. Exactparentanchors1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125,N256prefix/client0reference/canonicalfloat32appliedProcrustesunchanged.

Newarms use the exact datasets returned byprepare and used inonlineSGD,not reconstructedfromH02E. Indexreceipt assertsexact200/client,100/ownedclass,frozenclasssets;allordinaryindicesdisjointfromH02E andN256anchors. OfficialCIFARtrain/test areseparatesources;testindices are not confusedwithtrainindices sharingintegerIDs. Exactfulltrainindex/perclientindex/datasettensorhashes and2000indices retained. NoH02E images/labels enternewprototypecomputation;theirindexlist is usedonlyfor disjointness checking andtheirdataonlyforthe mandated historicalreference. Newmeans usefeatures() eval-mode/no-grad onfinalround10clientmodel. Onlineclient.protos isonlyhashed forimmutability,neverusedtobuildthe newbank.

20derivedlocalmeans,2/client100sampleseach;rawhashes/counts/labels exactlymatched betweennewaligned/nativearms. Exacthistoricalalignmentreceipt enforced. Countweightedaggregation intotenclasses,postaggregationnormalization,cosineargmax unchanged. Perclasshierarchicalequivalence passes sameH06Ctolerance1e-6/1e-5;alignedworstmax4.76837e-7,Fro1.22051e-6. No learnedhead/optimizer/calibration/newloss.

# H07-A ordinary local semantic source

| Arm | Seen % | Missing % | All % | Macro % | Predicted classes |
|---|---:|---:|---:|---:|---:|
| heldout_aligned_global_prototype_cosine_reference | 27.500000 | 23.237500 | 24.090000 | 24.090000 | 10 |
| localtrain_aligned_global_prototype_cosine | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| localtrain_native_global_prototype_cosine_control | 54.099999 | 0.000000 | 10.820000 | 10.820000 | 9 |

Frozen verdict: A: strong ordinary-local semantic closure. ret_missing=0.957504049; ret_all=0.97094229; alignment_gain=22.250000pp.

localtrain_aligned_global_prototype_cosine:
Aggregate SHA256: c3d0f35e1b61d066185b89f1e12558aa4a51d3f86f21fc2e274930d5852d1695
Norm range: 12.4210386 to 14.7693615
Prediction histograms (class0..9): {"overall": [1109, 1429, 1036, 367, 1113, 1532, 1823, 207, 640, 744], "seen": [232, 266, 229, 84, 213, 332, 337, 48, 117, 142], "missing": [877, 1163, 807, 283, 900, 1200, 1486, 159, 523, 602]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |
|---|---|---|---:|---:|---:|---|
| 0 | [6, 7] | [100, 100] | 200 | 4.76837158e-07 | 1.04314461e-06 | 1782cae65657f4d1f2513b49967511831691400e85fc30d78fe7c2993ac0bbd0 |
| 1 | [8, 9] | [100, 100] | 200 | 2.38418579e-07 | 1.02717377e-06 | dfe7b52e785fda6fe96b418b705f1d5bc9f00ddfcd779b44abcc4fff5de3eb73 |
| 2 | [1, 2] | [100, 100] | 200 | 2.38418579e-07 | 9.32140892e-07 | 5d7cfa00b1a61697a2ac57b7933dcf129746a4e4f030264b1d8fb89e1d0f0668 |
| 3 | [3, 4] | [100, 100] | 200 | 3.57627869e-07 | 1.14584122e-06 | 119e8d9a63a0f7e6390f75ad05afc557fa97ebd81231fa8c272b24318425478d |
| 4 | [0, 9] | [100, 100] | 200 | 2.38418579e-07 | 8.46350019e-07 | da586ee1df90418966dd81712af4f407750331d8eeca7a409b132d22d6bc0b47 |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07 | 9.63432285e-07 | a414080a39d4cbae314ffee649b9bfd54220fd8b01b6ebb382b2b140c9d39303 |
| 6 | [0, 1] | [100, 100] | 200 | 4.76837158e-07 | 1.22051335e-06 | be5f336db0a09d8b9002c8f043141031d93758826390bd5fe6c8fc4121df5a5b |
| 7 | [2, 3] | [100, 100] | 200 | 2.38418579e-07 | 1.08478116e-06 | 39c8958d44c020298e9ade6fd20a86e0f2ba78a71384cdb4cb3cb16e43b1d82c |
| 8 | [7, 8] | [100, 100] | 200 | 4.76837158e-07 | 1.12895168e-06 | 214a44eaaf80b28a853ab9c352f73021fa276426efb6790cf6a7e65c00143485 |
| 9 | [5, 6] | [100, 100] | 200 | 2.38418579e-07 | 9.80289997e-07 | 27e3eb4f746d1718415ee472e30b2bf10c82e552d2ba51131249940b798ed042 |

localtrain_native_global_prototype_cosine_control:
Aggregate SHA256: 82159cd36fc2376abff996f3a597cc031b12e47f8463daf951b590f10a7ce3e3
Norm range: 11.6439543 to 15.0607529
Prediction histograms (class0..9): {"overall": [1842, 1929, 2000, 1982, 1037, 553, 1, 0, 33, 623], "seen": [384, 390, 400, 390, 205, 113, 1, 0, 4, 113], "missing": [1458, 1539, 1600, 1592, 832, 440, 0, 0, 29, 510]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 0, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |
|---|---|---|---:|---:|---:|---|
| 0 | [6, 7] | [100, 100] | 200 | 3.57627869e-07 | 8.85221311e-07 | 969bfe5847a5885cb8b1c1b466c61352c9b1bf82d149efb23b0aa81b8059f1a3 |
| 1 | [8, 9] | [100, 100] | 200 | 3.57627869e-07 | 1.11544318e-06 | dbfcbcd536b0083c9fcd67dfcecd6dbd50435d2b0836bb6e92feda8d59facdb7 |
| 2 | [1, 2] | [100, 100] | 200 | 2.38418579e-07 | 8.65154448e-07 | 7f0c3d71f1cfca8f8d67c175fd81b55633fb0f90eb4188073497551edbee6a74 |
| 3 | [3, 4] | [100, 100] | 200 | 2.38418579e-07 | 1.00104e-06 | 493aae11782a9fd4da6ac08e452d9ac9f52feecd2c850a77b93cb591b576a3f1 |
| 4 | [0, 9] | [100, 100] | 200 | 3.57627869e-07 | 9.26148289e-07 | 6fbe042a078b9c69d4911b92510ae6892eec76458e4fb2ad8bb25c09c4b04a86 |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07 | 8.77466675e-07 | 24b08904adcf492b2d16900c182867a9b5107ac7afbaa8f3ceb76c14a5dce851 |
| 6 | [0, 1] | [100, 100] | 200 | 4.76837158e-07 | 1.01596265e-06 | 3b483596852487c38ede8fed150a38c2267646b99db45f86d04545ea17ad7449 |
| 7 | [2, 3] | [100, 100] | 200 | 4.76837158e-07 | 9.64835863e-07 | 17659f0496538431335a51a35fc912ba102c81012ecbc85f66062d51a6129353 |
| 8 | [7, 8] | [100, 100] | 200 | 3.57627869e-07 | 9.64718424e-07 | 84a279b53ff4cc1fe9ab6a67a41202b34eecb037f285e14af8eb5e24bfeb9632 |
| 9 | [5, 6] | [100, 100] | 200 | 3.57627869e-07 | 9.97603706e-07 | 2cfc54e6aa55d4f23fa4d6be38d0c575230928d27dcbe5f0d7c1629ccf581b16 |

Ordinary local train index SHA256: 5fbbd599df082e9f32630e9f65fc1e41aea4931f1ffe6957b022c2cfd56847a5
Perclient index SHA256: ["817fe99e3085d6924872ae39989466dbf9b3b63538c0e3120227460c73754b43", "ecb6f2a84c4c598e3fa2bcc7f38d8fbf982ae16d140b8e9bf2e70df8be191123", "e90b4bc94e79e7dc2aefa3a50a5da8b1fa5f1e4a1264494cf639f7fa6217990e", "699994b118d8217dbe65f598bd508de262437bcd841db81997dd66f4e75792a3", "a129cbaedc10d7e7c046855c58d44d040830b9c20ad8368690d53259531a4238", "7eb0c04dfcd9d1ac01f0db02d4df7f6d69e01351e9dd9420351185be72eec7fa", "bb2b19994bf19bd85405cf937464f0a77917eb92274ba111422996a08d41908c", "9fcb02324f1c2312bae0ef3b89d09d206b26ea947c199eb2dea3d91898c04508", "73ef4e51785dc3dfcfb269c2ec396282345741da603a390389ad76ef63e3f5b6", "8d944777dfe49ed996f17e6e7c879c0dca4974636e22a4c053e2feaf391bbb03"]
Prototype refresh forward examples: 2000 total; [200, 200, 200, 200, 200, 200, 200, 200, 200, 200] perclient. Local compute, not communication. Each diagnostic arm recomputes this same refresh.

| Class | New-vs-heldout cosine | L2 distance |
|---|---:|---:|
| 0 | 0.999792457 | 0.40283376 |
| 1 | 0.999921918 | 0.324709892 |
| 2 | 0.999927461 | 0.151780576 |
| 3 | 0.999920905 | 0.39572075 |
| 4 | 0.999927282 | 1.00032175 |
| 5 | 0.999955058 | 0.402224243 |
| 6 | 0.999985576 | 0.150540978 |
| 7 | 0.999943197 | 0.154670596 |
| 8 | 0.999676466 | 0.629140198 |
| 9 | 0.999951839 | 0.272004515 |


Interpretation: ret_missing.957504049>=.75,ret_all.970942290>=.80,alignmentgain22.25pp>=8,and10predictedclasses>=8 =>stronggatepasses. RemovingfreshH02Esemantics costs only.9875ppmissing/.70ppall inthissetting;ordinarylocaldata sufficesforcorefixedreadoutmechanism. Perclassglobalprototypecosines.999676–.999986 andL2.150541–1.000322 show modestdirectionalshift;these arediagnostic,notselectedortuned. No claimaboutonlineadaptationperformance/multiseedrobustness beyondthisseed0round10closure.

Localcompute accounting: one final-state prototype refresh needs200examples/client,2000total additionalforwardexamples,withnogradients/SGD. Thediagnostic executes2newarms independently,therefore4000ordinaryexamplesforwardedacrossthe pair;2000is perrefresh/perproposedreadout. Heldoutreferenceforwards2000heldoutexamples solelyforreproduction. Anchorandtestforwardsareseparate existingdiagnosticcosts. Thisextra pass islocalcompute,notcommunication. Semanticuplink41280Btotal(4128/client),anchors5242880B(524288/client),combined5284160B;globaldownlink20480B/client,204800Btotal,fixedclassorder. Rawanchorimages,transformdistribution/framing/amortization remainuncounted;noendtoendcommunicationclaim.

Integrity:allparameters/buffers/prototypes/serverstate,modulemodes,CPU/CUDA RNG,preexistinggradients unchanged;allthree arms samefrozenstate. Anchorlabelsunused,testevaluationonly,allfeatures/means/norms/logitsfinite. KnownNVMLinitializationwarningsonly,noSVDwarning/failure. Fullperclient/perclasscounts,predictionhistograms,all20local/10globalhashes,owner/counts,norms,source-shiftdiagnostics,andlocalindexprovenance retained.

Evidence `research_log/H07A/gate/`:RESULTS.md,verification.json,rawfinal/rounds/provenance/split/indices/tests/meta/run/log/compacttar. Remoteoriginals/checkpoints `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-092219-h07a-local-source`.

Stopawaitleadreview. Noonlineclient.protosprimarysource,extra labeleddata,newlearnedhead,metric/temperaturechanges,multiprototypes,otherseedsoronlineintegrationimplemented. Nextalready-produced-prototype/crossseed/onlineexperimentrequiresleadassignment.
