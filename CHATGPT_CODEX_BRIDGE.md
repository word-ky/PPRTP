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
