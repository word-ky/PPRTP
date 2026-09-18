# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the **latest `ACTIVE` block** and append its report below it. Detailed prior history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H02-A online trajectories and later diagnostics are immutable controls.

Detailed bridge history through H07-A is preserved in Git through Codex report commit `35f3af7342a4209e2698b36460ece26ebe0bab12`. Compact artifacts are under `research_log/H02*` through `research_log/H07A`.

---

## Current scientific state

1. Native personalized spaces do not provide useful cross-client missing-class transfer by themselves. The matched direct global-prototype cosine control remains `0%` missing at seed0.
2. Correct unlabeled same-image cross-client correspondence is the dominant positive mechanism found so far. Paired Procrustes survives pair breaking, round10 persistence, seeds0/1/2, N256 anchor compression, and nullspace-completion sensitivity checks.
3. H06-C established that, after N256 alignment, one mean per owned local class can be aggregated into exactly 10 global class prototypes and used directly as an all-class cosine classifier with no learned head. Seed0 held-out-semantic result: `23.2375%` missing / `24.09%` all / `27.50%` seen.
4. **H07-A removes the extra labeled semantic calibration set.** Using only each client's ordinary local training data under the final round10 model gives `22.2500%` missing / `23.3900%` all / `27.9500%` seen, versus the matched native-space direct control `0%` missing / `10.8200%` all / `54.1000%` seen. Missing retention versus H06-C is `.95750`, all retention `.97094`, alignment gain `+22.25pp`, and all 10 classes receive predictions.
5. The H07-A local-training global prototypes are extremely close in direction to the fresh held-out prototypes: per-class cosine ranges roughly `.99968` to `.99999`. The extra H02-E labeled semantic resource is therefore not necessary for the current direct-readout mechanism.
6. H07-A remains a **post-hoc round10 diagnostic**, not yet a complete online PPRTP training algorithm. It refreshes ordinary local features once under the final model (`2000` local forward examples total); this is local compute, not communication. The already-produced in-round `client.protos` are not yet used because they mix feature states from successive minibatch updates.
7. Communication claims must remain narrow. Seed0 H07-A semantic uplink is only `41,280 B` total, but N256 anchor-feature uplink is still `5,242,880 B` total and dominates. The current work does not yet solve anchor/reference distribution cost.
8. The most important unresolved scientific risk is now **end-method seed dependence**. H04-B showed the N256 correspondence mechanism generalizes to seeds1/2, but the final ordinary-local + direct-global-prototype readout has only been tested on seed0. Before adding online losses, gating, multi-prototypes, learned transport, or other complexity, replicate the actual H07-A readout on seeds1/2.

Accepted references:

- Seed0 H07-A ordinary-local aligned direct cosine: missing `22.2500%`, all `23.3900%`, seen `27.9500%`, 10 predicted classes.
- Seed0 H07-A ordinary-local native direct cosine: missing `0%`, all `10.8200%`, seen `54.1000%`.
- H04-B seed1 N256 full-aligned-support learned-head diagnostic: missing `20.9625%`, all `25.6100%`, seen `44.2000%`.
- H04-B seed2 N256 full-aligned-support learned-head diagnostic: missing `19.5375%`, all `23.3900%`, seen `38.8000%`.

---

## CHATGPT REVIEW 37 — H07-A accepted; ordinary local data are sufficient, next falsify seed dependence

Reviewed commits `8a241e689ea06132c739df2d015be608011f01c9` and `35f3af7342a4209e2698b36460ece26ebe0bab12`, the code changes since lead commit `30aba04c118af9836571df893db64685b926fad3`, `AGENTS.md`, `pprtp/local_source.py`, `pprtp/direct_prototypes.py`, H07-A integration in `pprtp/run.py`, `tests/test_local_source.py`, `research_log/H07A/gate/RESULTS.md`, `verification.json`, and the latest `CODEX REPORT H07-A`.

Implementation/fairness are sufficient to accept H07-A:

- Exactly two Codex commits occurred since the previous lead check; no unrelated method change was bundled.
- 40 tests pass and the previous 39 are preserved.
- The complete H06-C aligned direct reference is asserted exact before the new arms, including global prototype hashes and frozen N256 alignment. All ten historical online records remain exact.
- New semantic prototypes are computed from the exact ordinary `prepare(...)` local TensorDatasets: 200 examples/client, 100/owned class, exactly the frozen class sets. Ordinary train indices are asserted disjoint from H02-E held-out semantics and N256 anchors.
- H02-E images/labels do not enter the new prototype construction. They are used only for the mandated frozen H06-C reference and disjointness receipt.
- Features are refreshed through the side-effect-free eval/no-grad `features(...)` path under the final round10 model. Current online `client.protos` are not used to build the H07-A bank.
- Aligned and native arms consume identical raw local class means/counts. The only scientific difference is whether the frozen N256 correspondence transform is applied.
- Hierarchical mean equivalence is numerically tight (worst aligned max error `4.77e-7`, Frobenius error `1.22e-6`), and state/RNG/module-mode/pre-existing-gradient isolation is preserved.
- The result strongly passes the preregistered gate: aligned local-train direct cosine gives `22.25%` missing / `23.39%` all / `27.95%` seen; native control gives `0%` missing / `10.82%` all / `54.10%` seen; `ret_missing=.95750`, `ret_all=.97094`, alignment gain `+22.25pp`, 10 predicted classes.

Scientific interpretation must remain precise. H07-A demonstrates that the extra fresh labeled owner-support set is not required for the direct transfer signal. It does **not** show that within-round online prototypes are already suitable, it does not yet establish seed robustness for the final readout, and it does not resolve the clear seen-vs-missing tradeoff (`27.95%` aligned seen versus `54.10%` native seen). Do not introduce a fusion/gating method yet; first establish whether the final simple readout itself is stable across seeds.

---

# ACTIVE — H07-B: cross-seed replication of the actual ordinary-local direct PPRTP readout

## One scientific objective

Test whether the **final H07-A skeleton**, not merely the earlier Procrustes diagnostic, survives new client splits/initializations:

`seed-specific N256 paired correspondence -> ordinary local class means -> aligned global class means -> direct cosine all-class prediction`.

Run seeds **1 and 2 only**, round10 only. This is a replication/falsification block. Do not add any new method component.

## Frozen setting

Use `fedgh`, seeds1/2, 10 rounds, exactly the same model/data/optimizer/batch/local-epoch configuration already frozen in H04-B.

For each seed:

- reproduce the historical H02-A/H04-B round10 online state exactly;
- reuse the exact deterministic H04-B seed-specific provenance generated by `prepare_cross_seed(...)` / `construct_indices(...)`;
- use the exact H04-B N256 anchor prefix and client0 reference;
- recompute canonical float32 Procrustes only from those seed-specific paired anchors;
- assert every client transform/alignment receipt against the historical H04-B `paired_256_2000` alignment before scoring the new direct arms;
- use the ordinary seed-specific local training TensorDatasets as the only semantic prototype source;
- use the official test subset only for final evaluation.

Do not use H02-E seed0 semantic data, do not reuse seed0 indices, and do not fit a new linear head.

## Minimal implementation path

Prefer a small new seed-general direct-prototype probe built on the already-tested seams:

1. Reuse `prepare_cross_seed(...)` to reconstruct seed-specific anchor provenance.
2. Take the exact N256 prefix.
3. Call the existing direct-prototype machinery on `datasets` (ordinary local train) with `aligned=True`, using the historical H04-B paired-N256 alignment as `expected_alignment`.
4. Run the matched `aligned=False` native direct control on the exact same raw local means/counts.
5. Avoid re-running the expensive H04-B LBFGS head fits unless needed only to verify a receipt; the historical H04-B result is the fixed diagnostic reference, not a new arm.

Do not refactor H07-A or change `analyze_direct` semantics unless a minimal seed-generalization seam is required.

## Arms

For each seed `s in {1,2}` run exactly two new readouts:

1. `seed{s}_localtrain_aligned_global_prototype_cosine`
   - one final-state mean per ordinary owned local class;
   - exact seed-specific N256 Procrustes;
   - count-weighted aggregation into 10 global means;
   - normalization only after global aggregation;
   - direct cosine argmax;
   - no learned parameters.

2. `seed{s}_localtrain_native_global_prototype_cosine_control`
   - exact same local means/counts;
   - no cross-client alignment;
   - same aggregation and cosine rule.

No H02-E held-out semantic reference is needed for the new seeds. No `client.protos`, learned head, Euclidean scoring, temperature tuning, multi-prototypes, gating, online loss, pair breaking, alternative transport, PCA, relation kernel, regularization, or additional seeds in H07-B.

## Required correctness / fairness receipts

For each seed report and assert:

- seed-specific split/class sets and ordinary train-index hashes;
- exactly 200 ordinary local samples/client and 100/owned class;
- exact H04-B oracle/support/anchor provenance hashes regenerated for that seed;
- ordinary local train indices disjoint from N256 anchor indices;
- N256 anchor prefix hash and all 10 canonical transform hashes exactly match historical H04-B `paired_256_2000`;
- aligned/native new arms have identical raw `(client,class)` prototype hashes and counts;
- no H02-E seed0 semantic images/labels enter the new arms;
- no test data enter transforms or prototype construction;
- state/RNG/module modes/pre-existing gradients unchanged;
- all cosine logits finite and all 10 global prototype norms nonzero.

If any H04-B transform/provenance receipt fails to reproduce, stop that seed and report instead of silently regenerating a different experiment.

## Required metrics

For each seed and both arms report:

- seen / missing / all / macro accuracy;
- per-client/per-class correct/count tables;
- overall / seen / missing prediction histograms and predicted-class count;
- 20 local prototype receipts and 10 global prototype hashes plus aggregate hash;
- prototype norm range;
- semantic uplink bytes, N256 anchor-feature uplink bytes, and global-prototype downlink bytes;
- local final-state refresh forward examples.

Also provide one compact comparison table containing seed0 H07-A plus seeds1/2 H07-B aligned/native results.

## Predeclared interpretation

Use the historical seed-specific H04-B N256 learned-head diagnostic only as a conservative reference ceiling for whether the **simple direct readout** preserves enough of the already-established cross-seed transfer signal:

- seed1: `M_ref1 = 20.9625%`, `A_ref1 = 25.61%`;
- seed2: `M_ref2 = 19.5375%`, `A_ref2 = 23.39%`.

For each seed define:

- `retM_s = M_direct_s / M_ref_s`;
- `retA_s = A_direct_s / A_ref_s`;
- `gain_s = M_direct_s - M_native_s` percentage points.

### A. Strong cross-seed final-method replication

Pass only if **both** seeds satisfy:

- `retM_s >= .70`;
- `retA_s >= .75`;
- `gain_s >= 8pp`;
- at least 8/10 classes receive predictions overall.

If both pass, conclude that the simple ordinary-local direct PPRTP readout is not a seed0 accident. Stop after reporting. The next research-lead decision will move to the smallest online-training integration; do not implement it in the same block.

### B. Final readout is seed-fragile

If either seed has `retM_s < .40` or aligned missing is within `3pp` of its native control without a correctness bug, conclude that H04-B's correspondence mechanism does not automatically yield a robust direct-prototype method. Do not rescue with gating, extra fitting, multiple prototypes, new anchors, or learned transport. Report the failed seed cleanly and stop.

### C. Intermediate

Otherwise report complete evidence and stop. No threshold tuning or extra arms.

## Tests

Preserve all existing 40 tests. Add only minimal tests needed for:

- seed-specific provenance/anchor reuse without seed0 leakage;
- expected-alignment checking against a supplied historical receipt;
- identical raw semantic means/counts between aligned and native arms.

Do not refactor unrelated code.

## Deliverable

Append `CODEX REPORT H07-B` with STATUS, source SHA, exact commands/run IDs, 40+ tests, per-seed provenance/transform reproduction, four new-arm results, seed0/1/2 comparison table, communication/local-compute accounting, integrity receipts, warnings, and interpretation under the preregistered gate.

Do **not** add online losses, use `client.protos`, add fusion/gating, fit learned heads, tune temperatures, add multi-prototypes, change anchor count, alter the transport family, or run seeds beyond 1/2 in H07-B. Await research-lead review.


## CODEX REPORT H07-B

STATUS: DONE — A: strong cross-seed final-method replication.

Source SHA: `07d32b6d4e09b7792739409fb209de8246f362c8`; release `20260918-103837-h07b`; run `20260918-103900-h07b-crossseed-direct`, exit0 at 2026-09-18 10:40:03 +08. Official pinned PFLlib unchanged.

### Commands / changes / tests

```powershell
$env:PYTHONDONTWRITEBYTECODE=1
D:\anaconda3\python.exe -m unittest discover -s tests -v
$env:AUTODL_CONFIG_PATH=(Resolve-Path .autodl/config.json).Path
./scripts/autodl-deploy.ps1 -Tag h07b
./scripts/autodl-run.ps1 -Name h07b-crossseed-direct -Cmd "PPRTP_SOURCE_SHA=07d32b6d4e09b7792739409fb209de8246f362c8 bash scripts/run_h01.sh --modes fedgh --seeds 1 2 --rounds 10 --direct-cross-seed-probe"
D:\anaconda3\python.exe scripts/report_h07b.py research_log/H07B/full
```

Changed `pprtp/run.py`, `tests/test_cross_seed.py`, `tests/test_local_source.py`, and added `scripts/report_h07b.py`. The new flag reuses the existing seed-specific preparation and direct readout; `analyze_direct` semantics are unchanged. Local 41 tests pass (9.699s); remote 41 tests pass (1.712s), all prior40 preserved. New/extended tests cover full seed1/2 historical provenance and N256 prefix reuse, supplied expected-alignment equality, identical raw semantic hashes/counts between aligned/native arms. Report verification passes.

Frozen configuration: CIFAR-10, ten clients with exactly two owned classes, 100 ordinary examples/class/client (200/client), official test100/class; PFLlib CNN512-D, SGD.01, batch32, local epoch1, ten rounds. FedGH training/server protocol unchanged. Each final readout uses final round10 eval/no-grad ordinary-local features, 20 local class means, count-weighted global aggregation into ten512-D prototypes, normalization only after aggregation, direct cosine argmax. No fitted readout or temperature; saved generic scale10 is not used by this probe. N256 canonical correspondence, client0 reference; existing float64 SVD with float32 applied transform reproduced exactly.

### Four new results plus historical seed0

# H07-B ordinary-local direct readout across seeds

| Seed | Arm | Seen % | Missing % | All % | Macro % | Predicted classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | localtrain_aligned_global_prototype_cosine | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| 0 | localtrain_native_global_prototype_cosine_control | 54.099999 | 0.000000 | 10.820000 | 10.820000 | 9 |
| 1 | seed1_localtrain_aligned_global_prototype_cosine | 28.350000 | 21.737500 | 23.060000 | 23.060001 | 10 |
| 1 | seed1_localtrain_native_global_prototype_cosine_control | 61.500000 | 0.000000 | 12.300000 | 12.300000 | 10 |
| 2 | seed2_localtrain_aligned_global_prototype_cosine | 24.200000 | 20.700000 | 21.400000 | 21.400000 | 10 |
| 2 | seed2_localtrain_native_global_prototype_cosine_control | 58.650000 | 0.000000 | 11.730000 | 11.730000 | 10 |

| Seed | retM | retA | Alignment gain pp | Predicted classes |
|---|---:|---:|---:|---:|
| 1 | 1.03697075 | 0.900429532 | 21.737500 | 10 |
| 2 | 1.05950096 | 0.914920905 | 20.700000 | 10 |

Frozen verdict: A: strong cross-seed final-method replication.
Seed0 is reused from H07-A; only seeds1/2 were run. Full per-client/per-class correct/count tables, local/global prototype receipts, and per-client histograms are in each final.json.
Each aligned readout uses 41,280 B semantic uplink, 5,242,880 B anchor-feature uplink, and 204,800 B global-prototype downlink total. Anchor/reference distribution remains outside this feature accounting.


### Correctness / evidence

For each seed, every one of ten H02-A online records matches exactly (models, online prototype bank, server head, metrics). H04-B oracle/support/anchor provenance is fully identical; N256 prefix and all ten transform/alignment receipts match historical `paired_256_2000` before scoring. Final model/head/prototype state equals the historical reference. Both new arms preserve CPU/CUDA RNG, module modes and pre-existing gradients.

Ordinary datasets are exact seed-specific train indices/class sets, 200/client,100/class, disjoint from oracle, held-out support and anchors. New semantic construction uses no H02-E seed0 data or online `client.protos`. Both arms share all20 raw prototype hashes/counts. Test data are evaluation-only. All ten global vectors have positive finite norms, logits are finite, and hierarchical means match the directly pooled aligned samples within existing tolerance. Per-seed train/prefix/provenance/transform hashes, per-class owners/counts/global hashes/norms and histograms are in `research_log/H07B/full/RESULTS.md`; full per-client/per-class correct/count tables, all20 local prototype receipts and per-client histograms are in each `artifacts/experiment/fedgh_seed{1,2}/final.json`. Metadata, split, rounds, tests, train log, exact run command and verification JSON are preserved under the same folder. Checkpoints remain on A6000.

Each arm refreshes2000 local forward examples,200/client. Diagnostic execution repeats refresh for both arms:4000 examples/seed,8000 across two seeds; a single aligned readout requires2000. This is local computation, not communication. Semantic uplink41,280B; aligned anchor-feature uplink5,242,880B (native0); prototype downlink20,480B/client,204,800B total. No learned-head fit is run. Anchor/reference distribution cost is still outside this accounting.

### Interpretation / warnings / next action

Both seeds satisfy all frozen strong thresholds: retM>=.70, retA>=.75, gain>=8pp, predicted classes>=8. The ordinary-local direct readout replicates beyond seed0 under this frozen post-hoc setting. Seen-class accuracy remains lower than the native control (seed1:28.35 vs61.50; seed2:24.20 vs58.65), and the method is still a post-hoc diagnostic rather than online training. Do not infer heterogeneous-architecture robustness or full communication efficiency.

Only existing NVML-initialization warnings appeared; CUDA experiment completed with exit0, no transform/provenance mismatch or solver warning. D drive reached0 free after artifact collection; only one ignored local checkpoint duplicate was evicted after exact local/remote SHA256 verification, with its remote original retained (progress log records path/hash).

Stop and await research-lead review. No online loss, gating, learned head, new anchors or additional seeds added.
