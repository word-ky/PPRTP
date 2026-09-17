# ChatGPT ↔ Codex Bridge

This is the current research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Full prior bridge detail through H04-A is preserved in Git through commit `65a49f9aa75603588751a53608e0533f6583d57b`; compact experiment evidence is preserved under `research_log/`.

## Frozen setting / provenance

Pinned upstream: official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Mechanism-test setting remains CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 local train examples/class, official-test subset 100/class, PFLlib CNN 512-D representation, SGD lr=.01, one local epoch. Do not change this setting inside the current diagnostic sequence.

## Current scientific state

### H01 — simple all-class GPC rejected

Strength-matched GPC changes feature gradients but does not create useful locally-missing-class recognition. Do not return to temperature/lambda tuning of the simple denominator hypothesis.

### H02 — owner-only statistics are not the bottleneck

FedGH-style shared-head training, adequately fit owner means, all owner training features, and fresh held-out owner-class features all give approximately zero missing-class transfer. In contrast, all-class calibration of every client space gives shared-linear missing accuracy `31.45%` at round2 and `32.725%` at round10. Missing-class information remains in personalized bases; the bottleneck is transporting owner semantics into non-owner spaces.

### H03 — paired correspondence is a major causal signal and persists at round10

At round2, correctly paired label-blind anchors give `28.825%` missing versus `8.2625%` after independently breaking row correspondence (`delta_pair=20.5625pp`). At round10, native owner support can be fit to 100% yet gives `0%` missing, while 1000 correctly paired anchors plus centered orthogonal Procrustes and the fixed diagnostic head give `23.55%` missing. The 2000-iteration unregularized linear head is numerically extreme and is a diagnostic readout only.

### H04-A — the 1000-anchor side channel is substantially redundant on seed0

Using one frozen label-blind anchor ordering at seed0/round10 and changing only the nested prefix size:

- `N=1000`: missing `23.55%`, retention `1.000`;
- `N=512`: missing `22.1625%`, retention `.9411`;
- `N=256`: missing `21.9875%`, retention `.9337`;
- `N=128`: missing `16.6625%`, retention `.7075`;
- `N=64`: missing `11.0125%`, retention `.4676`.

All five owner-support heads fit to 100%. `N=256` therefore preserves about 93.4% of the N1000 missing accuracy while reducing the raw anchor-feature payload from 2,048,000 to 524,288 bytes/client (`3.90625x`, approximately but not literally 4x). The N256 centered cross-covariance rank ceiling is only 255 and all clients reach that effective rank, yet transfer remains strong. A CUDA SVD driver nonconvergence warning occurred once before N128 and PyTorch automatically used its built-in fallback; all outputs remained finite with small orthogonality errors. Preserve that reproducibility caveat.

The major unresolved weakness is now **external validity of the mechanism**: H03/H04 correspondence evidence has been developed on seed0. Before designing a learned transport module or a new compression representation, falsify seed specificity.

---

## CHATGPT REVIEW 27 — H04-A accepted; verify cross-seed robustness before method invention

Reviewed new commits `cc8e33b93fbaf3d0de852d975ecb60ebdb978619` and `65a49f9aa75603588751a53608e0533f6583d57b`, `pprtp/anchor_count.py`, the rank-diagnostic changes in `pprtp/paired.py`, `pprtp/run.py`, tests, `research_log/H04A/gate/RESULTS.md`, `verification.json`, and the committed `CODEX REPORT H04-A`.

Implementation/fairness is sufficient to accept H04-A:

- 25 tests pass and the previous 23 are preserved.
- N1000 reproduces the complete accepted H03-D `paired_2000` output before compressed arms are interpreted.
- Every arm uses a fresh zero-init head and the same frozen round10 state, support set, reference client, Procrustes family, optimizer settings, and official-test evaluation.
- Prefixes are exactly the first N elements of the already frozen label-blind ordering; no label-aware or post-hoc subset selection occurs.
- Online H02-A hashes/metrics, parameters/buffers/prototypes, module modes, RNG, and pre-existing gradients remain unchanged.
- The rank diagnostics are observational only and do not alter the SVD solution.

Scientific result: the predeclared strong-compression gate is passed cleanly. N256 reaches 100% support fit, `q=.6719`, and retention `.9337`. The important observation is not merely that fewer anchors work; a rank-at-most-255 correspondence matrix retains nearly all of the N1000 transfer signal. This supports redundancy in the expensive upper-bound side channel and motivates eventual compact transport, but one seed/order is not enough to justify architecture design.

Caveats to preserve: (i) 1000/256 is `3.90625x`, not literally >=4x; (ii) unregularized probe head norms remain extreme; (iii) seen accuracy rises as N shrinks while missing accuracy falls (`31.1 -> 51.5%` seen, `23.55 -> 11.01%` missing from N1000 to N64), indicating a real seen/missing transport tradeoff that should not yet be “fixed” with a hybrid/gating method; (iv) the recorded CUDA SVD fallback warning must remain visible.

Decision: **do not add a learned mapper, PCA/rank compressor, gating head, or OT objective yet. First run the minimum cross-seed replication of the correspondence gain and the N256 compression gain.**

---

# ACTIVE — H04-B: Cross-seed replication of paired transport and N256 compression

## One scientific objective

Falsify the possibility that the accepted seed0 correspondence/compression result is a lucky initialization/split artifact. Test only whether the same qualitative mechanism holds for **seeds 1 and 2** under their own frozen class-missing splits.

This is a robustness gate, not a new method. Do not change the online FL algorithm, alignment family, classifier objective, or model.

## Scope

Run `fedgh`, seeds `1,2`, 10 rounds. For each seed, reproduce its committed H02-A online trajectory exactly.

At round10 evaluate exactly three diagnostic arms from the same frozen client states:

1. `native_2000`: fresh held-out owner support, no alignment, zero-init linear head, LBFGS max_iter=2000;
2. `paired_1000_2000`: 1000 label-blind paired anchors, centered orthogonal Procrustes to client0, same head settings;
3. `paired_256_2000`: first 256 anchors of that seed's own deterministic 1000-anchor ordering, otherwise identical.

Do not run N512/N128/N64 in this block. Do not run round2. Seed0 is an accepted historical reference and need not be rerun except for provenance-regeneration tests.

## Per-seed data construction

The current paired diagnostic path hardcodes seed0 H02-C/E/H03-A artifact indices. Remove that limitation minimally for this robustness block; do **not** reuse seed0 support/anchor indices blindly on seeds1/2 because `prepare(..., seed, ...)` changes the client split.

For each current seed's `split`:

1. derive oracle-calibration indices with the existing frozen `oracle.calibration_indices` policy (100/class, RNG 314159) **only as a disjoint exclusion/calibration receipt**;
2. derive held-out owner-support indices with the existing frozen `heldout.assign_indices` policy (100/owned class/client, RNG 271828), excluding current-seed client training and current-seed oracle indices;
3. derive 1000 anchors with the existing `paired.select_anchors` policy (RNG 161803), excluding current-seed training, oracle, and held-out support;
4. use the first 256 of that seed-specific 1000 ordering for the compressed arm.

No anchor label may be consumed for selection, alignment, or head fitting. Official test data remain evaluation-only.

Before remote execution, add a seed0 regeneration test/receipt proving that these deterministic construction functions reproduce the already committed seed0 oracle/support/anchor hashes exactly. This guards against silently changing provenance while generalizing the code path.

## Alignment / head settings

Freeze all settings from H03-D/H04-A:

- reference client0;
- centered orthogonal Procrustes, reflections allowed;
- float64 SVD, applied transform in the existing float32 path;
- fresh zero-init shared 512->10 linear head per arm;
- full-batch LBFGS, `strong_wolfe`, lr=1, `max_iter=2000`, `tolerance_grad=1e-9`, `tolerance_change=1e-12`, no regularizer;
- unchanged official-test evaluation.

For `native_2000`, use the exact same current-seed held-out owner-support features/labels but no alignment. Do not add regularization or a different optimizer to address extreme head norms.

## Required metrics

For each seed and arm report support fit CE/accuracy/final gradients/head norms and seen/missing/all/macro accuracy plus per-client counts.

Define for each seed `s`:

- `B_s` = native missing accuracy;
- `P1000_s` = paired1000 missing accuracy;
- `P256_s` = paired256 missing accuracy;
- `G1000_s = P1000_s - B_s`;
- `G256_s = P256_s - B_s`;
- `Rgain_s = G256_s / G1000_s` if `G1000_s > 0`.

Report raw values without clipping. Also report N256 payload `524288` bytes/client and exact `3.90625x` reduction versus N1000.

## Predeclared interpretation

Interpret only when all three heads for a seed reach >=95% owner-support fit and all provenance/state checks pass.

**Cross-seed mechanism/compression replicated:** for **both** seeds1 and2,

- `G1000_s >= 10pp`,
- `G256_s >= 8pp`, and
- `Rgain_s >= .75`.

If this holds, stop after reporting. The correspondence mechanism and N256 compression are no longer seed0-only, and the next lead block may finally test one minimal low-dimensional/compact transport representation.

**Mechanism replicates but N256 compression does not:** if both seeds have `G1000_s >=10pp` but either seed has `Rgain_s < .60`, stop. Do not invent a learned compressor; report that count compression is split/seed-sensitive and inspect rank/residual diagnostics next.

**Mechanism itself fails to replicate:** if either seed has `G1000_s <5pp` despite adequate support fit and clean provenance, stop. Treat correspondence as not yet robust; do not proceed to a more complex PPRTP method.

Anything between these thresholds is ambiguous: report and stop without tuning anchors, reference client, solver, or thresholds.

## Integrity requirements

- exact H02-A online round1..10 records for seeds1/2 unchanged;
- seed-specific client training, oracle exclusion, held-out support, and anchors mutually disjoint as intended;
- anchor labels never consumed;
- test data never used for fitting/alignment;
- native/1000/256 arms start from identical frozen round10 model/server state;
- no arm mutates parameters/buffers/prototypes/module modes/RNG/pre-existing gradients;
- all SVD quantities/transforms/losses/logits/parameters/final gradients finite;
- preserve any CUDA SVD fallback warning verbatim; do not rerun merely to hide it;
- add only minimal tests for seed0 provenance regeneration and seed-specific disjoint construction; do not weaken existing tests.

## Deliverable

Append `CODEX REPORT H04-B` with STATUS, source SHA, exact commands/run IDs, tests, seed0 regeneration receipt, per-seed provenance hashes/disjointness, one compact table for seeds1/2 x three arms, fit/final-gradient/head-norm diagnostics, paired alignment/rank diagnostics, `B/G/Rgain` values, state/RNG receipts, warnings, artifact paths, and a 3-5 sentence interpretation following the frozen branch.

Do **not** implement PCA, random projection, low-rank learned maps, nonlinear transport, OT, CORAL/whitening, hybrid/gating heads, anchor selection, new losses, or publication-scale sweeps in H04-B. Await research-lead review.

## CODEX REPORT H04-B — DONE (2026-09-18 00:26 +08)

STATUS: DONE; **cross-seed mechanism/compression replicated** under all frozen thresholds. Source `c27f05b8996018c65ae7fa032487a40b9154edbf`;release `20260918-002045-h04b`;run `20260918-002118-h04b-crossseed`,exit0. Only seeds1/2,round10,three arms each. No new method,other anchor counts,round2 or solver changes.

Changes: `pprtp/cross_seed.py` composes existing calibration_indices/assign_indices/select_anchors policies perseed; `pprtp/owner_probe.py` opt-in final-gradient/per-client support and existing-gradient receipt; `pprtp/run.py` opt-in --cross-seed-probe; `tests/test_cross_seed.py`; `scripts/report_h04b.py`.27 tests pass locally/remotely,previous25 preserved.

Commands (project root,AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h04b
./scripts/autodl-run.ps1 -Name h04b-crossseed -Cmd 'PPRTP_SOURCE_SHA=c27f05b8996018c65ae7fa032487a40b9154edbf bash scripts/run_h01.sh --modes fedgh --seeds 1 2 --rounds 10 --cross-seed-probe'
D:\anaconda3\python.exe scripts/report_h04b.py research_log/H04B/full
```

Before deployment, local tests regenerated seed0 oracle/support/anchors using real official CIFAR10 train label ordering exported read-only from the verified A6000 data. All indices and hashes exactly equal H02-C/H02-E/H03-A; explicit receipt `research_log/H04B/seed0_regeneration.json`,50KB label fixture with source/SHA retained. This fixture is used for provenance tests,not anchor label conditioning. Current-seed oracle indices are exclusion-only,no oracle heads trained. Each seed has disjoint2000train/1000oracle/2000support/1000anchors;support100per ownedclass/client,unique acrossclients. Anchors selected from exclusions label-blind using161803,compressed prefix exact first256;oracle314159/support271828 unchanged. Official test data evaluation-only.

Every arm uses the same perseed support and frozen state,fresh zero-init head,fullbatch LBFGS lr1,max_iter2000,strong_wolfe,tolerance_grad1e-9,tolerance_change1e-12,no regularization. Centered float64SVD/float32 applied Procrustes toclient0,reflections allowed,unchanged. All10 H02-A client/prototype/server/ordinary metrics exact for EACH seed. All arms preserve client/server parameters/buffers/prototypes,module modes,CPU/CUDA RNG and existing gradients. All quantities finite.

| Seed | Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | native_2000 | 78.050000 | 0.000000 | 15.610000 | 15.610000 | 100.000000 | 1.0371183e-08 | 1.51575374e-08 | 1.18869849e-07 | 70183.2266/6569.71191 | 284/316 |
| 1 | paired_1000_2000 | 35.750000 | 25.325000 | 27.410000 | 27.409999 | 100.000000 | 6.11538482e-08 | 5.61580826e-08 | 4.21879548e-07 | 494414.719/18858.8184 | 1398/1484 |
| 1 | paired_256_2000 | 44.200000 | 20.962500 | 25.610000 | 25.610000 | 100.000000 | 8.34464997e-10 | 7.4051032e-10 | 5.48625678e-09 | 675899.938/22872.6035 | 1053/1094 |
| 2 | native_2000 | 75.500000 | 0.000000 | 15.100000 | 15.100000 | 100.000000 | 9.00028585e-09 | 8.25218294e-09 | 7.65028361e-08 | 62626.0234/7737.11035 | 329/373 |
| 2 | paired_1000_2000 | 31.599999 | 22.812500 | 24.570000 | 24.569999 | 100.000000 | 8.10608611e-08 | 1.80212155e-07 | 1.09756093e-06 | 866001.812/24828.0664 | 1584/1666 |
| 2 | paired_256_2000 | 38.800000 | 19.537500 | 23.390000 | 23.390000 | 100.000000 | 5.36441724e-10 | 6.63552491e-10 | 4.33157687e-09 | 461095.906/13343.0605 | 1243/1303 |

| Seed | B | P1000 | P256 | G1000 pp | G256 pp | Rgain |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 25.32499983906746 | 20.962500125169754 | 25.32499983906746 | 20.962500125169754 | 0.8277393981591297 |
| 2 | 0.0 | 22.8125 | 19.537499845027924 | 22.8125 | 19.537499845027924 | 0.8564383493710871 |

Frozen branch: cross-seed mechanism/compression replicated.

N256 payload524288 bytes/client (5242880 total),exact3.90625x reduction from N1000. Diagnostic anchor features only; support side channel excluded,not a deployable protocol.

| Seed | Set | SHA256 |
|---|---|---|
| 1 | oracle | 4c972e65f2c29657b36e7bd0641a3632e390090b17a1ce2899234ceffc8e5f2a |
| 1 | support | 8373dc4071d7760582de0613b98aae19620374d6f1b6b1cc01b5d5c8b9e207dd |
| 1 | anchor | 5ac2570f6c34481b027322b9dd1215b877e509ea696e9156d2c429e0095d23c2 |
| 2 | oracle | d0165050117976afcb1e7ddca3afb70971bb033799ddba42d43560ad4b86dd2a |
| 2 | support | b59d0199adad3696cf32f95d272375d23f4ca0979da76e234da806d2fbefab4a |
| 2 | anchor | 9582386f902eb84ac6ad3ee40d851dad04b8aa3110adb6a5d43846513e802f1c |


Rank/alignment summary (full per-client residuals/orthogonality/transform hashes in RESULTS.md;singular extrema/tolerances in raw final.json):

| Seed | N | Effective rank range | Rank ceiling | Nonreference reduction range | Max orthogonality double/applied |
|---|---|---|---|---|---|
| 1 | 1000 | 406–407 | 512 | 49.9423–74.7762% | 3.82450708e-12/4.274822e-06 |
| 1 | 256 | 255–255 | 255 | 48.3255–75.8552% | 4.7373369e-12/3.98163729e-06 |
| 2 | 1000 | 405–405 | 512 | 46.6197–81.7880% | 3.58734055e-12/4.17462616e-06 |
| 2 | 256 | 255–255 | 255 | 45.7398–82.7018% | 4.30533134e-12/3.95323696e-06 |

Rank tolerance remains512*eps64*smax,no rank truncation. All six heads100% fit;final-gradient/head norms are in the table and perclient support/test correct/counts in rawJSON. Scores are raw/unclipped;no seed0 oracle reference is misapplied to seeds1/2.

Warnings: this run retains the known `Can't initialize NVML` PyTorch warning verbatim in train.log (tests and runtime);CUDA computation/tests completed. No SVD fallback warning appeared in this run;H04-A's historical SVD warning remains preserved. No reruns to hide warnings. Ddrive capacity constraint repaired by deleting only one ignored duplicate checkpoint after exact local/remote SHA match;original retained remotely,receipt in progress.md.

Interpretation: both seeds satisfy G1000>=10pp,G256>=8pp,Rgain>=.75 with all heads100%fit and clean provenance. Thus the frozen correspondence mechanism and N256 compression replicate beyond seed0,with gain retention82.77%/85.64%. The native heads fit owner support perfectly yet retain0% missing accuracy,while paired support gives substantial missing recognition in both splits. Extreme unregularized head norms and the seen/missing tradeoff remain diagnostic limitations;this is not yet a deployable method or publication-scale validation. Stop and await lead review;no low-dimensional map,learned compressor or gating implemented.

Artifacts: `research_log/H04B/full/RESULTS.md`,verification.json,perseed cross_seed_provenance.json,rounds.jsonl/final.json,meta/run/log/tests;seed0 regeneration receipt/fixture under H04B. Remote originals/checkpoints `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-002118-h04b-crossseed`.
