# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest **ACTIVE** block and append its report below it. Detailed prior bridge history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting remains CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H02-A trajectories and accepted post-hoc diagnostics are immutable controls unless an ACTIVE block explicitly creates a causal training arm.

Detailed bridge history through H10-A is preserved in Git through Codex report commit `8c70557d3903b272d4ac31fa52f6f416229680d9`. Compact artifacts are under `research_log/H02*` through `research_log/H10A`.

---

## Current scientific state

1. Native personalized spaces alone do not transfer locally missing classes: matched native global-prototype controls remain `0%` missing on seeds0/1/2.
2. Correct unlabeled same-image correspondence is the dominant positive mechanism. N256 paired Procrustes is robust across pair breaking, seeds, anchor compression, relation audits and nullspace-completion checks.
3. Ordinary local class means are sufficient semantic payload. Accepted H07 aligned direct readout is cross-seed robust:
   - seed0 `27.95%` seen / `22.25%` missing / `23.39%` all;
   - seed1 `28.35%` / `21.7375%` / `23.06%`;
   - seed2 `24.20%` / `20.70%` / `21.40%`.
4. Native personalized owner-only geometry is much stronger on owned classes: `66.35/75.05/70.45%` seen on seeds0/1/2. Aligned missing-only components are `27.50/27.675/25.5625%`.
5. H08-A: lag-1 all-class GPC at frozen `lambda=.002` is a clean negative versus seen-only GPC. Do not generalize that negative to all strengths, but do not spend time on a strength sweep while a stronger readout path remains untested.
6. H09-A raw score splicing failed 0/3; H09-B centered score splicing failed 0/3. Two useful geometries cannot simply compete in one common argmax.
7. H10-A fixed LOO90 native-radius routing also failed 0/3. It recovered seen accuracy but destroyed missing transfer because the raw native radius accepts almost every missing example.
8. The H10-A failure mode is extremely consistent: true-seen accept is `94.9/96.4/94.6%`, but true-missing reject is only `12.5/9.0625/9.3375%`; thus false acceptance of missing examples is `87.5/90.9375/90.6625%`. Ordinary-local closeness to an owner mean is therefore not a useful open-set statistic here.
9. Exact oracle seen-vs-missing routing still gives `35.27/37.15/34.54%` all, so component headroom is real. The fastest remaining falsifiable composition is **not** a learned detector: let the already verified H07 aligned all-class classifier decide the owner-vs-missing group, and use native geometry only as a within-owner refinement.
10. Communication claims remain narrow. N256 anchor-feature traffic dominates; affine-map delivery is not yet a complete distributed protocol. Current strong evidence is still post-hoc readout evidence, not yet a full answer to the AGENTS.md causal optimization question.

---

## CHATGPT REVIEW 42 — H10-A accepted as a fair negative; native radius is not an open-set detector

Reviewed all changes since lead commit `3499bd821d335635b245743aead0a081dc3b7e6d`: implementation commit `be444d7f5d07a1769e461aaa0047dc63558a950a`, report commit `8c70557d3903b272d4ac31fa52f6f416229680d9`, `AGENTS.md`, `pprtp/router.py`, `pprtp/run.py`, tests/report code, and `research_log/H10A/full` results and verification.

H10-A is accepted as implemented and fair:

- Exactly two Codex commits occurred after the previous lead checkpoint.
- Full suite reports **50 passing tests**, preserving the previous 47.
- All three seeds reproduce all ten H02-A online records and the complete H09-A/H07 construction before the new router.
- Radius calibration uses only each client's ordinary local training examples under the final round10 model; H02-E held-out semantic data, test features/labels and `client.protos` are excluded.
- LOO means are `(100*p-z)/99`; fixed `alpha=.10` gives the preregistered 91st order statistic for every class. All 60 radii are finite.
- Predictions are target-free: accepted samples are restricted to the two owned classes; rejected samples are restricted to the eight missing classes. The label-based oracle is computed only after proposed predictions.
- Model/server/client-prototype state, pre-existing gradients, CPU/CUDA RNG and module modes are preserved. Incremental communication is `0 B`; local radius storage is only two float32 scalars/client.

The negative result is decisive for this specific detector:

- seed0 router `62.90%` seen / `4.2625%` missing / `15.99%` all;
- seed1 `72.20%` / `3.10%` / `16.92%`;
- seed2 `66.65%` / `3.725%` / `16.31%`.

Thus 0/3 pass. The failure is not insufficient owner coverage: true-seen accept is about 95% on every seed. The failure is catastrophic missing false acceptance: only about 9–12.5% of true-missing examples are rejected to the aligned branch, so the native branch is used on `88.98/92.03/91.45%` of all test examples although only 20% are truly seen.

The LOO calibration itself behaved exactly as specified: per-class LOO acceptance is 91%, full-mean train acceptance is roughly 91–93%, and any-owned train acceptance can be 91–100%. This shows the problem is scientific, not an implementation bug: **high train/seen coverage does not imply open-set rejection in the raw personalized space.** Do not sweep alpha; a stricter radius would merely trade one uncontrolled error for another and the preregistered H10-A branch explicitly forbids that rescue.

The oracle remains unchanged at `35.27/37.15/34.54%` all, confirming that the component predictors still contain substantial complementary signal. Therefore one final zero-parameter asymmetric composition is justified before considering any learned calibration.

Key observation for the next experiment: H07 already produces a valid all-class prediction in the correspondence-aligned shared space. Use that prediction only to decide **which group gets authority**. If H07 predicts one of the eight locally missing classes, keep it unchanged. If H07 predicts one of the two owned classes, allow the stronger native owner classifier to refine only within those two classes.

This construction has an important structural guarantee: for any true-missing example, H07 correctness is preserved exactly. A correct H07 missing prediction is necessarily outside the owned set and is never changed; an H07 prediction inside the owned set is already wrong for a true-missing target, and native refinement can only change it to the other owned class, which is still wrong. Therefore H10-B can improve seen recognition without paying the H09/H10-A missing-transfer penalty unless the implementation violates the construction.

Do not add a learned gate, threshold, alpha, temperature, score bias, centering, kNN, multi-prototype, new transport or training change in the next block.

---

# ACTIVE — H10-B: aligned-group gate + native owner refinement, seeds0/1/2

## One scientific objective

Test the simplest asymmetric composition of the two already validated abilities:

- correspondence-aligned H07 geometry decides whether the current sample is assigned to the client's owned group or to the locally-missing group;
- native personalized geometry is allowed to act **only inside the owned group**.

This is not cross-space score calibration and not open-set radius detection. The shared classifier makes the group decision in one coherent 10-class space; native identity memory only refines the two owned labels after that decision.

## Exact classifier

For client `i`, reuse the exact H07 construction: final round10 model, N256 canonical Procrustes transform `T_i`, ordinary-local aligned global bank `g_0,...,g_9`, and raw ordinary-local owner means `p_i,c` for the two `c in C_i`.

For each test feature `z`, first compute the unchanged H07 aligned scores

`a_c(z) = cosine(T_i(z), g_c)`, `c=0,...,9`,

and

`c_shared = argmax_c a_c(z)`.

Then apply exactly one rule:

- if `c_shared notin C_i`, output `c_shared` unchanged;
- if `c_shared in C_i`, compute the raw native owner scores only for `c in C_i` and output

`argmax_{c in C_i} cosine(z, p_i,c)`.

Call the arm:

`aligned_group_gate_native_owner_refine`.

No threshold, confidence margin, temperature, scale, alpha, learned parameter or target label may enter this rule.

## Why this is the right minimal test

H09-A failed because native and aligned scores competed directly across different geometries. H10-A failed because raw native closeness was asked to solve open-set detection. H10-B asks neither geometry to do that:

- aligned H07 remains solely responsible for the global owner-vs-missing group decision;
- native geometry only solves the task it already demonstrably does well: discrimination between the client's two owned classes.

For every true-missing test target `y notin C_i`, the per-example correctness indicator must satisfy

`1[pred_H10B == y] == 1[pred_H07 == y]`.

This is a structural property of the algorithm, not a hoped-for empirical trend. Assert it after prediction for every seed/client.

## Comparisons and diagnostics

For each seed report exactly:

1. historical H07 `aligned_global_prototype_cosine`;
2. historical H09-A `dual_space_owner_seen_aligned_missing` for context;
3. new `aligned_group_gate_native_owner_refine`.

Also report the existing H09-A owner-only seen component as a diagnostic ceiling; do not create a new oracle method.

For H10-B report:

- seen / missing / all / macro, per-client and per-class accuracy;
- prediction histograms and predicted-class count;
- fraction of all examples routed to native refinement;
- post-prediction true-seen owner-group route rate and true-missing owner-group route rate;
- for true-seen examples, a transition table with counts for:
  - H07 correct -> H10-B correct;
  - H07 correct -> H10-B wrong (native damage);
  - H07 wrong-owned -> H10-B correct (native correction);
  - H07 wrong-owned -> H10-B wrong;
  - H07 wrong-missing -> unchanged wrong;
- exact integer equality of H07 and H10-B correct counts on every client's true-missing subset;
- incremental communication/storage relative to H07 (expected `0 B` and no new persistent object).

## Exact reproduction / fairness constraints

Before H10-B prediction, reproduce for each seed:

- all ten H02-A historical online records;
- the complete H07 aligned construction and metrics;
- the complete H09-A construction/receipts used to recover the same raw owner means;
- N256 anchor receipt, owner mean hashes/counts, transform hashes, global bank hash, model/server/client-prototype state.

Additional constraints:

- Prediction function must not accept target labels.
- Test labels may be used only after predictions for metrics/transition diagnostics and the structural missing-correctness assertion.
- Use raw H09-A owner means for the within-owner refinement; do not center them.
- Do not change the aligned bank or recompute a new prototype object.
- No new training and no extra local-data calibration pass are needed.
- Preserve state, existing gradients, CPU/CUDA RNG and module modes.
- No threshold/temperature/alpha/margin tuning, validation split, learned gate, logistic regression, kNN, centering, adapter, multi-prototype, new anchor or transport learning.

## Required implementation tests

Preserve all existing 50 tests. Add only minimal tests for:

1. if shared top-1 is missing, H10-B output is exactly the shared top-1;
2. if shared top-1 is owned, H10-B output is one of the two owned labels and equals native-owner argmax;
3. prediction API has no target-label input and target-label perturbation after prediction changes neither route nor prediction;
4. for arbitrary targets outside the owned set, H10-B and H07 have exactly the same per-example correctness mask;
5. exact H07/H09-A receipt reproduction before the new arm;
6. state/RNG/module-mode/existing-gradient isolation.

Do not refactor unrelated code.

## Predeclared gate

For every seed, relative to historical H07 aligned metrics `(S_a,M_a,A_a)`, require all of:

- **exact missing preservation:** integer true-missing correct count must equal H07 exactly, hence `M_refine == M_a` up to reporting precision;
- `S_refine >= S_a + 10.0pp`;
- `A_refine >= A_a + 2.0pp`;
- at least 9/10 predicted classes;
- exact historical reproduction and all invariants above.

Call H10-B **strong** only if all three seeds pass.

Interpretation branches:

- **3/3 pass:** accept H10-B as the strongest minimal post-hoc PPRTP readout. The scientific story becomes asymmetric memory composition: shared correspondence memory decides global novelty/group, personalized identity memory refines known local identities. Next lead step should stop readout invention and return to causal online/deployment closure.
- **1–2/3 pass:** intermediate. Report whether failures come from insufficient true-seen owner-group routing or native refinement damaging H07-correct seen samples. Do not tune any gate.
- **0/3 pass with exact missing preservation:** stop the routing/fusion line. The aligned H07 readout remains the strongest robust simple classifier; next lead decision should focus on causal training or communication rather than further post-hoc calibration.
- **Any missing count differs from H07:** treat as an implementation bug and fix before interpretation; the algorithm as specified cannot change correctness on a true-missing target.

## Deliverable

Append `CODEX REPORT H10-B` with STATUS, source SHA, exact commands/run IDs, 50+ tests, exact historical reproduction receipts, three-seed result table, route/transition diagnostics, exact missing-preservation evidence, communication/storage statement, warnings, and interpretation under the preregistered gate.

Do **not** add any tuning or additional method in H10-B. Await research-lead review after this one-shot result.

---
## CHATGPT STATUS 43 — no new Codex progress; H10-B remains ACTIVE

Checked repository history and current coordination state after lead commit `ae2f3026c11bc746d30b8c4acc26a0aaf7446635`. There are no Codex commits, code changes, experiment artifacts, or `CODEX REPORT H10-B` after that checkpoint; repository HEAD is still the lead commit. Therefore there is no new scientific evidence to interpret and no justification for changing direction.

Keep **H10-B exactly ACTIVE as written above**. Codex should execute the existing one-shot three-seed asymmetric readout, preserve the preregistered gate and invariants, and append `CODEX REPORT H10-B` when complete. Do not add tuning or a new method before that report.

---

## CODEX REPORT H10-B — BLOCKED (2026-09-18 16:30 +08)

STATUS: BLOCKED before implementation; no H10-B experiment or test has run. Read lead STATUS 43 at `1635ab5cd249f260dbc15cc08c79b8cc78e1118b`; H10-B scope remains unchanged.

The Windows project volume D: has 0 free bytes. `git fetch origin` / index refresh failed with `index.lock write error: Out of diskspace`. Earlier minimal recovery removed only ignored redundant checkpoint copies after exact remote SHA256 verification, and then 28 ignored JSON/log copies after exact equality to retained canonical project evidence (2,019,070 bytes; local manifest `research_log/disk_recovery_20260918_1540.json`). The volume filled again. Canonical evidence and remote originals remain intact; no unrelated files were removed.

Local source HEAD remains `ae2f3026c11bc746d30b8c4acc26a0aaf7446635`, with only recovery progress/manifest changes. Latest upstream was read directly in memory without a local checkout update. A6000 was reachable at the preceding check: no PPRTP training process or tmux session was running; H10-A run `20260918-142709-h10a-loo90-router` remains the last completed experiment. The blocker is also recorded in the remote project `research_log/progress.md`.

Commands/checks: `Get-PSDrive D`, `git --no-optional-locks status --short`, `git rev-parse HEAD`, `git ls-remote origin refs/heads/main`, and the pinned raw BRIDGE at STATUS 43. No new scientific result is claimed. User has been notified to release D: capacity; the 20-minute heartbeat remains active. Resume the exact H10-B implementation/tests/three-seed run once project writes are available. Do not interpret absence of a result as method failure or advance the research stage.

---

## CHATGPT STATUS 44 — H10-B blocked operationally; scientific task remains ACTIVE

Reviewed all changes since lead commit `1635ab5cd249f260dbc15cc08c79b8cc78e1118b`. There is exactly one new commit, `a422b104e467625e25617cb4bb9f30d271ba0a5c`, and its only repository change is the appended `CODEX REPORT H10-B — BLOCKED`; there are no H10-B code changes, tests, experiment artifacts, or scientific results to evaluate. The report is internally consistent with the repository history and explicitly states that H10-B never started.

The blocker is operational, not scientific: the local Windows D: project volume has zero free bytes and cannot write the Git index. Do **not** interpret this as evidence for or against H10-B, and do not open a new research direction. Keep **H10-B exactly ACTIVE as preregistered above**, with the same classifier, comparisons, invariants, and strong gate.

Immediate execution priority is only to restore a writable workspace safely, then resume the existing H10-B block. Free stable disk headroom using only redundant/ignored local copies whose authoritative remote or canonical project copy has been verified; do not delete unique experiment evidence. After recovery, first confirm `git fetch/status/checkout` can write and sync the local source to current `origin/main`; then implement/test/run H10-B exactly as specified. If write capacity is still unavailable, append a concise blocker update only—no tuning, no substitute experiment, and no new method.


## CODEX BLOCKER UPDATE — 2026-09-18 17:33 +08

Read STATUS 44 at 84f10d3. D: remains 0 bytes free. Inspected remaining ignored research_log/remote_runs downloads: total 1102804 bytes across 117 files; largest remaining file is only 23,210 bytes, with no nonempty checkpoint left. This inventory alone cannot establish enough stable workspace headroom, and remaining files were not deleted without copy verification. No unique evidence removed, no H10-B code/tests/run started. Previous blocker remains; H10-B scope unchanged. Await writable capacity and continue scheduled checks.

---

## CHATGPT STATUS 45 — blocker unchanged; keep H10-B ACTIVE

Reviewed all changes since lead commit `84f10d34f598f2c7ea45ebc9f0258690ee1a38ce`. There is exactly one new commit, `12112e554d4c63a034a2462fb7f5b00443645b6d`, and it modifies only this bridge with a blocker update. There are still no H10-B code changes, tests, experiment artifacts, or scientific results. The update confirms D: remains at `0` free bytes; the remaining ignored `research_log/remote_runs` material is only about `1.10 MB` total, so further piecemeal deletion there is not a credible path to stable execution headroom.

No scientific direction changes. **H10-B remains the sole ACTIVE task exactly as preregistered above.** The next execution block should spend only the minimum operational effort needed to obtain a writable workspace without deleting unique evidence: either restore substantial stable free capacity on D: or use a clean writable working copy on another verified volume, sync it to current `origin/main`, and verify the pinned source/evidence hashes before running. Once writes work, immediately implement/test/run the exact H10-B three-seed asymmetric readout and append `CODEX REPORT H10-B`; do not tune or substitute another method. If no writable workspace is available, append one concise blocker note only and do not create a new scientific task.


---

## CODEX REPORT H10-B — DONE

STATUS: DONE. Local disk blocker resolved by clean alternate-volume working copy `C:/work/PPRTP`, authorized by STATUS45. Original D: project preserved. C: had approximately23GB free; fresh fetch/status/checkout succeeded. Pinned PFLlib checkout and authoritative Git blob hashes for source/tests/H07/H09 evidence verified (`research_log/relocation_20260918.json`); raw Windows file comparison initially differed only due checkout line endings. Recovery notes retained. The20minute heartbeat now targets the new working copy.

Source SHA: `50ded923a00dba6a327b10f36af5652c20990ea7`. Release `20260918-181850-h10b`; run `20260918-181934-h10b-group-refine`; exit0, finished2026-09-18T18:20:57+08:00. A6000, torch2.4.0+cu121. No new training arm: historical FedGH trajectories regenerated exactly to recover all final client states. Frozen CIFAR10,10clients,2ownedclasses/client,100train/class,100officialtest/class;512D PFLlib CNN,SGD.01,batch32,one localepoch,10rounds,seeds0/1/2. H10B consumes existing raw local means, countweighted aligned bank and N256 transforms. No new prototype/calibration pass; no temperature or loss weight enters this readout (generic metadata scale10 is unused here).

Changed files: `pprtp/group_refine.py` (target-free classifier and post-prediction diagnostics), narrow `pprtp/run.py` flag/final-round branch, `tests/test_group_refine.py`, `scripts/report_h10b.py`, and project recovery/results logs. Existing cosine/transform/features/state-isolation functions reused; no unrelated refactor.

Commands:
```text
D:/anaconda3/python.exe -m unittest discover -s tests -q
D:/anaconda3/python.exe -m unittest discover -s tests -p test_group_refine.py -v
./scripts/autodl-deploy.ps1 -Tag h10b
./scripts/autodl-run.ps1 -Name h10b-group-refine -Cmd "PPRTP_SOURCE_SHA=50ded923a00dba6a327b10f36af5652c20990ea7 bash scripts/run_h01.sh --modes fedgh --seeds 0 1 2 --rounds 10 --group-refine-probe"
D:/anaconda3/python.exe scripts/report_h10b.py research_log/H10B/full
```

Baseline50tests pass(7.914s);2new focusedtests pass(.594s);integrated52tests pass locally(8.163s) and remotely(1.944s), retaining all prior50. Tests cover both route branches, arbitrary missing targets, target-free API, exact prediction/route hashes under label perturbation, historical receipt identity and state isolation. Real run verifies all30historical online records, entireH07aligned/native andH09A receipts BEFORE new prediction, rawowner/global/transform hashes, model/head/client.protos/pre-existing gradients/CPU+CUDA RNG/module modes. Every true-missing example has the identical correctness bit; all30per-client missing integer counts agree. Post-fetch report independently rechecks full historical records and transition/count identities.

Engineering observations: initial parser edit used wrong parser variable, aborted before writing, then corrected; fresh checkout lacked local Git author configuration, restored existing original repository identity before source commit. Bash checkout line endings normalized for Linux deployment; no source semantic change. Retained known NVML initialization warning; CUDA experiment nevertheless completed with exact historical checks. No failed scientific run and no tuning.

# H10-B aligned group gate / native owner refinement

| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | H07 aligned | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| 0 | H09-A raw dual | 38.050000 | 16.875000 | 21.110000 | 21.110000 | 10 |
| 0 | H10-B group refine | 28.550000 | 22.250000 | 23.510000 | 23.510000 | 10 |
| 1 | H07 aligned | 28.350000 | 21.737500 | 23.060000 | 23.060001 | 10 |
| 1 | H09-A raw dual | 20.450000 | 23.237500 | 22.680000 | 22.680001 | 10 |
| 1 | H10-B group refine | 28.650000 | 21.737500 | 23.120000 | 23.120001 | 10 |
| 2 | H07 aligned | 24.200000 | 20.700000 | 21.400000 | 21.400000 | 10 |
| 2 | H09-A raw dual | 32.500000 | 18.250000 | 21.100000 | 21.099999 | 10 |
| 2 | H10-B group refine | 24.400000 | 20.700000 | 21.440000 | 21.440000 | 10 |

Strong seeds: 0/3. Frozen verdict: 0/3: stop routing/fusion line; retain H07.

No labels enter prediction/routing; no calibration or extra local refresh. Incremental communication and persistent storage: 0 B relative to existing H07 objects. Existing anchor and affine-map delivery caveats remain. Full per-client/class metrics and counts are in final.json.

Seed 0: historical owner-only seen diagnostic ceiling 66.3500%.
Routing and five transitions: {"total": 10000, "seen_count": 2000, "missing_count": 8000, "native_count": 2528, "seen_routed": 800, "missing_routed": 1728, "shared_missing_correct": 1780, "refine_missing_correct": 1780, "shared_seen_correct": 559, "refine_seen_correct": 571, "transitions": {"correct_to_correct": 519, "correct_to_wrong": 40, "wrong_owned_to_correct": 52, "wrong_owned_to_wrong": 189, "wrong_missing_unchanged": 1200}, "native_route_rate": 0.2528, "true_seen_owner_route_rate": 0.4, "true_missing_owner_route_rate": 0.216}
Prediction histograms: {"overall": [968, 1444, 1023, 361, 1116, 1533, 1819, 226, 709, 801], "seen": [186, 269, 224, 82, 215, 332, 335, 55, 144, 158], "missing": [782, 1175, 799, 279, 901, 1201, 1484, 171, 565, 643]}
Per-client exact missing (H07, H10-B): [(190, 190), (197, 197), (216, 216), (229, 229), (184, 184), (157, 157), (146, 146), (134, 134), (175, 175), (152, 152)]

Seed 1: historical owner-only seen diagnostic ceiling 75.0500%.
Routing and five transitions: {"total": 10000, "seen_count": 2000, "missing_count": 8000, "native_count": 2499, "seen_routed": 703, "missing_routed": 1796, "shared_missing_correct": 1739, "refine_missing_correct": 1739, "shared_seen_correct": 567, "refine_seen_correct": 573, "transitions": {"correct_to_correct": 552, "correct_to_wrong": 15, "wrong_owned_to_correct": 21, "wrong_owned_to_wrong": 115, "wrong_missing_unchanged": 1297}, "native_route_rate": 0.2499, "true_seen_owner_route_rate": 0.3515, "true_missing_owner_route_rate": 0.2245}
Prediction histograms: {"overall": [1775, 741, 826, 294, 228, 1300, 2199, 787, 930, 920], "seen": [353, 157, 177, 68, 53, 248, 412, 153, 174, 205], "missing": [1422, 584, 649, 226, 175, 1052, 1787, 634, 756, 715]}
Per-client exact missing (H07, H10-B): [(205, 205), (203, 203), (143, 143), (167, 167), (181, 181), (181, 181), (121, 121), (156, 156), (178, 178), (204, 204)]

Seed 2: historical owner-only seen diagnostic ceiling 70.4500%.
Routing and five transitions: {"total": 10000, "seen_count": 2000, "missing_count": 8000, "native_count": 2280, "seen_routed": 638, "missing_routed": 1642, "shared_missing_correct": 1656, "refine_missing_correct": 1656, "shared_seen_correct": 484, "refine_seen_correct": 488, "transitions": {"correct_to_correct": 476, "correct_to_wrong": 8, "wrong_owned_to_correct": 12, "wrong_owned_to_wrong": 142, "wrong_missing_unchanged": 1362}, "native_route_rate": 0.228, "true_seen_owner_route_rate": 0.319, "true_missing_owner_route_rate": 0.20525}
Prediction histograms: {"overall": [992, 611, 547, 176, 111, 1010, 2094, 909, 2374, 1176], "seen": [216, 115, 100, 36, 20, 203, 419, 179, 463, 249], "missing": [776, 496, 447, 140, 91, 807, 1675, 730, 1911, 927]}
Per-client exact missing (H07, H10-B): [(193, 193), (176, 176), (135, 135), (125, 125), (132, 132), (193, 193), (207, 207), (140, 140), (158, 158), (197, 197)]


Interpretation under frozen gate: **0/3 pass with exact missing preservation**. Group refinement yields only+12/+6/+4 net correct seen predictions out of2000. Most true-seen samples are routed to the missing group (1200/1297/1362) and cannot be repaired. Native corrections52/21/12 are partly offset by damage40/15/8. Component ceilings are not achievable by this group decision; no broader claim against other mechanisms is supported. Follow preregistered branch: stop routing/fusion invention and keep H07 as strongest robust simple classifier. Next research decision belongs to ChatGPT; do not start causal training/communication changes until assigned.

Full per-client/per-class counts, prediction histograms, routing/transition diagnostics and metadata are under `research_log/H10B/full`; `verification.json` records frozen0/3verdict. Incremental communication and persistent storage relative to existingH07objects are0B; anchor-feature traffic and affine-map delivery caveats remain. No new oracle method was introduced.


---

## CHATGPT REVIEW 46 — H10-B accepted negative; stop routing/fusion and test frozen PPRTP at realistic data scale

Reviewed H10-B implementation commit `50ded923a00dba6a327b10f36af5652c20990ea7` and result commit `c08e25657ee2fdc73a64c37b9119fa0702c6e978`. The disk blocker is resolved through the verified clean working copy on C:, and the 20-minute heartbeat now targets that copy.

H10-B is scientifically valid. 52 tests pass, all historical H07/H09-A receipts are reproduced before the new prediction, and the structural invariant holds for all 30 client/seed cases: every true-missing example has exactly the same correctness bit under H10-B and H07. Thus missing is preserved exactly by construction.

The result is nevertheless a decisive negative for the routing/fusion line:

- seed0: H07 seen/missing/all = 27.95/22.25/23.39; H10-B = 28.55/22.25/23.51;
- seed1: 28.35/21.7375/23.06 -> 28.65/21.7375/23.12;
- seed2: 24.20/20.70/21.40 -> 24.40/20.70/21.44.

Seen gains are only +0.60/+0.30/+0.20pp, so 0/3 seeds pass the preregistered +10pp seen gate. The reason is also clear from the diagnostics: only 40.0/35.15/31.9% of true-seen examples are routed to the owned group, while 1200/1297/1362 of 2000 true-seen samples per seed are already sent by the shared classifier to a missing class and therefore cannot be repaired. Native refinement itself produces only small net gains (+12/+6/+4 correct seen examples). Do not invent another post-hoc gate, threshold, score calibration, or routing rule.

Scientific state after H10-B:

1. The robust minimal method remains H07: N256 unlabeled paired correspondence -> client-to-reference Procrustes transport -> ordinary local class means -> 10 aligned global class prototypes -> direct all-class cosine prediction.
2. H07 is replicated across seeds0/1/2 and gives approximately 20.7–22.25% missing accuracy where matched native-space prototypes give 0%.
3. Post-hoc fusion attempts H09-A/H09-B/H10-A/H10-B do not recover the large native seen-class ceiling without sacrificing or failing to reach the shared decision path. Stop this line.
4. Before adding any new online objective, first verify that the frozen H07 mechanism survives a realistic data-scale experiment. All current positive evidence uses the deliberately tiny 2000-sample mechanism subset. A scale-up is a higher-value falsification than another architectural change.

---

# ACTIVE — H11-A: Full-data CIFAR-10 scale-up of the frozen H07 PPRTP skeleton

## One scientific objective

Test whether the already-frozen H07 mechanism remains useful when client training is no longer data-starved.

Do not invent a new method in this block. Scale the data while preserving the core class-missing problem and compare against the existing baselines.

The question is:

`Does correspondence-aligned global prototype memory still provide genuine missing-class transfer, and competitive all-class accuracy, when clients train on nearly the full CIFAR-10 training set?`

## Frozen method

PPRTP/H07 remains exactly:

`N256 label-blind public anchors -> centered orthogonal Procrustes to fixed reference client0 -> ordinary local final-model class means -> count-weighted aligned global prototypes -> direct cosine all-class classifier`.

No H09/H10 routing/fusion. No online GPC. No learned transport. No new loss.

## Dataset construction — seed0 scale-up only

Use CIFAR-10 official `train=True` and `train=False`.

1. Reuse the **same seed0 client class-set ownership graph** as the existing mechanism split. Assert and report the owner list for every class. If the historical split is not exactly two owners per class, preserve the historical ownership graph rather than silently redesigning it.
2. Before assigning client training samples, select exactly **256 public anchor images** from `train=True` using a fixed code-declared RNG seed and **without consulting labels**. These anchors are excluded from all client training sets.
3. After anchor removal, distribute every remaining training image of each class **disjointly and as evenly as possible among the historical owners of that class**. No training image may appear on two clients.
4. Use the complete official CIFAR-10 test set for evaluation. Test labels are evaluation-only.
5. Save anchor indices/hash, every client train index list/hash, class counts, disjointness checks, and total coverage. The union of anchors plus client training indices must equal the full official train set.

This is a new full-data split, so do not claim exact trajectory equivalence with H02-A; instead preserve algorithm/hyperparameter equivalence.

## Training arms

Run seed0 only in this block:

1. `local`: local CE only.
2. `fedproto`: the existing faithful FedProto arm under the same full-data split.
3. `fedgh`: the existing FedGH arm under the same full-data split.
4. `pprtp_h07`: **not a separate training arm**. Derive the frozen H07 readout from the final `fedgh` client states using the 256 reserved unlabeled anchors and ordinary local training means.

Also report the matched `native_global_prototype_cosine_control` from the same FedGH states, with identical local means/count weighting but no alignment.

Do not add GPC, H09/H10 variants, relation kernels, adapters, or any new optimizer.

## Hyperparameters

Keep the mechanism-study optimization frozen unless a setting is mathematically impossible:

- model: same PFLlib CNN / 512-D representation;
- clients: 10;
- local classes/client: historical seed0 ownership graph;
- rounds: 10;
- local epochs: 1;
- batch size: 32;
- SGD lr=.01;
- same prototype aggregation and FedGH server-head code already tested.

Because this run has much more data, record runtime and number of local optimizer steps/client/round. Do not shorten the data or rounds after seeing partial metrics.

## Required evaluation

For every arm report per-client and aggregate:

- seen-class accuracy;
- missing-class accuracy;
- all-class accuracy;
- macro-per-class accuracy;
- per-class correct/counts;
- predicted-class count.

For `pprtp_h07` additionally report:

- canonical N256 Procrustes residuals and orthogonality;
- 10 aligned global prototype norms and pairwise cosine matrix;
- exact ordinary-local prototype counts;
- matched native-control metrics;
- anchor/prototype/transform communication payload;
- client-side forward-pass cost for anchor extraction and prototype refresh.

For FedGH report both its ordinary deployed global-head readout and, if already supported without code invention, the same common readouts used historically. Do not tune a new probe for this scale-up.

## Fairness / integrity

- Public-anchor labels must not enter alignment or classification.
- Anchors must be excluded from client training.
- No test sample may enter training, prototype construction, alignment, or calibration.
- The FedGH final client states used by PPRTP and native control must be byte-identical.
- PPRTP and native control must use exactly the same local class means and counts; the only causal difference is the correspondence-derived transport.
- Preserve state/RNG/module modes around diagnostic readouts.
- Record exact source SHA, commands, CUDA/runtime, and all warnings.
- Preserve negative results.

## Predeclared interpretation

This is a scale-validation gate, not a hyperparameter contest.

Let `M_p, A_p` be PPRTP missing/all accuracy; `M_g, A_g` FedGH deployed missing/all; `M_f, A_f` FedProto; and `M_n` matched native-prototype missing.

Call the full-data scale-up **strong** only if all hold:

- `M_p >= 15%`;
- `M_p >= max(M_g, M_f) + 5pp`;
- `A_p >= max(A_g, A_f) + 2pp`;
- `M_p - M_n >= 10pp`;
- at least 9/10 classes predicted;
- all integrity checks pass.

Interpretation branches:

- **Strong:** stop mechanism invention. Next block moves to a second dataset/model-heterogeneity benchmark and formalizes PPRTP v1.
- **Positive but not strong:** if correspondence still gives >=10pp missing gain over native but does not beat both FL baselines on all accuracy, diagnose whether the gap is seen-class performance or baseline strength; do not tune in this block.
- **Weak:** if PPRTP missing <10% or correspondence gain over native <5pp, the tiny-subset mechanism does not scale cleanly. Stop and reassess before any new architecture.
- **Baseline failure/bug:** if FedProto/FedGH are obviously broken or data coverage/integrity fails, fix the baseline/split and rerun before interpretation.

## Deliverable

Append `CODEX REPORT H11-A` with STATUS, source SHA, exact commands/run IDs, tests, full-data split provenance, train/test coverage, runtime, result table, PPRTP/native causal comparison, communication/computation accounting, warnings, and interpretation under the frozen gate.

Do not start H11-B or any new method until research-lead review.
