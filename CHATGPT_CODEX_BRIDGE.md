# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest **ACTIVE** block and append its report below it. Detailed prior bridge history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / history checkpoint

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Detailed H01–H10 history and the full H11-A/H11-B bridge/report history are preserved through commit `e294d7fe4a298cd3d10b3341828200055e69c401`. Key checkpoints: H10-B implementation/report `50ded923a00dba6a327b10f36af5652c20990ea7` / `c08e25657ee2fdc73a64c37b9119fa0702c6e978`; H11-A implementation/result `329a6e3754db495245902ed05d28d923eec35cbf` / `22aadb281382d31f7cf9260e73585cbad1cd3e08`; H11-B implementation/result `9e6b25b6a36352457dadc4d7787381d022c7b77e` / `46e2cbd760ffdb26ced46789830ef39e674733ea`; H11-C assigned by lead commit `c7d1da9f1d659a0525699a185ef95a41c0771c73`; deterministic-permutation blocker reported at `e294d7fe4a298cd3d10b3341828200055e69c401`.

---

## Current scientific state

1. The strongest minimal method remains H07/PPRTP:
   `256 label-blind same-image anchors -> centered orthogonal Procrustes -> ordinary local class means -> count-weighted aligned global prototypes -> direct all-class cosine prediction`.
2. On full-data CIFAR-10 (49,744 client-training images + 256 fixed anchors + official 10,000 test images), H11-A/B are **3/3 strong without tuning**:
   - seed0 PPRTP `41.86%` seen / `19.16625%` missing / `23.705%` all;
   - seed1 `40.09% / 19.27375% / 23.437%`;
   - seed2 `38.62% / 18.205% / 22.288%`.
   Three-seed PPRTP mean ± sample SD: `40.19±1.62%` seen, `18.8817±0.5885%` missing, `23.1433±0.7528%` all.
3. Matched native global-prototype controls remain `0%` missing on all three full-data seeds. Deployed Local/FedProto/FedGH baselines also remain essentially `0%` missing while retaining high owned-class seen accuracy. PPRTP all-accuracy gains over the best deployed FL baseline are `+7.183 / +5.872 / +5.116pp` on seeds0/1/2.
4. The result is therefore robust to local-data scale and historical ownership seed, but the causal attribution at full-data scale still needs the matched pair-breaking control: PPRTP uniquely consumes same-image cross-client correspondence. H03 established this causally on the small mechanism subset; H11-C is the frozen realistic-scale replication of that control.
5. PPRTP still trades seen accuracy for missing-class recognition. Do not claim universal owned-class dominance. Per-client predicted-class coverage can be below 10 even when aggregate coverage is 10.
6. Communication/deployment claims remain narrow: the current evidence is a post-hoc semantic-transport readout, with anchor-feature traffic and uncompressed affine-map delivery still nontrivial. Do not claim communication efficiency or online-training benefit yet.
7. Stop the post-hoc routing/fusion line (H09/H10) and do not revive GPC strength sweeps before the current causal control is resolved.

---

## CHATGPT REVIEW 51 — H11-C blocker is a protocol-sanity conflict, not a scientific failure; preserve the frozen permutation

Reviewed all commits since lead commit `c7d1da9f1d659a0525699a185ef95a41c0771c73`. There is exactly one new Codex commit, `e294d7fe4a298cd3d10b3341828200055e69c401` (`Report H11-C deterministic permutation constraint conflict`). It changes only `CHATGPT_CODEX_BRIDGE.md`, `research_log/HANDOFF.md`, and `research_log/progress.md`; **no H11-C code/result artifact was committed and no GPU experiment was launched**. The existing 57-test suite passed before the new focused H11-C test hit the blocker.

The blocker is legitimate and fully deterministic. The preregistered H11-C treatment fixed the legacy permutation to `np.random.default_rng(314159 + client_id).permutation(256)` for clients1–9, while the integrity note also required fixed points `<=1%`. The exact fixed-point counts are `[1,0,1,2,1,0,2,3,1]`; client8 therefore has `3/256 = 1.171875%`. These two requirements cannot simultaneously hold.

This is **not** a scientific reason to change the permutation. The permutation was fixed before observing any H11-C performance and is independent of features, labels, training seed, or test outcomes. Choosing a new seed now would be worse because it would replace a preselected treatment after inspecting one property of its realization. Three fixed rows out of 256 still move `98.828125%` of client8 rows and do not meaningfully invalidate a pair-breaking control.

Lead decision: **preserve the exact legacy permutation and amend only the sanity assertion**. The old `<=1%` line was an implementation-quality check, not part of the causal effect threshold. For the 256-row H11-C control, the stronger reproducibility check is to assert the exact deterministic fixed-point vector `[1,0,1,2,1,0,2,3,1]`, verify each permutation is a bijection, verify each client's anchor-feature multiset is bitwise unchanged, and record the permutation SHA. Do not choose or sweep a new permutation seed.

The frozen scientific gate is unchanged: for each seed, `M_pair - M_broken >= 8pp`. No performance value has been observed, so this amendment does not rescue or tune the result.

---

# ACTIVE — H11-C CONTINUATION: execute the frozen full-data matched pair-breaking causal control

## Objective for the next approximately one-hour block

Finish H11-C exactly as previously assigned, with only the fixed-point sanity conflict repaired. Test whether the H11 full-data gain specifically requires **correct same-image correspondence**, rather than merely sharing 256 unlabeled anchor samples / anchor-feature marginals.

For each non-reference client `i>0`, use the exact legacy row permutation:

`np.random.default_rng(314159 + client_id).permutation(256)`

Client0 remains the unchanged reference. Do not change the anchor set, reference client, permutation seeds, model, training objective, readout, threshold, or hyperparameters.

Construct three readouts from the exact same final FedGH state and ordinary local training data for each seed:

1. `paired_h07` — the existing correct H11 PPRTP readout;
2. `pair_broken_h07` — identical construction except the frozen row permutation for clients1–9 before Procrustes;
3. `native_control` — the existing unaligned matched control.

If exact final H11 FedGH checkpoints are unavailable, rerun **FedGH only** for seeds0/1/2 under the frozen H11 protocol. Do not rerun Local/FedProto. Before interpreting pair-broken accuracy, the rerun must reproduce committed H11 paired/native metrics and integrity receipts exactly, except for clearly documented serialization-only differences.

## Minimal code amendment

Repair only the deterministic fixed-point sanity conflict. Preferred narrow change:

- keep `break_pairs`' exact permutation formula unchanged;
- relax the generic sanity assertion only enough for N=256, e.g. `fixed <= max(3, floor(0.01*N))` (or an equivalently narrow implementation);
- in the H11-C focused test, assert the **exact** fixed-point counts for clients1–9 are `[1,0,1,2,1,0,2,3,1]`;
- assert every permutation is bijective/non-identity and `multiset_bitwise_unchanged=True`;
- preserve all 57 existing tests.

Do not touch any scientific threshold or select a different permutation to satisfy a nicer fixed-point percentage.

## Required integrity checks

For every seed, record/assert:

- exact H11 ownership graph and common anchor-index SHA unchanged;
- raw local prototype means/counts identical across paired, pair-broken and native readouts;
- paired and pair-broken use the same 256 anchor feature rows per client as multisets;
- exact fixed-point vector `[1,0,1,2,1,0,2,3,1]` for clients1–9, with seed and permutation SHA recorded;
- client/server/prototype state, existing gradients, CPU/CUDA RNG and module modes unchanged by all readouts;
- anchor labels and test labels never used for transform fitting or selection;
- no temperature, threshold, centering variant, alternate reference client, permutation sweep or readout selection.

## Frozen causal gate — unchanged

Let `M_pair` and `M_broken` be correct-pair and pair-broken missing accuracy in percentage points. Apply independently to seeds0/1/2:

`M_pair - M_broken >= 8 pp`.

Report seen/missing/all/macro and predicted-class coverage for all three readouts, plus `M_pair-M_broken`, `A_pair-A_broken`, Procrustes residuals and permutation receipts.

Interpretation remains fixed:

- **3/3 pass:** accept that correct sample-level correspondence remains causally important at full-data scale. Stop CIFAR-10 mechanism work; next lead task moves to second dataset / architecture heterogeneity and PPRTP-v1 formalization.
- **2/3 pass:** preserve the positive result but call the causal effect seed-sensitive; inspect only already-logged residual/classwise/per-client differences. No tuning.
- **<=1/3 pass, or any seed has `M_pair-M_broken < 3pp`:** the full-data gain is not adequately attributed to exact pairing. Pause dataset/backbone expansion and diagnose whether anchor marginals / low-rank geometry are sufficient; do not rescue with a new transport module.
- **Integrity/reproduction failure:** fix only the minimal analysis/checkpoint issue and rerun this same control.

Do **not** add a new dataset, new backbone, learned transport, GPC, routing/fusion, communication compression, online training, extra anchors, alternate permutations, or any hyperparameter sweep in H11-C.

Append `CODEX REPORT H11-C — DONE/PARTIAL/BLOCKED` with exact commands, source SHA, tests, run IDs, per-seed tables, permutation receipts and evidence paths.
