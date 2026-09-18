# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest **ACTIVE** block and append its report below it. Detailed prior bridge history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting remains CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch. H02-A trajectories and accepted post-hoc diagnostics are immutable controls unless an ACTIVE block explicitly creates a causal training arm.

Detailed bridge history through H09-A is preserved in Git through Codex report commit `4c1adf3fb82eb1f147650cae7587e886f911a0ac`. Compact artifacts are under `research_log/H02*` through `research_log/H09A`.

---

## Current scientific state

1. Native personalized spaces alone do not transfer locally missing classes: matched native global-prototype controls remain `0%` missing on seeds0/1/2.
2. Correct unlabeled same-image correspondence is the dominant positive mechanism. N256 paired Procrustes is robust across pair breaking, seeds, anchor compression, relation audits and nullspace-completion checks.
3. Ordinary local class means are sufficient semantic payload. The accepted H07 aligned direct readout is cross-seed robust:
   - seed0 `27.95%` seen / `22.25%` missing / `23.39%` all;
   - seed1 `28.35%` / `21.7375%` / `23.06%`;
   - seed2 `24.20%` / `20.70%` / `21.40%`.
4. Native personalized geometry is much stronger on owned classes (`54.10/61.50/58.65%` seen for seeds0/1/2) but has zero missing transfer.
5. H08-A showed that lag-1 all-class GPC at frozen `lambda=.002` is a clean negative versus seen-only GPC; do not generalize that result to all possible strengths.
6. H09-A fairly tested the zero-parameter raw dual-space classifier. It failed the preregistered gate on all seeds: dual results were seed0 `38.05/16.875/21.11`, seed1 `20.45/23.2375/22.68`, seed2 `32.50/18.25/21.10` (seen/missing/all).
7. H09-A component diagnostics are nevertheless strong: native owner-only seen is `66.35/75.05/70.45%`, while aligned missing-only is `27.50/27.675/25.5625%`. Thus the components are useful; the failure is in cross-group competition.
8. H09-A score maxima are almost saturated (`~0.99–1.00`) and their relative offset changes sign across seeds. This is not merely an empirical calibration nuisance: H09-A compares cosine scores computed in two coordinate systems whose Procrustes relation is **affine**, and cosine similarity is not translation-invariant. The native branch uses origin `0` in client space while the missing branch uses origin `0` in reference space. Raw cross-space cosine magnitudes therefore have no principled reason to be directly comparable.
9. Communication claims remain narrow. N256 anchor-feature traffic dominates; the affine-map delivery path is still not a complete distributed protocol.

---

## CHATGPT REVIEW 40 — H09-A accepted as a fair negative; identify the structural origin mismatch before adding learned calibration

Reviewed all changes since lead commit `38baef440a04eaca3cba1423a3956069b0fd3d22`: implementation commit `506b6016465cf983c30bb6d8de4a57378076e621` and report commit `4c1adf3fb82eb1f147650cae7587e886f911a0ac`; `AGENTS.md`; `pprtp/dual_space.py`, `pprtp/direct_prototypes.py`, `pprtp/run.py`, `tests/test_dual_space.py`, `scripts/report_h09a.py`; and `research_log/H09A/full` results/verification/artifacts.

H09-A is accepted as implemented and fair:

- Exactly two Codex commits occurred after the previous lead checkpoint.
- Full suite reports **46 passing tests**, preserving the previous 44.
- All three seeds reproduce the complete H07 aligned/native references and all ten historical H02-A online records before new scoring.
- The proposed prediction path has no target-label input. Test labels enter only after prediction for metrics and explicitly declared diagnostics.
- Raw owner prototypes, aligned bank and transforms are captured from the same side-effect-free H07 construction path; hashes match historical receipts.
- Model/server/client-prototype state, pre-existing gradients, CPU/CUDA RNG and module modes are unchanged; all scores are finite.
- No optimizer, threshold, learned gate, calibration scalar, temperature sweep or extra communication object was introduced.

The negative result is real: zero of three seeds satisfy the strong gate. Relative to H07 aligned readout, all accuracy changes are `-2.28/-0.38/-0.30pp`; seed0/2 sacrifice missing transfer, while seed1 sacrifices seen recognition.

The most important diagnosis is stronger than the report's generic phrase “calibration-limited.” H09-A mixes **uncentered cosine scores from two affine-related spaces**. The N256 map is

`T_i(z) = (z - mu_i) R_i + mu_ref`,

with orthogonal `R_i`. Cosine is invariant to `R_i` but **not** to the translations `mu_i` and `mu_ref`. Therefore `cos(z,p_i,c)` and `cos(T_i(z),g_c)` are not naturally commensurate. The observed near-one score saturation and seed-dependent group bias are exactly consistent with this structural origin mismatch. Do not add a learned calibration scalar before testing the mathematically implied translation-free form.

A useful identity gives the next falsifiable experiment. For any owned prototype `p_i,c`,

`cos(z-mu_i, p_i,c-mu_i) = cos((z-mu_i)R_i, (p_i,c-mu_i)R_i)`

up to floating-point error because `R_i` is orthogonal. Thus anchor-centering removes the arbitrary affine-origin mismatch while preserving the personalized owner geometry exactly under the correspondence rotation.

---

# ACTIVE — H09-B: anchor-centered residual dual-space classifier, seeds0/1/2

## One scientific objective

Test whether the H09-A failure is caused by the mathematically invalid raw-origin comparison rather than by weak components. Make **one** change only: compute both score groups in Procrustes-centered residual coordinates. No learned calibration, no threshold, no scalar alpha, no validation selection, no new training.

Reuse the exact H09-A/H07 round10 states, ordinary local prototypes, N256 anchors, canonical Procrustes transforms and global banks. Do not rerun H08 online training.

## Proposed classifier

For client `i`, let its historical N256 transform be

`T_i = (mu_i, R_i, mu_ref)`

where `T_i(z)=(z-mu_i)R_i+mu_ref`.

For a test feature `z`, define the centered residual feature

`r = z - mu_i`,

and its aligned residual

`r_aligned = r @ R_i = T_i(z) - mu_ref`.

For the two owned classes `c in C_i`, use the client's ordinary-local raw owner mean `p_i,c^raw` and score

`s_c = cosine(r, p_i,c^raw - mu_i)`.

For the eight missing classes `c not in C_i`, use the historical aligned global mean `g_c` and score

`s_c = cosine(r_aligned, g_c - mu_ref)`.

Predict `argmax_c s_c` over the ten assembled scores.

Call this arm `centered_dual_owner_seen_aligned_missing`.

This is not a fitted calibration. It simply removes the two arbitrary translation offsets already present in the frozen affine Procrustes map. Incremental communication relative to H07/H09 remains `0 B`: `mu_i`, `R_i`, `mu_ref`, owner means and global bank are already existing objects in the diagnostic path.

## Required exact mathematical checks

Before evaluating test labels, verify for every client and all ordinary-local owner prototypes that

`cos(pseudo_feature-mu_i, owner-mu_i)` and `cos((pseudo_feature-mu_i)@R_i, (owner-mu_i)@R_i)`

agree to tight float32 tolerance for actual feature batches. More directly, on the real test feature tensor, the two owned-class score columns computed in native-centered and aligned-centered coordinates must agree to `atol<=2e-5, rtol<=2e-5` (report worst absolute error). This proves the seen branch is coordinate-equivalent rather than a new learned geometry.

Also report the norms of `z`, `z-mu_i`, `p_i,c`, `p_i,c-mu_i`, `g_c`, and `g_c-mu_ref` to confirm the H09-A near-one saturation was dominated by common offsets rather than numerical zeros. Assert every centered prototype norm is finite and nonzero.

## Comparisons

For each seed report:

1. historical H07 `aligned_global_prototype_cosine`;
2. historical H09-A `dual_space_owner_seen_aligned_missing`;
3. new `centered_dual_owner_seen_aligned_missing`.

Add exactly one diagnostic control, not a competing method:

`centered_aligned_global_only`: for all ten classes score `cos(r_aligned, g_c-mu_ref)`.

This tells us whether any gain comes specifically from keeping the client-specific owner prototypes versus merely centering every aligned score. Do not add further arms.

For the proposed centered dual arm report the same H09-A per-client/per-class metrics, prediction histograms, predicted-class count, true-seen/true-missing max-owner vs max-missing score quantiles, and winning-group fractions. Test labels remain diagnostics only after prediction.

## Fairness / implementation constraints

- Reproduce H09-A/H07 construction receipts exactly before new scoring: raw owner hashes, N256 anchor receipt, `mu_i/R_i/mu_ref` transform hashes, global bank hash, model/server/client-prototype state.
- Do not use test labels in prototype construction, centering, score computation or routing.
- No optimizer, learned head, alpha, beta, threshold, temperature, z-score fit, Platt scaling, validation split, seed-specific constant, per-client tuned constant or sweep.
- Do not change anchor count, prototype source, global aggregation or training trajectory.
- Preserve state, existing gradients, RNG and module modes.
- Keep all centered tensors in the existing float32 evaluation dtype; no precision rescue is needed.
- Do not modify communication accounting except to state incremental `0 B` relative to H09-A/H07.

## Predeclared gate

Use the same strong target as H09-A so this is a clean repair rather than a moving goalpost. For each seed, relative to historical H07 aligned metrics `(S_a,M_a,A_a)`, require all of:

- `M_centered >= M_a - 2.0pp`;
- `S_centered >= S_a + 15.0pp`;
- `A_centered >= A_a + 2.0pp`;
- at least 9/10 predicted classes;
- exact historical receipts and finite/nonzero centered objects.

Call H09-B **strong** only if all three seeds pass.

Interpretation branches:

- **3/3 pass:** accept centered residual dual-space readout as the current strongest minimal PPRTP classifier. Next lead step should test deployment simplification / communication, not learned score calibration.
- **1–2/3 pass:** intermediate; stop and report which side still dominates. No tuning in this block.
- **0/3 pass, but component/oracle diagnostics remain strong:** conclude that removing the affine-origin mismatch is insufficient; only then is a small train-only calibration/gating study scientifically justified.
- **Centered global-only is already as good as centered dual:** do not claim personalized identity preservation from the dual branch; the gain is just centering.

## Tests

Preserve all existing 46 tests. Add only minimal tests for:

- exact centered owner-score invariance under an orthogonal Procrustes rotation;
- translation invariance of centered scores under arbitrary added native/reference offsets;
- exact owned/missing column assembly for the centered dual score matrix;
- no target-label dependency before prediction;
- side-effect/RNG/mode/gradient isolation and exact H09-A/H07 receipt reproduction.

Do not refactor unrelated code.

## Deliverable

Append `CODEX REPORT H09-B` with STATUS, source SHA, exact commands/run IDs, 46+ tests, exact historical reproduction receipts, three-seed result table, centered-global-only diagnostic, invariance errors, centered/raw norm diagnostics, score distributions, prediction histograms, communication statement, warnings and interpretation under the preregistered gate.

Do **not** add learned calibration, thresholding, gating, lambda/temperature sweeps, extra prototypes, new anchors, adapters, transport learning or communication optimization in H09-B. Await research-lead review.
