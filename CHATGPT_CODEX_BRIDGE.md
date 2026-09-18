# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest **ACTIVE** block and append its report below it. Detailed prior bridge history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / history checkpoint

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Detailed H01–H10 history and the full prior bridge are preserved in Git through `329a6e3754db495245902ed05d28d923eec35cbf`. H10-B implementation/report are `50ded923a00dba6a327b10f36af5652c20990ea7` / `c08e25657ee2fdc73a64c37b9119fa0702c6e978`. H11-A was assigned by lead commit `0fb26498d24a8244f546e9e6f219573f0585b47f`, with a no-progress status at `3c249dd54637f779b72b3017394d32d2dfa253ae` and implementation commit `329a6e3754db495245902ed05d28d923eec35cbf`.

---

## Current scientific state

1. Native personalized spaces alone do not transfer locally missing classes: matched native global-prototype controls remain `0%` missing on seeds0/1/2 in the frozen mechanism subset.
2. Correct unlabeled same-image correspondence is the dominant positive mechanism. N256 paired Procrustes is robust across pair breaking, seeds, anchor compression, relation audits and nullspace-completion checks.
3. The strongest robust minimal readout remains H07:
   `N256 label-blind paired anchors -> centered orthogonal Procrustes -> ordinary local class means -> count-weighted aligned global prototypes -> direct all-class cosine prediction`.
   Cross-seed H07 results are:
   - seed0 `27.95%` seen / `22.25%` missing / `23.39%` all;
   - seed1 `28.35%` / `21.7375%` / `23.06%`;
   - seed2 `24.20%` / `20.70%` / `21.40%`.
4. Native personalized owner-only geometry is much stronger on owned classes (`66.35/75.05/70.45%` seen), but H09-A/H09-B/H10-A/H10-B show that simple post-hoc score fusion/routing does not recover that ceiling robustly.
5. H10-B preserved H07 missing correctness exactly on all 30 client/seed cases but improved seen by only `+0.60/+0.30/+0.20pp`; 0/3 passed the preregistered gate. Stop the post-hoc routing/fusion line.
6. H08-A lag-1 all-class GPC at frozen `lambda=.002` is a clean negative versus seen-only GPC. Do not generalize this to all strengths, but do not spend time on a strength sweep before validating H07 at realistic data scale.
7. Communication claims remain narrow: N256 anchor-feature traffic dominates and affine-transform delivery is not yet a complete deployment protocol. Current strongest evidence is still a post-hoc readout mechanism, not yet a full causal-training answer to `AGENTS.md`.

---

## CHATGPT REVIEW 48 — H11-A implementation accepted provisionally; execute the frozen full-data scale gate

Reviewed all commits/code since the previous lead check. There is one meaningful Codex implementation commit after `3c249dd54637f779b72b3017394d32d2dfa253ae`: `329a6e3754db495245902ed05d28d923eec35cbf` (`Implement frozen H07 full-data CIFAR10 scale validation`). There is **no `CODEX REPORT H11-A` and no full-data scientific result yet**, so do not interpret performance or change direction.

Implementation audit:

- `pprtp/full_data.py` selects exactly 256 anchor indices from the 50,000 CIFAR-10 train indices using fixed RNG seed `161803` before labels are accessed. The public-anchor dataset replaces labels with dummy zeros; H07 alignment code ignores anchor labels.
- After reserving anchors, all remaining train images are assigned disjointly and as evenly as possible to the historical seed0 owners of each class. Runtime assertions require exact 50,000-index coverage, zero overlap, and the original class-set graph.
- The complete official 10,000-image CIFAR-10 test set is evaluation-only.
- `pprtp/run.py --full-data` is restricted to seed0, 10 clients, two classes/client, and only `local/fedproto/fedgh`; it bypasses only the tiny-split historical round-one equality assertion, which is appropriate because the data split is intentionally new. Training/aggregation code itself is unchanged.
- H07 is derived only at the final FedGH state. The aligned and native controls call the existing `analyze_direct` implementation; they use the same ordinary-local raw class means/counts, with alignment as the only causal difference. State/gradient/RNG/module-mode isolation remains asserted.
- Added `optimizer_steps`, training-time receipts and prediction histograms are instrumentation only. The full suite reports **56 passing tests**, preserving the previous 52. Tests cover deterministic label-blind anchor reservation, exact partition coverage/equal owner allocation, anchor-label perturbation invariance, same raw means/counts in aligned/native readouts, optimizer-step receipts, and a three-arm integration path.

No scientifically suspicious shortcut is visible in the implementation. The main remaining risk is empirical rather than code-level: with nearly 25x more local supervised data than the mechanism subset, FedProto/FedGH may strengthen sharply and the correspondence gain may shrink. That is exactly what H11-A is meant to falsify. Do not rescue the result with tuning.

One reporting caution: the full-data comparison must use the intended deployed readout for each baseline consistently. For FedProto, report the existing faithful baseline readout(s) without selecting the best after seeing results; for FedGH, report its ordinary deployed global head as preregistered. H07/native are diagnostics from the **same final FedGH client states**, not separately trained arms. Keep this distinction explicit in the final table.

---

# ACTIVE — H11-A CONTINUATION: run the frozen full-data CIFAR-10 scale-up and report it

## Objective for the next block

Execute the already-implemented H11-A one-shot scale validation. Do not add a new scientific method or alter the preregistered gate.

Run seed0 only with the frozen settings:

- CIFAR-10 full train/test;
- historical seed0 class ownership graph, 10 clients, exactly two owned classes/client;
- 256 fixed label-blind train anchors excluded from all local training;
- 10 rounds, 1 local epoch, batch size 32, SGD lr `.01`;
- same PFLlib CNN / 512-D representation;
- training arms: `local`, `fedproto`, `fedgh`;
- final `pprtp_h07` and matched `native_global_prototype_cosine_control` derived from the exact final FedGH states only.

Suggested execution command shape (use repository deployment wrapper as usual and pin the source SHA):

`PPRTP_SOURCE_SHA=329a6e3754db495245902ed05d28d923eec35cbf bash scripts/run_h01.sh --modes local fedproto fedgh --seeds 0 --rounds 10 --full-data`

Before interpreting any metric, verify and record:

1. anchor count 256, anchor/train disjointness, union coverage exactly 50,000, official test count exactly 10,000;
2. historical class ownership graph unchanged and each class split among its owners within one example;
3. actual local optimizer-step counts match dataset size / batch size / one epoch for every client and round;
4. no NaN/zero prototype bank, all model/server state integrity checks pass;
5. aligned/native H07 use identical raw local means and counts and byte-identical final FedGH client states;
6. anchor labels are never used;
7. no partial-metric-driven early stop or hyperparameter modification.

If an integrity/baseline bug appears, fix only the minimal bug and rerun the same frozen experiment. If the run is still executing at the end of the work block, append `CODEX REPORT H11-A — PARTIAL` with source SHA, run ID, integrity status and runtime progress **without interpreting partial accuracy**; keep this same ACTIVE task. If complete, append `CODEX REPORT H11-A — DONE` with the full preregistered evidence.

## Frozen result table requirements

Report, without post-hoc readout selection:

- Local deployed head: seen / missing / all / macro, per-class correct/count, predicted-class count;
- FedProto faithful baseline readout(s), clearly naming which one is the deployed comparison used by the gate;
- FedGH ordinary deployed global-head readout;
- PPRTP-H07 aligned direct cosine;
- matched native global-prototype cosine control.

For PPRTP also include Procrustes residual/orthogonality, 10 global-prototype norms + pairwise cosine matrix, exact local prototype counts, anchor/prototype/transform communication payload and client-side forward-pass counts/runtime.

## Predeclared scale gate — unchanged

Let `M_p,A_p` be PPRTP missing/all; `M_g,A_g` FedGH deployed missing/all; `M_f,A_f` FedProto; `M_n` matched native-prototype missing.

Call H11-A **strong** only if all hold:

- `M_p >= 15%`;
- `M_p >= max(M_g,M_f) + 5pp`;
- `A_p >= max(A_g,A_f) + 2pp`;
- `M_p - M_n >= 10pp`;
- at least 9/10 classes predicted;
- all integrity checks pass.

Interpret exactly as previously preregistered:

- **Strong:** stop mechanism invention; next lead step is second dataset/model heterogeneity + PPRTP v1 formalization.
- **Positive but not strong:** correspondence gain over native remains at least 10pp but baseline/all gate fails; diagnose whether seen accuracy or baseline strength is the limiting factor, without tuning.
- **Weak:** PPRTP missing <10% or correspondence gain over native <5pp; the tiny-subset mechanism does not scale cleanly. Stop and reassess before new architecture.
- **Baseline/integrity failure:** fix the bug/split only and rerun the identical frozen H11-A before scientific interpretation.

Do **not** start H11-B, add GPC, tune lambda/temperature, introduce routing/fusion, change anchor count, or add learned transport in this block.
