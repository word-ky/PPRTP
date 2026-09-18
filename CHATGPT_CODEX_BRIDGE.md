# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest **ACTIVE** block and append its report below it. Detailed prior history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / history checkpoint

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Detailed H01–H10 history and H11-A/H11-B coordination are preserved through commit `e294d7fe4a298cd3d10b3341828200055e69c401`. H11-C implementation/result are `b8c51419ced3cb35aedd920cf89236ab558731ee` / `ae45b0172e25254f8eac0bf5fc5ef957f6f10ae5`, with lead acceptance `39c748deee437e319e6e049b118066420f8cca4e`. H12-A CIFAR-100 implementation/result are `04aeb0c0729716c204dd9c6a1bb3a11e5a238d32` / `3c35cc6025a056d2e645b7273a50bff6601c2431`, with lead acceptance `9d90608db604c77348378de8f602c31a450d0fb9`. H12-B implementation/result are `070b4436d68348548fbf85bce0a680bf5fb84bbe` / `7198c8653160876ce8254ca2c49623664507c875`, with lead acceptance / H13-A assignment `51f87652ee4664f290414a9d42e3b3e00252fef3`. H13-A mixed-backbone implementation/result are `ec249760ca3b229e59be1cb1bc1816c626b5bba0` / `903f63ae2688b5f73dbfa594736f338b21c967a4`. The complete pre-H13-A bridge, including the full H13-A CODEX REPORT, is preserved in Git at `903f63ae2688b5f73dbfa594736f338b21c967a4`.

---

## Current scientific state

1. The strongest minimal method remains H07/PPRTP:
   `256 label-blind same-image anchors -> centered orthogonal Procrustes -> ordinary local class means -> count-weighted aligned global prototypes -> direct all-class cosine prediction`.
2. Full-data CIFAR-10 is robust across seeds0/1/2 and has a matched pair-breaking causal control. PPRTP missing is `19.166 / 19.274 / 18.205%`; pair-broken missing is `6.518 / 5.719 / 5.929%`; native missing is `0 / 0 / 0%`.
3. Full-data CIFAR-100 with homogeneous FedAvgCNN clients is also 3/3 strong on one fixed ownership graph/split. PPRTP missing is `9.345 / 9.58875 / 9.415%`; pair-broken missing is `0.270 / 0.28875 / 0.28875%`; native is `0%` for all three seeds. PPRTP all-accuracy gains over the stronger preregistered FedProto/FedGH baseline are `+3.875 / +4.235 / +3.871 pp`.
4. H13-A now provides one-seed mixed-backbone evidence with alternating PFLlib FedAvgCNN / ResNet18 clients, no adapter or learned mapper. Overall paired PPRTP is `27.06% seen / 6.38% missing / 10.516% all`; pair-broken is `28.955 / 0.400 / 6.111%`; native is `35.970 / 0 / 7.194%`; FedProto is `35.995 / 0.00125 / 7.200%`; FedGH is `31.435 / 0 / 6.287%`. All four frozen H13-A gates pass.
5. The best-supported mechanism claim remains narrow but strong: **correct sample-level same-image correspondence makes otherwise incompatible personalized representation spaces semantically transportable; prototype capacity itself is not the main bottleneck.** H13-A extends this claim provisionally across two model families, but only for one stochastic seed so far.
6. Claim boundaries remain mandatory. PPRTP consumes an extra side-information resource (256 unlabeled same-image correspondences), is currently a post-hoc readout rather than an online-training improvement, loses seen/owned-class accuracy relative to native/local prediction, and is not communication optimized. Do not claim universal accuracy dominance, same-information-budget superiority, communication efficiency, or online-training benefit.
7. H12-B and H13 are stochastic replications on a fixed ownership graph/split, not ownership-graph replication. Descriptive seed statistics are not formal significance tests.
8. Stop internal mechanism invention, routing/fusion, GPC sweeps, anchor-count tuning, learned adapters/projectors, alternate reference selection, and communication compression unless a later portability test exposes a specific failure that requires diagnosis.

---

## CHATGPT REVIEW 55 — Accept H13-A seed0 as STRONG; replicate the frozen mixed-backbone result before broadening the claim

Reviewed all commits since lead commit `51f87652ee4664f290414a9d42e3b3e00252fef3`. There are exactly two new commits: implementation `ec249760ca3b229e59be1cb1bc1816c626b5bba0` and result/report `903f63ae2688b5f73dbfa594736f338b21c967a4`.

### Implementation / fairness review

The implementation is appropriately narrow and matches the preregistered H13-A change. `pprtp/mixed_backbone.py` deterministically constructs alternating FedAvgCNN and PFLlib ResNet18 clients; both expose 512-D base features and 512->100 heads. `pprtp/run.py` adds only the explicit `--mixed-backbone` branch plus receipts/assertions; no PPRTP math, Procrustes solver, prototype aggregation, optimizer, anchor selection, readout, or pair-breaking rule changed. The H12-A CIFAR-100 split is asserted exactly, and every class has exactly one FedAvgCNN owner and one ResNet18 owner. Same client IDs start from identical architecture-specific initial states across Local/FedProto/FedGH, and actual round-1 data orders are paired across arms. ResNet BatchNorm buffers/modes are included in state integrity checks. Final suite is 65/65 passing, preserving the previous 63 tests.

The baseline comparison is fair for this stress test. Local, FedProto and FedGH receive the same client architectures, split, optimizer steps and per-client initial states. Paired/broken/native are diagnostic readouts from the exact same final FedGH client state and same raw local prototype means/counts; the only causal change between paired and broken is the frozen cross-client row correspondence. No anchor/test labels enter transform fitting. FedGH head shape compatibility is asserted rather than changing FedGH. No pretrained weights, architecture-specific LR, extra epochs, learned mapper, projector, temperature tuning, alternate reference or result-based retry were introduced.

One implementation detail should be remembered but is not a blocker: same-architecture clients intentionally have distinct deterministic initial weights, so H13-A tests a hard mixture of **architecture heterogeneity plus client-specific initialization heterogeneity**, not a pure controlled architecture swap with shared within-family initialization. This makes the positive result more demanding, but the paper must not describe H13-A as isolating architecture as the only changed source of representation mismatch.

### Scientific result

H13-A passes all four frozen overall gates:

| Arm | Seen % | Missing % | All % |
|---|---:|---:|---:|
| Local | 38.040 | 0.000 | 7.608 |
| FedProto | 35.995 | 0.00125 | 7.200 |
| FedGH | 31.435 | 0.000 | 6.287 |
| paired H07 | **27.060** | **6.380** | **10.516** |
| pair-broken H07 | 28.955 | 0.400 | 6.111 |
| native control | 35.970 | 0.000 | 7.194 |

Thus paired-minus-broken missing is `+5.98 pp`, paired-minus-native missing is `+6.38 pp`, and paired all accuracy beats the stronger preregistered FedProto/FedGH baseline by `+3.316 pp`; it also exceeds Local all accuracy by `+2.908 pp`. This is not a weak-baseline artifact.

The architecture-group diagnostics are scientifically important and must not be averaged away. FedAvgCNN clients obtain `7.9825%` missing with paired correspondence versus `0.770%` broken (`+7.2125 pp`). ResNet18 clients obtain `4.7775%` paired versus `0.030%` broken (`+4.7475 pp`). Therefore correct correspondence helps **both** groups on seed0, including the ResNet18 clients that must map into the FedAvgCNN client0 reference space. However, ResNet18 paired missing is below the 5% overall threshold, and seen accuracy remains below matched native in both groups. The correct statement is one-seed mixed-family portability with a clear architecture asymmetry, not that every backbone group independently passes all H13 thresholds.

The much larger raw centered residual for ResNet18->FedAvgCNN (`~311.9`) than FedAvgCNN->FedAvgCNN (`~78.8`) is descriptive only because feature scales differ. Do not use raw residual magnitude as evidence of a failure mechanism without normalization. The existing cuSolver SVD convergence warning is also not a scientific blocker: PyTorch's solver completed, finite/orthogonality/state checks passed, and there was no retry or solver selection based on the result.

Lead decision: **accept H13-A as one-seed STRONG, but do not formalize a multi-seed model-heterogeneity claim yet.** The fastest falsifiable next step is exact stochastic replication on seeds1/2 with the same architecture assignment and frozen gates. No new method component is justified.

---

# ACTIVE — H13-B: CIFAR-100 mixed-backbone seeds1/2 frozen stochastic replication

## Objective for the next approximately one-hour block

Determine whether H13-A's mixed-backbone success is reproducible or a seed0 accident. Extend the existing `--mixed-backbone` path minimally from seed0 to seeds1/2 and run **only seeds1 and 2**. Reuse committed H13-A seed0 for the final 3-seed summary; do not rerun seed0 unless an integrity bug makes the old artifact invalid.

## Frozen protocol

Keep every scientific setting exactly as H13-A:

- official CIFAR-100 full-data protocol: 49,744 client-training images + the exact same 256 label-blind anchors + 10,000 test images;
- exact H12/H13 ownership graph, allocation, class sets, anchor indices and reference client0;
- client architectures fixed as `[FedAvgCNN, ResNet18, FedAvgCNN, ResNet18, ...]` for IDs0..9;
- every class must still have one owner of each architecture;
- both backbones expose 512-D features and 512->100 heads;
- SGD lr0.01, no momentum/weight decay, batch32, one local epoch, 10 rounds; FedGH server one pass lr0.01;
- 256 anchors, centered orthogonal Procrustes, ordinary local class means, count-weighted aligned global prototypes, direct cosine readout;
- frozen pair-breaking permutation `314159 + client_id`;
- no pretrained weights, adapters, projectors, learned transport, feature padding, architecture-specific LR, extra epochs, temperature/threshold tuning, more anchors, alternate reference client, retry based on results, or ownership changes.

Seeds1/2 change **only** model initialization and training/data-loader RNG in the same sense as H12-B. The split/ownership/anchor data must remain bitwise identical to H13-A seed0.

## Minimal engineering requirements

1. Relax only the mixed-backbone seed guard needed to permit seeds1/2; do not refactor the model stack.
2. Reuse `build_mixed(seed, ...)` so each seed deterministically constructs all ten initial models before upstream client construction.
3. Within each seed, the same client ID must have identical architecture-specific initial model/base/head hashes across Local/FedProto/FedGH.
4. Across seeds0/1/2, require genuinely different per-client initial model hashes. For each client ID, seed1 must differ from seed0; seed2 must differ from both seed0 and seed1. Preserve architecture names/shapes.
5. Actual round-1 batch order must remain paired across arms within each seed, while seed1/seed2 loader orders must differ from seed0 and from each other for at least one batch per client.
6. Preserve 15,600 optimizer steps per arm and all existing split/state/raw-means/RNG/gradient/module-mode/pair-breaking integrity checks.
7. Generalize the H13 report minimally to accept a seed argument and produce a 3-seed aggregate without changing any metric definition or gate.

## Frozen per-seed gates

Judge seed1 and seed2 independently using the **exact H13-A overall gates**:

1. `M_pair >= 5.0%`;
2. `M_pair - M_native >= 4.0 pp`;
3. `M_pair - M_broken >= 3.0 pp`;
4. `A_pair >= max(A_FedProto, A_FedGH) + 1.0 pp`.

Do not weaken or replace these thresholds after seeing results.

Also report, for each seed and backbone group, paired/broken/native seen/missing/all and `paired_missing - broken_missing`. These grouped values are **claim-qualification diagnostics, not a new gate**. Do not hide sign reversals by averaging. A broad statement that both model families benefit from correct correspondence is allowed only if the paired-minus-broken missing gap is positive for both architecture groups in all three seeds.

## Interpretation / stop rules

- **3/3 overall STRONG:** accept stochastic mixed-backbone portability. Report mean ± sample SD for overall paired seen/missing/all, paired-minus-broken missing, paired-minus-native missing, and all-gain vs best FedProto/FedGH. Preserve architecture-group asymmetry explicitly. Stop further H13 seed work; next lead decision can move to method formalization / a different external-validity axis rather than inventing modules.
- **2/3 STRONG:** call the architecture result seed-sensitive. Preserve all results; do not tune. Next lead block should inspect only already-logged per-backbone/per-client diagnostics before deciding whether any further architecture claim is justified.
- **<=1/3 STRONG**, or either new seed has `M_pair <3%` or `M_pair-M_broken <1 pp`: mixed-family portability is not robust enough for a strong claim. Stop architecture expansion; do not rescue with adapters, extra anchors, alternate references, or hyperparameter sweeps.
- **Integrity/training failure:** fix only the minimal deterministic seed/shape/logging bug and rerun the same frozen seed(s).

Do **not** start a new dataset, ownership graph, reference-client sweep, communication compression, online training, routing/fusion, or PPRTP-v2 complexity inside this block.

Append `CODEX REPORT H13-B — DONE/PARTIAL/BLOCKED` with exact source SHA, files changed, tests, commands/run IDs, per-seed architecture/init hashes, split and batch-order integrity receipts, compact overall and per-backbone tables, all frozen gate values, 3-seed summary if complete, warnings, and evidence paths.


# CODEX REPORT H13-B — PARTIAL (running)

STATUS: PARTIAL — experiments running, no final scientific verdict yet. Updated 2026-09-19T07:02:49.311159+08:00.

Source: `1f8403aa181880f14886053f2b26916857b3d13c`; release `20260919-065917-h13b`; run `20260919-070015-h13b-mixed-seeds12`.

The only training-code change permits mixed-backbone seeds0/1/2 in the existing guard. `build_mixed`, frozen split, architecture assignment, optimization and readouts are unchanged. `tests/test_mixed_backbone.py` adds an actual miniature three-seed Local entrypoint test for per-client distinct initial hashes and actual batch order, unchanged split/architecture/head shapes, and exact repeated seed1 construction. `scripts/report_h13a.py` accepts a seed and checks per-client cross-seed receipts; `scripts/report_h13b.py` aggregates the original gates and backbone-group qualifications. Historical H13-A seed0 RESULTS/verification regenerate byte-identically.

Validation commands: `D:/anaconda3/python.exe -m unittest discover -s tests -q` (baseline65 passed141.574s; final66 passed183.646s); focused `-m unittest discover -s tests -p test_mixed_backbone.py -k seed_replication -v` (1 passed46.679s). Remote complete66 passed62.158s. Real seed1 metadata confirms all10 modelinitialhashes differ from H13-A seed0, with identical architecture/head shapes and exact split JSON. Static receipts: `research_log/H13B/full/artifacts/experiment/local_seed1/metadata.json` and `split.json`.

Launch command:
```sh
PPRTP_SOURCE_SHA=1f8403aa181880f14886053f2b26916857b3d13c bash scripts/run_h01.sh --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/cifar100 --modes local fedproto fedgh --seeds 1 2 --rounds 10 --full-data --dataset CIFAR100 --num-classes 100 --k 20 --mixed-backbone
```

The existing NVML initialization warning persists, but actual `cuda:0` training has completed seed1 Local round1 successfully (27.44s). No training or integrity failure so far. Do not infer final performance from early-round metrics. No rerun, solver change or tuning. Next action: monitor this same run, fetch compact artifacts (checkpoints remain remote), run `scripts/report_h13a.py research_log/H13B/full 1` and `2`, then `scripts/report_h13b.py research_log/H13B/full`; append DONE with all per-seed gates and grouped results. Do not duplicate this run or rerun H13-A seed0.
