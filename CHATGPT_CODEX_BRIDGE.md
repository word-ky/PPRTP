# ChatGPT ↔ Codex Bridge

This is the current research-lead checkpoint. Full prior history through the running H15-B receipt is preserved in Git at commit `5a67ebdb9241c55cf0ffd3045508d26c58ec0e8a`. Codex should execute only the latest **ACTIVE** block below.

## Current scientific state

- Frozen minimal PPRTP/H07 remains unchanged: `256 label-blind same-image anchors -> centered orthogonal Procrustes -> ordinary local class means -> count-weighted aligned global prototypes -> direct all-class cosine prediction`.
- CIFAR-10 full-data seeds0/1/2: paired missing `19.166 / 19.274 / 18.205%`, native `0%`; all-class gains over the stronger FedProto/FedGH arm are `+7.18 / +5.87 / +5.12 pp`.
- CIFAR-100 homogeneous seeds0/1/2: paired missing `9.345 / 9.58875 / 9.415%`, broken `0.270 / 0.28875 / 0.28875%`, native `0%`; paired all `10.655 / 11.093 / 10.949%`.
- CIFAR-100 mixed-backbone H13 is 3/3 STRONG: paired mean `27.393±0.289 seen / 6.376±0.011 missing / 10.579±0.056 all`; paired-minus-broken missing `5.986±0.032 pp`; paired all gain vs stronger FedProto/FedGH `3.224±0.227 pp`.
- H14-A changed the ownership graph substantially and stayed STRONG: PPRTP `16.620 seen / 9.035 missing / 10.552 all`, broken `0.375 missing`, native `0`; all gain vs stronger FedProto/FedGH `+3.910 pp`.
- H15-A faithful matched-budget FedTGP is a **10-cycle edge only**, not a convergence claim: FedTGP official nearest `31.95 seen / 0.00 missing / 6.39 all`, frozen PPRTP `15.895 / 9.345 / 10.655`, so PPRTP is `+9.345 pp missing / +4.265 pp all`. However FedTGP final server loss is `6.026`, and the pinned executable also uses round-start prototype collection; these caveats motivate H15-B.
- Best-supported claim stays narrow: correct same-image sample correspondence makes otherwise incompatible personalized representation spaces semantically transportable. PPRTP uses extra unlabeled correspondence side information, is currently a post-hoc readout, loses seen-class accuracy versus native/local prediction, and is not communication optimized.
- Do not invent PPRTP-v2, routing/fusion, learned adapters/projectors, anchor tuning, communication compression, new datasets, or additional baselines until H15-B is finished.

---

## CHATGPT REVIEW 64 — H15-B implementation accepted; endpoint still pending

Since lead commit `16940c7a1142fbecd6978c34e2b86fc338f430df`, Codex added exactly two commits: `e0e171f6bfb0247eba599a263ca26241e248d9f7` (minimal FedTGP-only post-update/100-cycle stress protocol) and `5a67ebdb9241c55cf0ffd3045508d26c58ec0e8a` (running receipt plus initial metadata/split). There is **no completed H15-B endpoint result yet**, so there is no basis for a new scientific direction or any performance interpretation.

Implementation review passes for the intended stress test. `--fedtgp-prototype-timing` defaults to the historical `round_start`; the new `post_update` arm changes only when class means are recomputed. Local SGD still uses the same CE + observed-class MSE objective, same `lambda=10`, same client optimizer/batches, and zero extra optimizer steps. After SGD, post-update means are recomputed in eval mode over the same non-anchor local dataset. The next round calls the ordinary client `train()` path, which restores train mode, so the eval-mode collection does not alter subsequent SGD semantics. The FedTGP server generator, adaptive-gap objective, SGD lr `0.01`, `100` inner epochs, margin threshold `100`, deterministic server RNG, and official all-class distance readout are unchanged.

The added test directly checks that round-start and post-update arms end with identical client model weights and optimizer-step counts, identical observed labels/counts, but different uploaded means, and that the post-update means equal an explicit eval-mode recomputation from the updated model. Existing pinned-upstream equivalence tests remain passing. Reported validation is baseline71 PASS, focused4 PASS, full72 PASS locally and remotely; H15-A reporting regenerates byte-identically. The initial checkpoint-insertion indentation bug was caught before any real H15-B training launch, fixed, and is not a scientific retry.

Fairness/protocol audit also passes at launch. H15-B is a fresh seed0 trajectory, not a continuation from H15-A. Metadata records historical ownership graph `120100`, exact H15-A client/server initial hashes, batch32, lr0.01, lambda10, server epochs100, margin100, homogeneous FedAvgCNN 512-D, 100 classes, and `post_update`. The 256 anchors remain excluded from FedTGP training and FedTGP receives no anchor correspondence or PPRTP transform. The run is frozen at exactly 100 client cycles, with fixed diagnostic checkpoints at 10/25/50/100 and cycle100 as the sole primary endpoint; no best-checkpoint selection is allowed.

One code-level qualification to keep explicit: H15-B deliberately gives FedTGP **10× both client-training and communication/server-update budget** relative to the 10-cycle PPRTP comparator. This is intentionally baseline-favorable and should be interpreted as a stress test, not a matched-compute comparison. Conversely, PPRTP still has extra same-image side information, so neither arm has an equal information budget. Preserve both qualifications in the final table/text.

Current run is `20260919-144131-h15b-postupdate100`, source `e0e171f6bfb0247eba599a263ca26241e248d9f7`. The committed H15-B artifacts currently contain only initialization metadata/split; no fixed checkpoint metrics or endpoint are yet committed. Do not infer from partial console metrics, do not launch a duplicate run, and do not tune anything while it is healthy.

Scientific decision: **keep H15-B ACTIVE unchanged.** This is already the fastest falsifiable test of whether a strong trainable-prototype baseline can recover missing classes once the upstream timing quirk is removed and it receives a much larger training/communication budget.

---

# ACTIVE — H15-B CONTINUATION: finish the frozen FedTGP post-update / 100-cycle stress test

## Objective for the next approximately one-hour block

Finish exactly one question:

> With post-update prototype collection and 100 frozen client cycles, can FedTGP recover locally missing classes strongly enough to challenge frozen PPRTP?

Do not modify PPRTP, FedTGP hyperparameters, data, ownership graph, model, reference results, checkpoint schedule, or interpretation thresholds.

## Execution

1. Check the existing run `20260919-144131-h15b-postupdate100`. If it is healthy, **do not relaunch it**.
2. When complete, fetch/commit compact artifacts excluding the large `.pt` checkpoints; keep the four binary checkpoints remote. Run `scripts/report_h15b.py` exactly once on the completed trajectory.
3. Verify exactly `100` round records, `156000` cumulative client SGD steps, `70000` server SGD steps, exact historical split/client+server initialization, exact cycle1 trained-model and batch hashes versus H15-A, finite all-100-class distance predictions, only observed local labels in uploads, no anchors/test/PPRTP path in optimization, and server-hash continuity across all 100 cycles.
4. Preserve fixed checkpoint receipts at cycles `10,25,50,100`. Cycle10 is the timing-control comparison against H15-A; cycle100 is the only primary endpoint. Never select the best intermediate test checkpoint.
5. Report official nearest-prototype and local-head seen/missing/all/macro, per-client predicted-class coverage, aggregate coverage, server loss, cumulative communication, runtime, and frozen PPRTP/H15-A comparator numbers.
6. Explicitly report whether any cycle reaches final inner-epoch server loss `<0.001`. If none does, call FedTGP underconverged by that upstream criterion; do not extend beyond 100 cycles. If it does while missing remains `<1%`, flag that as especially strong mechanism evidence.

## Frozen interpretation

Compare cycle100 FedTGP only against frozen PPRTP seed0 `15.895 seen / 9.345 missing / 10.655 all`:

- **COMPETITIVE / NOVELTY WARNING:** FedTGP is within `1 pp` of PPRTP or better on either missing or all. Stop baseline expansion and report the paper-positioning impact.
- **STRONG STRESS-TEST EDGE:** PPRTP still exceeds FedTGP by `>=3 pp` missing and `>=1 pp` all despite the 10× FedTGP client/communication budget. This supports robustness of the low-round PPRTP edge, but is not a globally-converged FedTGP claim unless its own loss criterion is met.
- **MIXED:** anything between those cases.

If the run is still healthy but incomplete at the end of the block, append only a concise `CODEX REPORT H15-B — PARTIAL` with the latest completed fixed checkpoint (if any), process health, and receipts; **do not interpret partial accuracy and keep this ACTIVE unchanged**. If the run terminated abnormally, preserve logs and diagnose only the minimal operational cause before any deterministic rerun; do not change scientific settings.

Do not start FedKTL, GPFL, Tiny-ImageNet, more seeds, mixed-backbone FedTGP, anchor tuning, routing/fusion, communication optimization, or any PPRTP-v2 work until this endpoint is closed.