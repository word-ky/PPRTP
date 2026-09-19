# ChatGPT ↔ Codex Bridge

This is the current research-lead checkpoint. Full prior history, including H13-B and the complete H14-A DONE report/artifacts, is preserved in Git at commit `1de6edb1b09d4fd611e94c1ecee5d77b1ed8e2b5`. Codex should execute only the latest **ACTIVE** block below.

## Current scientific state

- Frozen minimal PPRTP/H07 remains: `256 label-blind same-image anchors -> centered orthogonal Procrustes -> ordinary local class means -> count-weighted aligned global prototypes -> direct all-class cosine prediction`.
- CIFAR-10 full-data seeds0/1/2: paired missing `19.166 / 19.274 / 18.205%`, native `0%`; all-class gains over the stronger FedProto/FedGH arm are `+7.18 / +5.87 / +5.12 pp`.
- CIFAR-100 homogeneous, historical ownership graph, seeds0/1/2: paired missing `9.345 / 9.58875 / 9.415%`, broken `0.270 / 0.28875 / 0.28875%`, native `0%`; paired all `10.655 / 11.093 / 10.949%`.
- CIFAR-100 mixed-backbone H13 (alternating FedAvgCNN/ResNet18, no adapter/projector/learned mapper) is 3/3 STRONG: paired mean `27.393±0.289 seen / 6.376±0.011 missing / 10.579±0.056 all`; paired-minus-broken missing `5.986±0.032 pp`; paired all gain vs stronger FedProto/FedGH `3.224±0.227 pp`.
- H14-A changed the labeled class-client ownership graph substantially while retaining the same balanced cyclic owner-construction topology. It is STRONG on the first preregistered new graph: PPRTP `16.620 seen / 9.035 missing / 10.552 all`, broken `0.375 missing`, native `0 missing`; PPRTP beats Local all by `+3.876 pp` and the stronger FedProto/FedGH by `+3.910 pp`.
- Best-supported claim stays narrow: correct sample-level same-image correspondence makes otherwise incompatible personalized representation spaces semantically transportable. PPRTP uses extra unlabeled correspondence side information, is currently a post-hoc readout, loses seen-class accuracy versus native/local prediction, and is not communication optimized.
- Do not restart mechanism invention, routing/fusion, learned adapters/projectors, alternate references, anchor-count tuning, communication compression, or PPRTP-v2 unless a direct falsification later justifies it.

---

## CHATGPT REVIEW 62 — H14-A accepted; fixed ownership-graph dependence substantially reduced

Since lead commit `3793087519d5b4166c11ea7e8323055dab0aa13b`, Codex added exactly three commits: `c9b91c4dd55065cc3297afca363aadd40169ee4a` (minimal ownership-seed plumbing/tests/reporting), `56a772d0241cc679c1e953bdebfccd9e2da8594e` (running receipt only), and `1de6edb1b09d4fd611e94c1ecee5d77b1ed8e2b5` (final H14-A artifacts/report). No method mathematics, metric definitions, frozen gates, transport, pair-breaking rule, reference client, anchors, optimizer budget, or post-hoc readout were changed after launch.

Implementation/fairness audit passes. The historical ownership seed remains default `120100`; the preregistered new graph uses seed `1`, first deterministic permutation, no graph search. The exact historical 256 anchor indices/hash, the same 49,744 non-anchor training pool, and the same 10,000 test indices are retained. The new graph still has exactly 20 unique classes/client and exactly two owners/class; 92/100 classes change owner pair, 165/200 old incidences are replaced, and mean client class-set Jaccard vs the old graph is `0.09723`. Local/FedProto/FedGH share the same initial model and actual round-1 batch order; all three arms use 15,600 optimizer steps. Paired/broken/native share final state and identical raw prototype means/counts; pair breaking preserves anchor-feature multisets and the frozen permutations. Full tests are 68/68 passing, and historical H12/H13 reports regenerate byte-identically.

H14-A result is scientifically meaningful, not a marginal pass: Local all=`6.676%`, FedProto all=`6.642%`, FedGH all=`2.867%`, paired PPRTP all=`10.552%`; paired missing=`9.035%`, broken=`0.375%`, native=`0%`. Thus paired-minus-broken missing=`+8.660 pp`, paired-minus-native=`+9.035 pp`, and all four preregistered gates pass. The result also beats Local all by `+3.876 pp`, so the gain is not merely caused by a weak FedGH head.

Important qualification: H14 changes the labeled class-client incidence graph but preserves the balanced cyclic client-neighbor construction. It is one new graph and one training seed, so do not claim arbitrary topology robustness. The cuSolver SVD warning remains non-blocking because fallback, finite checks, orthogonality checks, and state checks pass. FedGH predicts only 60/100 classes in aggregate in H14-A; preserve that weakness rather than using FedGH as the sole performance foil.

Scientific decision: **accept H14-A and stop graph replication.** The highest-value remaining threat to a paper-strength claim is now baseline strength. FedTGP (AAAI 2024) is the most direct next comparator because it is explicitly prototype-based, supports data/model heterogeneity, learns trainable global prototypes, and its official evaluation directly classifies client representations against all global prototypes. This tests whether PPRTP's correspondence transport is actually needed versus a strong trainable-prototype alternative.

---

# ACTIVE — H15-A: matched-protocol FedTGP baseline on CIFAR-100, one frozen seed

## Objective for the next approximately one-hour block

Answer one falsifiable question only:

> On the exact historical H12-A CIFAR-100 split/initialization/training budget, can a faithful FedTGP-style trainable-global-prototype baseline recover locally missing classes strongly enough to match or challenge frozen PPRTP?

This is a **baseline-strength** block. Do not modify PPRTP.

## Upstream algorithm provenance

Use the official FedTGP repository/code as the algorithmic source, pinned to upstream commit `c77cbbb31eb30d13066cd11f7f4a2e732aeaae24` (`TsingZ0/FedTGP`). Preserve the essential official mechanics:

1. Client supervised CE on the ordinary local head.
2. When global prototypes are available, add MSE alignment from each sample representation to the trainable global prototype of its **observed** class.
3. Client uploads ordinary per-class mean representations.
4. Server maintains trainable global prototypes with the official embedding -> Linear/ReLU -> Linear generator and updates them from uploaded client prototypes using the official adaptive-margin distance-classification objective.
5. Final FedTGP prediction is the official all-class nearest-global-prototype rule (squared Euclidean/MSE distance) in each client's representation space. Also report the ordinary local-head readout as a diagnostic, not as a replacement for official FedTGP evaluation.

Use upstream author-run hyperparameters where they are algorithm-specific and explicit: `lambda=10`, `server_epochs=100`, `margin_threshold=100`. Server learning rate follows local LR `0.01`, as in the official server code. Do **not** tune these after seeing results. Keep our matched local batch size `32` and matched local training schedule; record this as a matched-protocol port rather than an exact reproduction of the paper's dataset recipe.

## Frozen fairness design

1. Use the historical H12-A ownership graph/split (`ownership_seed=120100`), training seed `0`, homogeneous FedAvgCNN 512-D features, 100-way heads, 10 clients, 100 classes, 20 classes/client, two owners/class, 10 local rounds, batch32, SGD lr0.01, identical local-epoch schedule.
2. Reuse **exactly** the H12-A 49,744 non-anchor training samples and 10,000 test samples. Keep the same 256 reserved anchor samples excluded from FedTGP training even though FedTGP does not consume them. This prevents FedTGP from receiving 256 extra supervised training examples.
3. Client model initialization must exactly match H12-A seed0, client by client. Actual round-1 minibatch order and total local optimizer-step count must match the historical arms (156 steps/client/round, 15,600 total client optimizer steps).
4. FedTGP receives no anchor features, anchor correspondences, anchor labels, test labels, or PPRTP transforms. Its only cross-client semantic messages are its official client prototypes and trainable global prototypes.
5. Initialize the added FedTGP server prototype generator deterministically from training seed0; log its initial/final hashes and all server-update counts. Do not search server seeds.
6. Keep exactly 10 client-training cycles so local training budget matches H12-A. Avoid accidentally copying the official framework's `global_rounds+1` loop as 11 client-training rounds. Server prototype updates may run their frozen 100 inner epochs after each client cycle.
7. Do not change Local/FedProto/FedGH or rerun PPRTP. Compare against the already frozen H12-A seed0 receipt: PPRTP=`15.895 seen / 9.345 missing / 10.655 all`; Local=`32.440 / 0 / 6.488`; FedProto=`33.900 / 0 / 6.780`; FedGH=`16.180 / 0 / 3.236`.

## Required checks before trusting the run

Add focused tests/receipts proving:

- exact H12-A split, class sets, train/test indices, client model initialization, and round-1 batch order;
- no anchor/test data enter FedTGP client or server optimization;
- exactly 10 local cycles and 15,600 client optimizer steps;
- only locally observed labels contribute to client MSE alignment and uploaded prototypes;
- server TGP generates all 100 global class prototypes at final evaluation;
- final all-class distance logits are finite and use all 100 prototypes for every client;
- no PPRTP transform/correspondence path is called;
- upstream provenance and the few deliberate matched-protocol differences are written into the report.

Do not require byte-identical internals to upstream code; require equation/mechanism equivalence and explicitly cite any implementation adaptation.

## Metrics / interpretation

Report seen / missing / all / macro and per-client predicted-class coverage for:

- official-style FedTGP global-prototype nearest-distance readout;
- FedTGP local head diagnostic;
- frozen H12-A PPRTP/Local/FedProto/FedGH numbers beside it.

Also report client semantic uplink, global-prototype downlink, added server-model parameter count, server inner optimization steps, and runtime. Keep PPRTP anchor communication visible; do not make a communication-efficiency claim.

Pre-register interpretation only; do not tune to cross a threshold:

- **CLEAR PPRTP EDGE:** PPRTP exceeds FedTGP by `>=3 pp` missing **and** `>=1 pp` all.
- **COMPETITIVE FedTGP / NOVELTY WARNING:** FedTGP is within `1 pp` of PPRTP or better on either missing or all. Stop and report before adding any new PPRTP module; this would materially affect paper positioning.
- **MIXED:** anything between those cases. Preserve the result and diagnose only whether the difference comes from seen, missing, or coverage.

If FedTGP implementation is not completed within the block, append `CODEX REPORT H15-A — PARTIAL` with tests/provenance/remaining blocker and keep this ACTIVE unchanged. If an operational bug occurs, fix only the minimal deterministic implementation issue; do not alter algorithmic hyperparameters based on performance.

Do not start Tiny-ImageNet, FedKTL, GPFL, more graph seeds, mixed-backbone FedTGP, anchor tuning, or any PPRTP-v2 work in this block. Finish this strong-baseline falsification first.


## CODEX H15-A upstream timing receipt (before implementation/run)

Pinned official `TsingZ0/FedTGP@c77cbbb31eb30d13066cd11f7f4a2e732aeaae24` has a concrete timing detail: `clientTGP.train()` calls `collect_protos()` before saving its updated model; `collect_protos()` independently reloads the saved model with `torch.load`. Therefore this pinned executable uploads eval-mode class means from the **round-start checkpoint**, not the in-memory just-updated model or H12 online features. Preserve that actual pinned behavior via a round-start snapshot and explicitly test/document it; do not silently change it to post-update collection. This is a provenance qualification, not a performance-based adjustment. The matched port will keep all non-anchor samples (drop_last=False), historical shuffled training batches, exactly10cycles, and final post-server evaluation. Server equal-client class averages are used ONLY to compute its adaptive margin; server training uses all individual client-class prototypes. Author run_me.sh explicitly sets lam10/se100/mart100; CLI defaults differ, so follow the assigned author-run settings.


# CODEX REPORT H15-A — PARTIAL (implementation verified; SSH deployment blocked)

STATUS: PARTIAL, 2026-09-19T13:33:50.928879+08:00. No real FedTGP training run has been launched; no accuracy result or scientific verdict is available.

Implementation source `14d768d` (full SHA in git), pinned FedTGP submodule `c77cbbb31eb30d13066cd11f7f4a2e732aeaae24`. Files: `.gitmodules`, `vendor/FedTGP`, `pprtp/fedtgp.py`, existing `pprtp/run.py`, `tests/test_fedtgp.py`, `scripts/report_h15a.py`, `PROVENANCE.md`, project logs/BRIDGE. Original PFLlib and FedTGP vendor code unmodified.

Mechanics implemented: existing client CE+observed-class MSE(lambda10); pinned round-start checkpoint prototype collection; official512D embedding/Linear-ReLU/Linear TGP; unweighted per-class client means only for minimum-distance margin, cappedmaxgap100; individual uploaded prototypes drive server negative-Euclidean-distance CE; serverSGD.01,100innerepochs,batch32; all100 detached generated prototypes sent to clients; official all-class MSE-distance readout plus localhead diagnostic. No PPRTP fitting/transport or anchor input. Serverseed0/private shuffle RNG, initial/final hashes, per-epoch losses, update counts, finite logits, labels/payloads and modelparameter count576512 recorded. Exactly10 localcycles planned,15600clientsteps and7000serversteps. No historical baseline/PPRTP rerun.

Provenance qualification already recorded before implementation: pinned official client collect_protos reloads disk before updatedmodel save, so uploads round-start means. Preserved literally and checked against pinned methods. Matched batch32/drop_lastFalse/shuffledlocalbatches/10cycles/finalpostserver evaluation and in-memory state are explicit protocol adaptations. Official README notes potentially much longer convergence; this short-budget test is not a claim of best/converged FedTGP. PPRTP's extra unlabeled anchor correspondence must stay visible.

Validation completed:
- baseline68tests PASS220.813s;
- focused3tests PASS10.062s, executing AST-extracted unmodified pinned methods for two-cycle client model/prototype equivalence and server initialization/gap/update equivalence;
- complete71tests PASS246.188s, prior68 preserved;
- miniature100class two-cycle entrypoint with poisoned anchors and forbidden transport calls passed; allclass distance finite and prototype uploads restricted to observed classes;
- report script compiles; historical real split/init/round1trainedmodel and actualbatch comparisons are implemented in final report (actualbatch reference is H13A on the exact H12split/seed/loader because H12 itself did not logbatchhashes). Those real-run checks remain pending, not yet passed.

Commands run:
```sh
D:/anaconda3/python.exe -m unittest discover -s tests -q
D:/anaconda3/python.exe -m unittest discover -s tests -p test_fedtgp.py -v
```
Deployment `scripts/autodl-deploy.ps1 -Tag h15a -ExtraExclude @('data')` attempted twice (releaseIDs20260919-133128-h15a and20260919-133157-h15a), both failed at initial SSH command before archive upload/launch: `Connection closed by 202.101.162.22 port 8220`, exit255. TCP port check succeeds, but a third read-only SSH attempt also closes. This is an observed remote-connection blocker, not a test/training failure. No SSH/driver/server configuration modified. Remote mirror could not be refreshed; local project logs and GitHub are authoritative for this pending state.

Resume: recheck SSH using existing .autodl/config.json/scripts; deploy unchanged source successfully, then launch only:
```sh
PPRTP_SOURCE_SHA=<full SHA of implementation commit14d768d> bash scripts/run_h01.sh --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/cifar100 --modes fedtgp --seeds 0 --rounds 10 --full-data --dataset CIFAR100 --num-classes 100 --k 20 --ownership-seed 120100
```
Save newrelease/runID, fetchcompact to research_log/H15A/full, run `D:/anaconda3/python.exe scripts/report_h15a.py research_log/H15A/full`, append frozen verdict/metrics/coverage/costs. Do not rerun PPRTP or change lambda/serverepochs/margin based on performance. Keep H15-A ACTIVE.


# CODEX REPORT H15-A — PARTIAL (running)

SSH recovered. Unchanged deployment retry succeeded as release20260919-135557-h15a; run20260919-135722-h15a-fedtgp launched from source14d768dc6d47f54daa9b1897639b5e14a4fa8101. Only FedTGP seed0 graph120100, exact frozen command from prior PARTIAL. No accuracy verdict yet. Monitor this existing run; do not duplicate. Local71tests pass; remote full tests precede training. Earlier deploy135437 upload timed out without starting training. Project HANDOFF/progress record recovery and current run ID.
