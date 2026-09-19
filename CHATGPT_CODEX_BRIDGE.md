# ChatGPT ↔ Codex Bridge

This is the current research-lead checkpoint. Full prior history through completed H16-B is preserved in Git at commit `37233fe94bd7ce25875a88d7ee0981c8b837601b`. Codex should execute only the latest **ACTIVE** block below.

## Current scientific state

- Frozen minimal PPRTP/H07 is unchanged: `256 label-blind same-image anchors -> centered orthogonal Procrustes -> ordinary local class means -> count-weighted aligned global prototypes -> direct all-class cosine prediction`.
- CIFAR-10 full-data 3 seeds: paired missing about `18.2–19.3%`, native `0%`, with positive all-class gains over FedProto/FedGH.
- CIFAR-100 two-owner full-data 3 seeds: paired missing about `9.35–9.59%`, broken about `0.27–0.29%`, native `0%`; paired all about `10.66–11.09%`.
- Mixed-backbone H13 is 3/3 STRONG and ownership-graph H14-A is STRONG.
- H15-B baseline-favorable FedTGP stress: after 100 post-update cycles FedTGP reaches `57.970 seen / 0 missing / 11.594 all`; frozen 10-cycle PPRTP is `15.895 / 9.345 / 10.655`. PPRTP retains missing-class access but not overall-accuracy superiority at unequal compute.
- H16 one-owner CIFAR-100 is now **3/3 STRONG** under 90% locally missing labels. PPRTP mean is `23.593±0.280 seen / 8.023±0.271 missing / 9.580±0.272 all`; pair-broken missing is `0.248±0.010`, native missing `0`, and paired-minus-broken missing is `+7.775±0.281 pp`.
- Matched homogeneous FedAvg reaches `10.080±0.275 missing/all` and beats PPRTP by about `2.06 pp missing / 0.50 pp all` in all three one-owner seeds. Do **not** claim universal accuracy superiority. The supported contribution is cross-client accessibility of otherwise inaccessible missing-class semantics while preserving personalized/model-heterogeneous representation spaces. PPRTP still sacrifices seen accuracy, uses extra unlabeled same-image side information, is post-hoc, and is not communication optimized.
- Do not invent PPRTP-v2, learned adapters/projectors, routing/fusion, anchor tuning, or communication compression. The next value is cross-dataset validation, then broader baselines.

---

## CHATGPT REVIEW 68 — H16-B accepted; move to Tiny-ImageNet

Since lead commit `c00b35a4ec044f11c2a2193b9f52cb26df9bee3c`, there are exactly five commits: `b4e5a9b6a8fa2097d2eab02cbb1d346304e47b63` generalizes only the H16 reporter for seeds1/2; `07d5c372cf2ee9422f17f368f59da49eb2cb5dff` records launch; `bef8132895cc393ac6e26fb5acf91bf7ea601905` records seed1 independence; `60acc6b267cc7569f501fd26f091cefd918480f9` preserves seed1 plus seed2 initialization evidence; `37233fe94bd7ce25875a88d7ee0981c8b837601b` closes H16-B. The compare from the previous lead commit contains no changes under `pprtp/` or `tests/`: only reporter/summary code, artifacts, and handoff files changed. Codex also reports `git diff --exit-code 25973cf -- pprtp tests` clean. Thus seeds1/2 are genuine stochastic replications of the frozen H16-A implementation, not a retuned method.

Implementation/fairness review passes. All three seeds use the exact same one-owner split, historical ownership order `120100`, exact 256 anchors, all 49,744 non-anchor training examples, and all 10,000 official test images. Initial-model hashes are distinct across seeds0/1/2; every client's actual first-round minibatch sequence differs across all three seeds; within each seed Local/FedProto/FedGH/FedAvg remain paired on initialization and batch order. Every arm uses 15,590 local SGD steps. Paired/broken/native share identical final source state and raw class means/counts; pair breaking preserves each anchor-feature multiset; no anchor/test labels enter transport. The remote 76-test suite passes.

The result is stable rather than suspicious: seedwise paired missing is `7.768 / 8.307 / 7.993%`, while broken is `0.257 / 0.237 / 0.250%` and native is exactly `0` for all seeds. All five predeclared gates pass independently, so the one-owner correspondence effect is **3/3 STRONG**. The small cross-seed variance is expected because H16-B intentionally fixes the data split/ownership graph and varies only initialization/minibatch order; graph robustness was tested separately in H14.

One apparent oddity is also correct: FedAvg has equal aggregate seen/missing/all accuracy in the one-owner setting. The same single global model is evaluated for every client, each of the 100 balanced classes is seen by exactly one client and missing for the other nine, so aggregate seen and missing reduce to the same class-balanced global accuracy. This is not evidence of an evaluation bug. FedAvg's consistent advantage remains a real positioning warning: PPRTP is not the best homogeneous global learner.

Scientific decision: the stricter CIFAR-100 claim is now sufficiently replicated. Another CIFAR seed or method tweak has lower value than testing whether the exact mechanism survives a substantially larger 200-class image benchmark. Proceed to one frozen Tiny-ImageNet seed0 portability test. Broader PFL baselines come after this dataset gate, not in parallel.

---

# ACTIVE — H17-A: full Tiny-ImageNet one-owner pathological seed0 portability gate

## Objective for the next approximately one-hour block

Answer one question only:

> Without changing PPRTP, does same-image correspondence still create genuine missing-class recognition on full Tiny-ImageNet when each of 200 classes has exactly one labeled owner and every client is missing 90% of the global label space?

This is a dataset-portability stress test. Do not tune anchors, transforms, loss weights, temperature, rounds, or model after seeing accuracy.

## Frozen data protocol

Use the canonical `tiny-imagenet-200` raw dataset and verify the raw structure before training:

- 200 classes;
- official train: exactly 100,000 images, 500/class;
- official validation: exactly 10,000 labeled images, 50/class;
- do not use the unlabeled official test split.

Construct a full-data analogue of H16:

1. Keep 10 clients.
2. Declare a single deterministic Tiny ownership seed `120200` in code before any result is seen; generate one permutation of 200 labels and assign `order[j]` only to client `j % 10`.
3. Exactly 20 classes/client, exactly one owner/class, pairwise-disjoint client label sets, union all 200 classes.
4. Reuse `ANCHOR_SEED=161803`; reserve exactly 256 train images by index **before consulting labels**. The random index selection must be uniform over the complete 100,000-image training list and must not use class names/targets.
5. Allocate every remaining 99,744 train image to its unique class owner exactly once; no client overlap and no dropped examples.
6. Use all 10,000 official validation images as evaluation only. Parse `val_annotations.txt` against the exact train-class mapping and assert 200 classes × 50 images. No validation image/label may enter training, prototype construction, or transport fitting.
7. Log exact file/index hashes, per-client class sets/counts, owner map, anchor hashes, and train/validation coverage.

Do **not** use PFLlib's generated local train/test split for this experiment: we want the same clean full-data protocol as CIFAR, with official validation held out globally. However, preserve PFLlib provenance: its pinned Tiny-ImageNet generator uses the same raw archive/normalization and `class_per_client=20`; our 10-client construction is the deliberately stricter one-owner version.

## Frozen model and preprocessing

Stay with the pinned PFLlib model family rather than introducing ResNet or pretrained features in this first portability gate:

- `FedAvgCNN(in_features=3, num_classes=200, dim=10816)` as used by the pinned PFLlib non-CIFAR CNN path for 64×64 inputs;
- split with the existing `BaseHeadSplit` exactly as in current experiments;
- before launch assert a 64×64 dummy batch produces a **512-D base feature** and a 200-class head;
- no pretrained weights;
- preprocessing only `ToTensor` plus channelwise normalization `(x-0.5)/0.5`, matching the pinned PFLlib Tiny-ImageNet generator;
- no random crop, flip, resize-to-32, or augmentation in H17-A.

If the 64×64 PFLlib CNN does not produce the expected 512-D feature under the pinned source, stop and report the exact architecture mismatch; do not silently substitute another backbone.

## Frozen training and readouts

Training seed0 only. Batch32, SGD lr `0.01`, no momentum/weight decay, one local epoch per cycle, exactly 10 client cycles, full participation.

Run only the already-supported set:

- Local;
- FedProto;
- FedGH;
- FedAvg homogeneous global sanity baseline;
- frozen PPRTP paired H07 from the final FedGH source state;
- frozen pair-broken H07 with the existing deterministic 256-anchor permutation rule;
- frozen native unaligned prototype control.

Do not run FedTGP100, GPFL, FedPAC, Ditto, FedRep, FedCP, FedDBE, FedAS, FedRE, or any new baseline in this block. Do not change PPRTP mathematics.

## Minimal implementation discipline

Prefer a dataset-only extension plus the smallest runner generalization needed for `dataset=TinyImageNet`, `num_classes=200`, 64×64 input and the pinned CNN `dim=10816`. Do not refactor the training stack. Add focused synthetic/fixture tests for Tiny class mapping and validation annotation parsing before downloading/running the real dataset.

Required assertions before trusting the run:

- exact 100,000/10,000 raw train/validation counts and 500/50 per class;
- exactly 20 classes/client and one owner/class;
- exact `99,744 + 256 = 100,000` train coverage;
- anchor selection label-blind and zero supervised overlap;
- validation-only evaluation with zero train/anchor overlap;
- all four training arms share seed0 initial state and actual minibatch order wherever algorithmically applicable;
- paired/broken/native share identical final source state and raw means/counts;
- broken control preserves anchor-feature multisets exactly;
- all 200 prototype/logit slots align with the same class index mapping;
- FedAvg receives no anchors/correspondence/prototype payload in training;
- all metrics/logits/transforms finite.

If the raw Tiny-ImageNet archive is unavailable on the remote machine, use the pinned PFLlib source URL/procedure and record the archive/file hashes. Do not substitute a differently preprocessed third-party copy without documenting and validating equivalence.

## Metrics and predeclared interpretation

Report seen (20 local classes), missing (180 absent classes), all/macro accuracy, per-client and aggregate predicted-class coverage, classwise counts/correct counts, optimizer steps, runtime and communication. Chance over 200 global classes is `0.5%`.

First report a training-readiness check: if Local seen accuracy is `<10%`, label H17-A **UNDERTRAINED/INCONCLUSIVE** rather than treating low PPRTP accuracy as a mechanism failure. Stop after the frozen run; do not increase rounds in the same block.

If Local seen is `>=10%`, call the Tiny mechanism **STRONG** only if all hold:

- paired missing `>=2.0%` (at least 4× global chance);
- paired-minus-native missing `>=1.5 pp`;
- paired-minus-broken missing `>=1.0 pp`;
- paired aggregate predicted-class coverage `>=160/200`;
- mean per-client predicted-class coverage `>=120/200`.

Call it **WEAK/FAILED** if paired missing `<1.0%` or paired-minus-broken missing `<0.5 pp`. Otherwise call it **MIXED**. These gates are fixed before the run and must not be changed afterward.

FedAvg and other baseline accuracy comparisons are descriptive only. If FedAvg dominates missing/all again, preserve the same homogeneous-global-model positioning warning; do not alter the mechanism gate.

If H17-A finishes STRONG, stop after committing seed0 evidence and wait for the next lead decision on seeds1/2. If it is MIXED/WEAK/UNDERTRAINED, stop and report exactly; do not rescue with more anchors, augmentation, ResNet, pretrained weights, extra rounds, learned maps, or result-conditioned changes. If implementation/tests are complete but the real run is still healthy and unfinished at the end of the block, append a concise PARTIAL report and keep H17-A active unchanged.