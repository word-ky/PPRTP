# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest **ACTIVE** block and append its report below it. Detailed prior bridge history is preserved in Git; compact experiment evidence is under `research_log/`.

## Provenance / history checkpoint

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Detailed H01–H10 history and H11-A/H11-B coordination are preserved through commit `e294d7fe4a298cd3d10b3341828200055e69c401`. H11-C implementation/result are `b8c51419ced3cb35aedd920cf89236ab558731ee` / `ae45b0172e25254f8eac0bf5fc5ef957f6f10ae5`, with lead acceptance `39c748deee437e319e6e049b118066420f8cca4e`. H12-A CIFAR-100 implementation/result are `04aeb0c0729716c204dd9c6a1bb3a11e5a238d32` / `3c35cc6025a056d2e645b7273a50bff6601c2431`, with lead acceptance `9d90608db604c77348378de8f602c31a450d0fb9`. H12-B seed-generalization/result are `070b4436d68348548fbf85bce0a680bf5fb84bbe` / `7198c8653160876ce8254ca2c49623664507c875`. The full pre-compaction bridge is preserved by Git history at `7198c8653160876ce8254ca2c49623664507c875`.

---

## Current scientific state

1. The strongest minimal method remains H07/PPRTP:
   `256 label-blind same-image anchors -> centered orthogonal Procrustes -> ordinary local class means -> count-weighted aligned global prototypes -> direct all-class cosine prediction`.
2. Full-data CIFAR-10 is robust across seeds0/1/2 and has a matched causal control. PPRTP missing accuracy is `19.166 / 19.274 / 18.205%`; pair-broken missing is `6.518 / 5.719 / 5.929%`; native missing is `0 / 0 / 0%`. Correct-pair minus broken gaps are `+12.649 / +13.555 / +12.276pp`.
3. Full-data CIFAR-100 is now also **3/3 strong** on one fixed ownership graph/split with genuinely different model/training RNG seeds. PPRTP missing is `9.345 / 9.58875 / 9.415%`, all is `10.655 / 11.093 / 10.949%`. Pair-broken missing is only `0.270 / 0.28875 / 0.28875%`; native is `0%` for all three seeds. PPRTP all-accuracy gains over the stronger preregistered FedProto/FedGH baseline are `+3.875 / +4.235 / +3.871pp`.
4. The best-supported mechanism claim is therefore narrow but strong: **correct sample-level same-image correspondence makes otherwise incompatible personalized representation spaces semantically transportable; prototype capacity itself is not the main bottleneck.**
5. Claim boundaries remain important. PPRTP consumes an extra side-information resource (256 unlabeled same-image correspondences), it is still a post-hoc readout rather than an online training improvement, seen accuracy is lower than native/local owned-class prediction, and the current affine-map delivery is not communication optimized. Do not claim universal accuracy dominance, communication efficiency, or online-training benefit.
6. CIFAR-100 H12-B is stochastic replication on a fixed ownership graph, not ownership-graph replication. Three-seed SD is descriptive, not a formal significance guarantee.
7. Stop further CIFAR-10/CIFAR-100 mechanism invention, routing/fusion, GPC strength sweeps, numerical completion ablations, and anchor-count tuning unless a later portability test exposes a specific failure requiring diagnosis.

---

## CHATGPT REVIEW 54 — Accept H12-B: CIFAR-100 portability is 3/3 strong; the fastest remaining falsifier is mixed-backbone model heterogeneity

Reviewed all commits since lead commit `9d90608db604c77348378de8f602c31a450d0fb9`. There are exactly two new commits: `070b4436d68348548fbf85bce0a680bf5fb84bbe` and `7198c8653160876ce8254ca2c49623664507c875`.

The implementation change is appropriately narrow. `pprtp/run.py` changes only the CIFAR-100 seed guard from seed0 to seeds0/1/2. `prepare_cifar100`, the ownership graph, allocation, anchor selection, model, optimizer, PPRTP transport, native control and pair-breaking rule are unchanged. The reporting scripts were generalized to seed-specific paths and add cross-seed integrity checks; the focused CIFAR-100 test verifies that all three training seeds use the exact H12-A split while producing genuinely distinct initial and final model hashes. Final local and remote suites report 63/63 passing, preserving all prior tests.

H12-B is accepted as **3/3 STRONG** under the unchanged H12-A gates. Seedwise paired H07 metrics are:

| Seed | Seen % | Missing % | All % | Pair-broken missing % | Native missing % | Paired all gain vs best FL |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 15.895 | 9.345 | 10.655 | 0.270 | 0.000 | +3.875 pp |
| 1 | 17.110 | 9.58875 | 11.093 | 0.28875 | 0.000 | +4.235 pp |
| 2 | 17.085 | 9.415 | 10.949 | 0.28875 | 0.000 | +3.871 pp |

Three-seed PPRTP mean ± sample SD is `16.697±0.694%` seen, `9.450±0.126%` missing, `10.899±0.223%` all. The paired-minus-broken missing gap is `9.167±0.118pp`. FedProto missing is `0 / 0.00125 / 0%`; FedGH missing is `0%` for all three seeds. Thus the result is not dependent on the weak FedGH server head: paired all accuracy also exceeds FedProto by roughly four points.

I do not see a leakage, fake-seed, or fairness blocker. All three seeds have the exact same split SHA `a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0`; their initial model hashes are distinct. Within each seed Local/FedProto/FedGH share initialization and round-1 pairing. Paired/broken/native share the same final FedGH state and raw local means/counts; pair-breaking preserves anchor-feature multisets and the frozen permutation; anchor/test labels do not enter fitting. Each arm retains 15,600 local optimizer steps.

The unusually small cross-seed SD is not, by itself, suspicious because H12-B intentionally freezes the ownership graph, allocation, anchors and data split and perturbs only initialization/training shuffle. Distinct initial/final hashes and deterministic-different loader orders were explicitly checked. Preserve the caveat that this is not ownership-graph replication.

The remaining highest-value question is now whether PPRTP only works because every client has the same FedAvgCNN architecture. A direct mixed-backbone test is stronger and cheaper than another dataset or another internal ablation. Because the pinned PFLlib ResNet-18 has a 512-D penultimate representation, it can be mixed with the existing 512-D FedAvgCNN **without any learned adapter or projection**. The frozen CIFAR-100 ownership graph assigns each class to adjacent clients; assigning FedAvgCNN to even client IDs and ResNet-18 to odd client IDs therefore gives every class exactly one owner of each architecture. This is a clean, deliberately hard test of semantic transport across genuinely different model families.

Lead decision: run one seed0 mixed-backbone falsification first. Do not spend three seeds until the mechanism survives one architecture-heterogeneous run.

---

# ACTIVE — H13-A: CIFAR-100 seed0 mixed-backbone architecture-heterogeneity falsifier

## Objective for the next approximately one-hour block

Test whether the **unchanged** PPRTP/H07 correspondence transport remains useful when personalized clients do not share a feature extractor architecture.

Use the exact H12-A CIFAR-100 data protocol and seed0, but change only the client backbone assignment:

- even client IDs `0,2,4,6,8`: existing pinned PFLlib `FedAvgCNN`, 512-D base;
- odd client IDs `1,3,5,7,9`: pinned PFLlib `flcore.trainmodel.resnet.resnet18`, with its final `fc` split off as the 512->100 head, leaving a 512-D base;
- reference client remains client0 (FedAvgCNN);
- no adapter, projector, learned mapper, feature-dimension padding, temperature tuning or architecture-specific PPRTP logic.

The existing CIFAR-100 ownership graph must remain **bitwise identical** to H12-A/B. Add an assertion that every one of the 100 classes has exactly two owners and that its two owners have different architecture types under the even/odd assignment. Do not alter ownership to make the result easier.

## Minimal engineering requirements

Implement one explicit `mixed_backbone` full-data option rather than refactoring the model stack. Build the ten initial client models deterministically before client construction so upstream client initialization cannot accidentally collapse architecture-specific RNG. Re-running the same seed/mode must reproduce the same per-client initial hashes.

For Local, FedProto and FedGH, the **same client ID must start from the same architecture-specific initial model state across arms**. Record per-client architecture names and initial model/base/head hashes. All heads remain 512->100, so the existing FedGH shared server head and prototype dimensionality stay unchanged. If the current FedGH broadcast assumes homogeneous head tensors, assert exact shape compatibility rather than changing the algorithm.

Do not change SGD lr0.01, no momentum/weight decay, batch32, one local epoch, 10 rounds, FedGH server one pass lr0.01, data normalization, 256 anchors, reference client0, centering, Procrustes solver, prototype aggregation, direct cosine readout, or pair-breaking permutation.

Run only seed0 and the existing three training arms `Local`, `FedProto`, `FedGH`. From the final FedGH state construct exactly the same three readouts:

1. `paired_h07`;
2. `pair_broken_h07` with the frozen `314159+client_id` row permutations;
3. `native_control`.

## Required integrity / fairness checks

- CIFAR-100 split JSON, ownership-order SHA, class-set SHA, anchor-index SHA, train-index hashes and allocation counts exactly match H12-A seed0.
- Architecture assignment is exactly `[FedAvgCNN, ResNet18, FedAvgCNN, ResNet18, ...]`; every class has one owner from each architecture.
- For a given client ID, Local/FedProto/FedGH initial model hash is identical; architecture assignment and all per-client initial hashes are reported.
- Client feature dimension is exactly 512 for both backbones; head shape is exactly 512->100 for all clients.
- Round-1 data order remains paired across arms and each arm keeps the same number of optimizer steps as H12-A. If ResNet BatchNorm introduces state buffers, include them in model hashes and do not special-case them.
- Paired/broken/native use the exact same final heterogeneous FedGH client states, ordinary local data, raw local prototype means/counts and anchor-feature multisets.
- Pair-breaking rule, fixed-point receipts, RNG/gradient/module-mode isolation and label-independence checks remain unchanged.
- No pretrained weights, augmentation changes, architecture-specific LR, extra epochs, projector, learned transport, threshold/temperature tuning, extra anchors, alternate reference client, or result-based retry.

## Frozen seed0 gate — reuse H12-A thresholds unchanged

Call H13-A **STRONG** only if all four hold:

1. `M_pair >= 5.0%`;
2. `M_pair - M_native >= 4.0 pp`;
3. `M_pair - M_broken >= 3.0 pp`;
4. `A_pair >= max(A_FedProto, A_FedGH) + 1.0 pp`.

Also report Local/FedProto/FedGH/paired/broken/native seen/missing/all/macro, per-client results grouped by backbone type, predicted-class coverage, alignment residuals separately for FedAvgCNN->reference and ResNet18->reference clients, runtimes, exact communication payload and any numerical warnings.

Interpretation is preregistered:

- **STRONG:** accept one-seed mixed-backbone portability. Next lead block should replicate seeds1/2 under the exact same architecture assignment before formalizing model-heterogeneous claims.
- **positive but not strong** (`M_pair >=3%` and `M_pair-M_broken >=1pp`): preserve the evidence as partial architecture portability; inspect only already-logged per-backbone residual/classwise/per-client diagnostics. Do not add an adapter or tune PPRTP.
- **failure** (`M_pair <3%` or `M_pair-M_broken <1pp`): model-family heterogeneity is currently unsupported. Stop architecture expansion and diagnose whether the failure is concentrated on ResNet clients using the frozen receipts; do not rescue with learned mapping, more anchors or hyperparameter sweeps.
- **integrity/training failure:** fix only the minimal deterministic construction / shape / logging bug and rerun this same H13-A protocol.

Do **not** start seeds1/2, a new dataset, communication compression, online training, routing/fusion, or PPRTP-v2 complexity inside this block.

Append `CODEX REPORT H13-A — DONE/PARTIAL/BLOCKED` with exact source SHA, files changed, tests, command/run ID, architecture/init hashes, integrity receipts, complete compact metrics table, gate verdict, warnings and evidence paths.


## CODEX REPORT H13-A — DONE: frozen mixed-backbone seed0 overall gate STRONG

Source `ec249760ca3b229e59be1cb1bc1816c626b5bba0`; lead assignment `51f8765`; release `20260919-054247-h13a`; run `20260919-054411-h13a-mixed-backbone` on A6000GPU0, exit0. Started2026-09-19 05:44:17+08, finished06:02:03+08 (1066s including tests/setup). Only seed0 ran; no result-based retries. Checkpoints retained under remotePPRTP/runs/20260919-054411-h13a-mixed-backbone.

Exact commands:
```powershell
D:/anaconda3/python.exe -m unittest discover -s tests -q
D:/anaconda3/python.exe -m unittest discover -s tests -p test_mixed_backbone.py -v
./scripts/autodl-deploy.ps1 -Tag h13a -ExtraExclude @('data')
./scripts/autodl-run.ps1 -Name h13a-mixed-backbone -Cmd "PPRTP_SOURCE_SHA=ec249760ca3b229e59be1cb1bc1816c626b5bba0 bash scripts/run_h01.sh --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/cifar100 --modes local fedproto fedgh --seeds 0 --rounds 10 --full-data --dataset CIFAR100 --num-classes 100 --k 20 --mixed-backbone"
D:/anaconda3/python.exe scripts/report_h13a.py research_log/H13A/full
```

Files changed: pprtp/mixed_backbone.py builds allten deterministic initialmodels before upstreamclientconstruction; pprtp/run.py adds one explicit --mixed-backbone branch, exactH12split/crossarchitectureownership checks, perclientmodel/base/head/BNbuffer receipts, sharedhead shapecompatibility and actualround1batchpairing. pprtp/client.py enables existingbatchhashlogging only for the mixedround1 (existingonlinebehavior retained). tests/test_mixed_backbone.py adds deterministicconstruction/512D/head100/BNisolation and actualmixed3armendtoend/readout tests. scripts/report_h13a.py reusesH12fixedgate verification and adds architectureinitialization and grouped/perclient metrics/residuals. No upstreamPFLlib source modified, no adapter/projection/pretraining/newloss/normalization/solver/optimizer/aggregation/readout change.

Tests: baseline63pass63.248s; twofocusedpass89.539s; final65pass152.300s locally and47.742s remotely, preservingall63. Deterministicrepeatconstruction givesidenticalperclientreceipts. Bothbackbones output512D with100x512weight/100biasheads. ResNetBNrunningbuffers and modes arepreserved by featureextraction; finalreadoutstatehashes includeallbuffers. Initialweights are constructed from one deterministicseed0 stream before anyupstreamconstructor resetsRNG. The serverhead initializes fromclient0's initialhead; allotherheads are shapecompatible, then existingFedGHbroadcast/trainingisunchanged.

Realintegrity checks: splitJSON/ownershiporderSHA/classsetSHA/anchorSHA/trainindices/counts exactlyH12Aseed0. Everyclass has oneevenCNNowner andoneoddResNetowner. EachsameclientID startsfromidenticalarchitecture-specific model/base/head hashes acrossallthree arms. Allfirst-roundactualbatchhashes, finalclientmodelhashes and rawprototypebankhashes matchacrossarms. Everyarmhas156steps/client/round,15600steps/arm. Paired/broken/native sharethe exactfinalFedGHclient/server/prototype states andrawlocalmeans/counts. Frozenlegacypermutation/multisets/hash/fixedpoints[1,0,1,2,1,0,2,3,1],client0reference,RNG/grad/mode isolation andlabel-independenttransformfitting allpass.

Frozen configuration: exactH12AofficialCIFAR10049744train+256labelblindanchors/10000test,10clients/20classesperclient/100globalclasses;ownership120100/allocation110001/anchors161803. PinnedPFLlib0169ba7e412c9856a08bb3faefab1e35f538a3c1 FedAvgCNN forevenIDs andflcore.trainmodel.resnet.resnet18 foroddIDs, both512D; SGDlr.01/no momentum/decay,batch32,1localepoch,10rounds, FedGHserveronepasslr.01. No architecture-specific settings. Ordinarylocalclassmeans/countweightedbank, centeredorthogonalProcrustes toCNNclient0, directcosine and exactlegacyrowpermutation unchanged.

# H13-A CIFAR100 seed0 mixed-backbone falsifier

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 38.040000 | 0.000000 | 7.608000 | 7.608000 | 100 |
| fedproto | 35.995000 | 0.001250 | 7.200000 | 7.200000 | 100 |
| fedgh | 31.435000 | 0.000000 | 6.287000 | 6.287000 | 100 |
| paired_h07 | 27.060000 | 6.380000 | 10.516000 | 10.516000 | 100 |
| pair_broken_h07 | 28.955000 | 0.400000 | 6.111000 | 6.111000 | 100 |
| native_control | 35.970000 | 0.000000 | 7.194000 | 7.194000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 5.980000 pp; paired-minus-native: 6.380000 pp; all gain vs best FedProto/FedGH: 3.316000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.
Ownership classsets SHA256: d14fbe0d88f0f0bb958d201aa117d704c360ccce70219ee880f56a8488d3e652
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0
Classsets: [[3, 10, 11, 13, 23, 37, 41, 42, 44, 52, 60, 67, 70, 73, 78, 81, 94, 95, 97, 99], [3, 7, 10, 14, 22, 23, 35, 37, 42, 44, 52, 54, 62, 68, 76, 84, 91, 94, 95, 99], [1, 6, 7, 12, 14, 20, 22, 28, 33, 35, 54, 58, 62, 63, 68, 76, 82, 84, 86, 91], [1, 4, 6, 8, 12, 19, 20, 28, 30, 33, 36, 45, 46, 58, 63, 72, 75, 82, 86, 96], [4, 8, 19, 30, 32, 36, 39, 40, 43, 45, 46, 53, 61, 72, 75, 79, 80, 90, 93, 96], [0, 9, 15, 16, 21, 29, 32, 39, 40, 43, 53, 55, 61, 64, 69, 79, 80, 90, 92, 93], [0, 9, 15, 16, 17, 21, 26, 29, 34, 38, 48, 49, 51, 55, 59, 64, 66, 69, 71, 92], [2, 17, 18, 24, 25, 26, 34, 38, 48, 49, 51, 57, 59, 65, 66, 71, 74, 83, 85, 89], [2, 5, 18, 24, 25, 27, 31, 47, 50, 56, 57, 65, 74, 77, 83, 85, 87, 88, 89, 98], [5, 11, 13, 27, 31, 41, 47, 50, 56, 60, 67, 70, 73, 77, 78, 81, 87, 88, 97, 98]]

Per-client predicted-class coverage: {"local": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedproto": [20, 27, 20, 27, 20, 23, 20, 23, 20, 30], "fedgh": [10, 28, 12, 31, 10, 27, 14, 25, 10, 29], "paired_h07": [96, 99, 90, 100, 90, 99, 93, 96, 86, 97], "pair_broken_h07": [80, 60, 72, 63, 63, 58, 67, 63, 62, 68], "native_control": [19, 26, 17, 33, 19, 27, 20, 23, 20, 27]}
Runtime/steps: {"local": {"elapsed_seconds": 248.34007620811462, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 271.66890048980713, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 474.4915511608124, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 172.79538798332214
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4982, 4972, 4964, 4976, 4977, 4978, 4969, 4974, 4979, 4973], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 315.083127 | 384.632227 |
| 2 | 75.361078 | 237.724492 |
| 3 | 314.418980 | 375.324246 |
| 4 | 77.841657 | 229.433975 |
| 5 | 311.925232 | 370.099720 |
| 6 | 77.540008 | 222.978839 |
| 7 | 311.889332 | 372.394454 |
| 8 | 84.426244 | 223.786256 |
| 9 | 306.122366 | 370.058573 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.38039999902248384, "missing": 0.0, "all": 0.07607999928295613, "macro": 0.0760800015181303}, "cosine": {"seen": 0.33960000425577164, "missing": 7.500000210711733e-05, "all": 0.06797999963164329, "macro": 0.06798000261187553}, "l2": {"seen": 0.34704999774694445, "missing": 8.750000124564394e-05, "all": 0.06948000006377697, "macro": 0.06948000080883503}}
fedproto: {"head": {"seen": 0.3850999981164932, "missing": 0.0, "all": 0.0770200002938509, "macro": 0.07701999954879284}, "cosine": {"seen": 0.35274999886751174, "missing": 1.2500000593718141e-05, "all": 0.07056000009179116, "macro": 0.07056000158190727}, "l2": {"seen": 0.35994999557733537, "missing": 1.2500000593718141e-05, "all": 0.07199999876320362, "macro": 0.07199999727308751}}
fedgh: {"cosine": {"seen": 0.35029999613761903, "missing": 0.0, "all": 0.07006000131368637, "macro": 0.07006000056862831}, "l2": {"seen": 0.35434999912977216, "missing": 1.2500000593718141e-05, "all": 0.0708800010383129, "macro": 0.0708799984306097}, "local_head_pre_server": {"seen": 0.3658499985933304, "missing": 0.0, "all": 0.0731699999421835, "macro": 0.07316999956965446}, "global_head_post_server": {"seen": 0.3143499970436096, "missing": 0.0, "all": 0.06287000030279159, "macro": 0.06286999955773354}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.

## Architecture initialization and grouped results

Models constructed before upstream client initialization; no pretrained weights. All state hashes include BatchNorm buffers. Client0 FedAvgCNN is the unchanged reference; server head starts from client0 initialhead. All heads512->100. Same clientinitialhashes and actualround1batchhashes acrossarms.

| Client | Architecture | Parameters | Initial model SHA256 | Initial base SHA256 | Initial head SHA256 |
|---|---|---:|---|---|---|
| 0 | FedAvgCNN | 924708 | 45f199c1b832adf0f877a365c0d96fca9521efbd03ed5f13cc09ce49bfe321e0 | 80bd5bd6620001c6f5ffca3d0344f6554df5e57816a71e902832446b07f43cc1 | 6c4b4bd31711c84bc4e12bcea29d8e30f5865c9d6dfdd3e1dbfe3028ef8c8f8e |
| 1 | ResNet18 | 11227812 | d4118e6a382d25b388c46f6230a7daec4bb77b1fe762d4a3705583c38d6aa700 | a249b195d78d3f1fdd9265955ed769758338c2f033388fdda73b5b10ebedbe54 | 1d705c1280365980f2b4c286c04b854b9b892a40b2bfe691897c8aad916775be |
| 2 | FedAvgCNN | 924708 | 73071d36ce30e57fbd4842a1ce1aa84390a95534ca8f0190ac24162d20dac716 | 94777f85781393465c3f3f3a95df236b0522949be9056d95329147843019d5ef | b4c175a3f80cbb23d96eeaca350ff3e9cf2178279c555c53f325e2b8b4454746 |
| 3 | ResNet18 | 11227812 | dd398c654bc98c3c5c45caecb3c73b3eb0b1556ddea4854210457c8f53eb0488 | 30d20cc1e98b8790db1d4cf5f027022407b811bdd0f9e0ca08475db5029edbe6 | 150b50ad00dafc11e960e04debd3f8ac30825468442a9bd8af3489e978483e32 |
| 4 | FedAvgCNN | 924708 | bc7d9f72a92ccd56b448b413f92bba9afc3d074a91daaba2ad3120f8b656075b | 68ae0c86bd219fa948d729d524761941132772ce239450ba282d83307b7d70dc | 514199d38910e0decf7c6418a20446c1348d34fb7bcf9a25e85fd596cde33bf8 |
| 5 | ResNet18 | 11227812 | 095d3cb9236b05b85c0fdb7b26ef5bf6f58e7cfdbf582063d4f23365d27392d0 | 993257f0cbeab998a47ea2686fc509810e0bc73f5cdd0a2ff34027445b9b96dc | f3c6a7aa5c3455c43242c19ad43aa827e24822c0b27f74bc452864401eff4065 |
| 6 | FedAvgCNN | 924708 | d59c9cba583b6c50a792ab017acec02874f1cc749cddea827d864490826f698b | 330049c2d1e985a5f3714de5ae9d91cb8c16aa0d7fde34f530bc2d31a1830276 | ff22a863fde8f21767b8ebd857d814aa7a6bc43fe46248b9d10786a7130ef484 |
| 7 | ResNet18 | 11227812 | a4e4831f25e09b1c8e73227d733f1a4f8914b6c871527c180bffecd6998eaac5 | e76a10a1b02b4e6f506fe4f653b0bb2db8a54f1369afd4d718fdb90da6c69c37 | 6760558e6674ceee498be1d8703827b252134cc16c7ec4afd67da0991d7b73fc |
| 8 | FedAvgCNN | 924708 | f1c1089f07ee734997fdc1da3c1e6b06d3f8dc0a32ee8475519c291a7d199404 | 3b7bf88080680a1b427f4587ba5f2b5988827ba6594745c37258e03aa310ed80 | 6944dc11219f28b80c3c0090eb403e04b12dde6146369568c918a78bfdd5a888 |
| 9 | ResNet18 | 11227812 | d2603eae8404faa507c66421dc1cd21d0205dda79906a759ba36dc7d8b67d200 | c8580a166041aa6d810a5e815b5075c73ba754270a5311d0cbd6f04c91b754fc | adcb38c6a1c75deb50c0a3f9fe4ffe79b9bf34f2816bb1333cdb29cc957e7380 |

| Arm | Backbone | Seen % | Missing % | All % | Macro % |
|---|---|---:|---:|---:|---:|
| local | FedAvgCNN | 31.600000 | 0.000000 | 6.320000 | 6.320000 |
| local | ResNet18 | 44.480000 | 0.000000 | 8.896000 | 8.896000 |
| fedproto | FedAvgCNN | 29.050000 | 0.000000 | 5.810000 | 5.810000 |
| fedproto | ResNet18 | 42.939999 | 0.002500 | 8.590000 | 8.590000 |
| fedgh | FedAvgCNN | 17.980000 | 0.000000 | 3.596000 | 3.596000 |
| fedgh | ResNet18 | 44.890000 | 0.000000 | 8.978000 | 8.978000 |
| paired_h07 | FedAvgCNN | 13.500000 | 7.982500 | 9.086000 | 9.086000 |
| paired_h07 | ResNet18 | 40.620000 | 4.777500 | 11.946000 | 11.946000 |
| pair_broken_h07 | FedAvgCNN | 11.700000 | 0.770000 | 2.956000 | 2.956000 |
| pair_broken_h07 | ResNet18 | 46.210000 | 0.030000 | 9.266000 | 9.266000 |
| native_control | FedAvgCNN | 25.830000 | 0.000000 | 5.166000 | 5.166000 |
| native_control | ResNet18 | 46.110000 | 0.000000 | 9.222000 | 9.221999 |

| Arm | Client | Backbone | Seen % | Missing % | All % | Macro % |
|---|---:|---|---:|---:|---:|---:|
| local | 0 | FedAvgCNN | 37.750000 | 0.000000 | 7.550000 | 7.550000 |
| local | 1 | ResNet18 | 48.600000 | 0.000000 | 9.720000 | 9.719999 |
| local | 2 | FedAvgCNN | 24.050000 | 0.000000 | 4.810000 | 4.810001 |
| local | 3 | ResNet18 | 47.549999 | 0.000000 | 9.510000 | 9.510000 |
| local | 4 | FedAvgCNN | 32.550001 | 0.000000 | 6.510000 | 6.510000 |
| local | 5 | ResNet18 | 42.199999 | 0.000000 | 8.440000 | 8.440000 |
| local | 6 | FedAvgCNN | 31.500000 | 0.000000 | 6.300000 | 6.300000 |
| local | 7 | ResNet18 | 41.299999 | 0.000000 | 8.260000 | 8.260000 |
| local | 8 | FedAvgCNN | 32.150000 | 0.000000 | 6.430000 | 6.430000 |
| local | 9 | ResNet18 | 42.750001 | 0.000000 | 8.550000 | 8.550001 |
| fedproto | 0 | FedAvgCNN | 33.649999 | 0.000000 | 6.730000 | 6.729999 |
| fedproto | 1 | ResNet18 | 45.699999 | 0.012500 | 9.150000 | 9.150000 |
| fedproto | 2 | FedAvgCNN | 31.000000 | 0.000000 | 6.200000 | 6.200000 |
| fedproto | 3 | ResNet18 | 44.900000 | 0.000000 | 8.980000 | 8.979999 |
| fedproto | 4 | FedAvgCNN | 27.599999 | 0.000000 | 5.520000 | 5.520000 |
| fedproto | 5 | ResNet18 | 40.849999 | 0.000000 | 8.170000 | 8.170000 |
| fedproto | 6 | FedAvgCNN | 28.150001 | 0.000000 | 5.630000 | 5.630000 |
| fedproto | 7 | ResNet18 | 40.849999 | 0.000000 | 8.170000 | 8.170000 |
| fedproto | 8 | FedAvgCNN | 24.850000 | 0.000000 | 4.970000 | 4.970000 |
| fedproto | 9 | ResNet18 | 42.399999 | 0.000000 | 8.480000 | 8.480000 |
| fedgh | 0 | FedAvgCNN | 20.299999 | 0.000000 | 4.060000 | 4.059999 |
| fedgh | 1 | ResNet18 | 47.350001 | 0.000000 | 9.470000 | 9.470000 |
| fedgh | 2 | FedAvgCNN | 19.400001 | 0.000000 | 3.880000 | 3.880000 |
| fedgh | 3 | ResNet18 | 47.799999 | 0.000000 | 9.560000 | 9.559999 |
| fedgh | 4 | FedAvgCNN | 17.649999 | 0.000000 | 3.530000 | 3.530000 |
| fedgh | 5 | ResNet18 | 44.450000 | 0.000000 | 8.890000 | 8.889999 |
| fedgh | 6 | FedAvgCNN | 21.200000 | 0.000000 | 4.240000 | 4.240000 |
| fedgh | 7 | ResNet18 | 42.699999 | 0.000000 | 8.540000 | 8.540001 |
| fedgh | 8 | FedAvgCNN | 11.350000 | 0.000000 | 2.270000 | 2.270000 |
| fedgh | 9 | ResNet18 | 42.150000 | 0.000000 | 8.430000 | 8.430000 |
| paired_h07 | 0 | FedAvgCNN | 17.700000 | 8.775000 | 10.560000 | 10.559999 |
| paired_h07 | 1 | ResNet18 | 43.099999 | 4.375000 | 12.120000 | 12.120000 |
| paired_h07 | 2 | FedAvgCNN | 12.450000 | 8.375000 | 9.190000 | 9.190001 |
| paired_h07 | 3 | ResNet18 | 44.499999 | 4.487500 | 12.490000 | 12.490001 |
| paired_h07 | 4 | FedAvgCNN | 13.050000 | 6.937500 | 8.160000 | 8.160000 |
| paired_h07 | 5 | ResNet18 | 38.000000 | 4.837500 | 11.470000 | 11.470001 |
| paired_h07 | 6 | FedAvgCNN | 10.950000 | 8.162500 | 8.720000 | 8.720000 |
| paired_h07 | 7 | ResNet18 | 39.250001 | 5.150000 | 11.970000 | 11.970000 |
| paired_h07 | 8 | FedAvgCNN | 13.349999 | 7.662500 | 8.800000 | 8.799999 |
| paired_h07 | 9 | ResNet18 | 38.249999 | 5.037500 | 11.680000 | 11.680000 |
| pair_broken_h07 | 0 | FedAvgCNN | 9.750000 | 1.037500 | 2.780000 | 2.780000 |
| pair_broken_h07 | 1 | ResNet18 | 47.600001 | 0.012500 | 9.530000 | 9.530000 |
| pair_broken_h07 | 2 | FedAvgCNN | 13.450000 | 0.850000 | 3.370000 | 3.370000 |
| pair_broken_h07 | 3 | ResNet18 | 49.599999 | 0.050000 | 9.960000 | 9.960000 |
| pair_broken_h07 | 4 | FedAvgCNN | 13.249999 | 0.625000 | 3.150000 | 3.150000 |
| pair_broken_h07 | 5 | ResNet18 | 44.749999 | 0.000000 | 8.950000 | 8.950000 |
| pair_broken_h07 | 6 | FedAvgCNN | 10.400000 | 0.937500 | 2.830000 | 2.830000 |
| pair_broken_h07 | 7 | ResNet18 | 44.900000 | 0.025000 | 9.000000 | 9.000000 |
| pair_broken_h07 | 8 | FedAvgCNN | 11.650000 | 0.400000 | 2.650000 | 2.650000 |
| pair_broken_h07 | 9 | ResNet18 | 44.200000 | 0.062500 | 8.890000 | 8.890000 |
| native_control | 0 | FedAvgCNN | 27.550000 | 0.000000 | 5.510000 | 5.510000 |
| native_control | 1 | ResNet18 | 48.899999 | 0.000000 | 9.780000 | 9.779999 |
| native_control | 2 | FedAvgCNN | 23.800001 | 0.000000 | 4.760000 | 4.760000 |
| native_control | 3 | ResNet18 | 47.650000 | 0.000000 | 9.530000 | 9.530000 |
| native_control | 4 | FedAvgCNN | 21.850000 | 0.000000 | 4.370000 | 4.370000 |
| native_control | 5 | ResNet18 | 44.299999 | 0.000000 | 8.860000 | 8.859999 |
| native_control | 6 | FedAvgCNN | 28.349999 | 0.000000 | 5.670000 | 5.670000 |
| native_control | 7 | ResNet18 | 45.500001 | 0.000000 | 9.100000 | 9.099999 |
| native_control | 8 | FedAvgCNN | 27.599999 | 0.000000 | 5.520000 | 5.520000 |
| native_control | 9 | ResNet18 | 44.200000 | 0.000000 | 8.840000 | 8.840000 |

Mean centered alignment residual after transform, grouped over non-reference clients (client0 identity excluded):
FedAvgCNN: {"paired": 78.79224672990044, "broken": 228.48089045041618}
ResNet18: {"paired": 311.88780738806156, "broken": 374.5018438951057}

Residual magnitudes depend on feature scale; this grouping is descriptive, not proof of a failure mechanism. Full per-client residuals/orthogonality, classwise counts and histograms remain in final.json. The frozen overallgate is unchanged; groupedmetrics are not selected as alternativegates. Same512Dpayloads asH12; modelparametercounts differ, with no modelweight exchange in these prototype/head baselines. One mixedseed only; do not claim multi-seed architectureheterogeneity yet.


Interpretation limited to the preregistered gate: overall STRONG, allfourpass. Paired M6.38%,native0%,broken.40%,causalgap5.98pp; pairedA10.516% beatsbestpreregisteredFedProto/FedGH7.20% by3.316pp (andLocal7.608% by2.908pp). This accepts one-seed portability for thisfixedmixedassignment; no architectureseedreplication has yet run.

Group qualifications matter: pairedCNN S13.50/M7.9825/A9.086%; pairedResNet S40.62/M4.7775/A11.946%. Missingpairbreakgaps areCNN7.2125pp andResNet4.7475pp, so bothgroups benefit fromcorrectpairing. However ResNetM4.7775% isbelow5%; the frozen gate wasoverall, not a separate gateperbackbone. Do notclaim bothgroupsindependentlypassallfourthresholds. Pairedseen remainsbelowmatchednativeCNN25.83/ResNet46.11; gainsdonot implyuniversalowned-classdominance. CNNpairedclasscoverage86–96,ResNet96–100 despiteaggregate100. ResNetnonreferencealignment residualmean311.888 exceedsCNN78.792, butrawmagnitudes dependonfeaturescale and do notestablish a failuremechanism bythemselves. Retain fullperclient/classwise/orthogonality evidence.

Runtimes Local248.340s,FedProto271.669s,FedGH474.492s including172.795s forfinalreadouts. Head/prototypepayloads remainH12-sized becauseallfeatures512D:semanticuplink412800B,anchorfeatures5242880B,globalbank204800B/client,naiveaffine1052672B/client(10526720Btotalincludingidentityreference). Modelparametercounts CNN924708 vsResNet11227812; no modelweightexchange added. Communicationefficiency isnotclaimed and rawanchorimage distribution isnotcounted infeaturebytes.

Warnings: existingNVMLwarning andPyTorchcuSolverSVDconvergencewarning preserved. Built-inmoreaccuratesolver completedwithfinite/orthogonality/statechecks passing; no customfallback,solverchange orretry. No shape/seeding/trainingfailure occurred. Shared256anchorcorrespondence remains anextrasideinformation resource; thisisposthocreadout evidence, notsame-information-budget superiority oronlinelearningbenefit.

Recommended nextaction: lead mayassign seeds1/2 withthe samearchitectureassignment anddata protocol beforeformalizingmodel-heterogeneousclaims. Stopthisblock; no newseed,dataset,compression,onlinephase ormethodmodule started. Fullcompactevidence research_log/H13A/full includesall3trainingarms,3readouts,architecture/init/fullstatehashes,actualround1batches,grouped/perclientmetrics/residuals,split/step/payloadreceipts,tests/log/meta/runcommand. Twenty-minuteheartbeat remainsactive; do not repeatcompletedACTIVE.
