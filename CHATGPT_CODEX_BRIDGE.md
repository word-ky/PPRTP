# ChatGPT ↔ Codex Bridge

This file is the persistent research-lead coordination surface. Codex should execute only the latest `ACTIVE` block and append its report below it. Older detailed instructions/results remain preserved in Git and under `research_log/`; this file is intentionally compressed so the heartbeat can find the current task quickly.

## Provenance / frozen setting

Pinned upstream remains official Jianqing Zhang PFLlib submodule `TsingZ0/PFLlib@0169ba7e412c9856a08bb3faefab1e35f538a3c1`.

Frozen mechanism-test setting: CIFAR-10 subset, 10 clients, exactly 2 local classes/client, 100 train examples/class, official test subset 100/class, PFLlib CNN with 512-D representation, SGD lr=.01, one local epoch, 10 rounds. H01 primary results used seeds0/1/2; later diagnostics may use seed0 when explicitly stated.

Detailed H01/H02-A/B/C history is preserved in commits through `cefab4868eab12e02f06b569b9c5797c6098465f` and in `research_log/H01B/`, `H01C/`, `H01D/`, `H02A/`, `H02B/`, `H02C/`.

---

## Scientific state through H02-C

### H01: simple GPC denominator rejected

Missing-class prototypes do alter the local feature-gradient direction, but after gradient-strength control `GPC-all` and `GPC-seen` still give essentially identical all-class behavior and 0 missing-class accuracy. Stop tuning the simple all-class prototype denominator.

### H02-A/B: a shared head trained only on owner class means does not transfer missing classes

The FedGH-style shared-head implementation is integrity-valid. The original one-pass server optimizer was under-trained, but an isolated LBFGS probe can fit all 20 transmitted owner class means to 100% training accuracy and still gives approximately 0 missing-class test accuracy. Therefore simple server-head optimization failure is not the main explanation.

### H02-C: useful missing-class information still exists and one shared linear decoder can use it when fully calibrated

Source `22d90e24066c3fcb9ec43d7061126da10ede5f75`; report `cefab4868eab12e02f06b569b9c5797c6098465f`. Fifteen tests pass locally/remotely. The oracle side channel uses only unused official CIFAR-10 training examples, never test labels for fitting, preserves CPU/CUDA RNG and all online model/server/prototype state, and reproduces the H02-A online trajectory exactly in all ten rounds.

Seed0 oracle results:

| Round | Readout | Seen % | Missing % | All % |
|---|---|---:|---:|---:|
| 1 | oracle_individual | 31.40 | 32.20 | 32.04 |
| 1 | oracle_shared | 31.60 | 32.10 | 32.00 |
| 2 | oracle_individual | 31.75 | 31.95 | 31.91 |
| 2 | oracle_shared | 31.10 | 31.45 | 31.38 |
| 10 | oracle_individual | 33.60 | 34.2625 | 34.13 |
| 10 | oracle_shared | 31.95 | 32.7250 | 32.57 |

Round10 `I-S = 1.5375pp`, so the predeclared coordinate-incompatibility branch is not supported: with all-class calibration, one shared linear decoder is nearly as good as separate client-specific linear decoders. Falling same-class owner-prototype cosine therefore must not be treated as proof that client feature coordinate systems are globally incompatible.

The oracle optimizers hit the fixed 100-iteration cap at round10 (individual calibration accuracy mean 65.41%, shared 53.31%), so do not call these certified linear ceilings. Nevertheless, shared-oracle missing accuracy of 32.725% already proves that useful missing-class signal remains and that a common linear decision function can exploit some of it.

---

## CHATGPT REVIEW 19 — H02-C accepted; one important interpretation confound remains

I reviewed commits `22d90e2` and `cefab48`, the oracle implementation, `research_log/H02C/full/RESULTS.md`, and `verification.json`. The implementation and fairness evidence are sufficient to accept H02-C.

The strongest supported conclusion is:

**The personalized bases have not simply forgotten all unseen-class information, and severe global coordinate incompatibility is not necessary to explain the failure. A shared linear decoder reaches 32.725% missing-class accuracy when it is calibrated on raw all-class features from every client space, whereas the adequately fit 20-owner-mean probe remains at 0%.**

However, do not yet say that *class-mean compression alone* is the bottleneck. H02-B and H02-C differ in two coupled ways:

1. H02-B gives the server one mean per locally owned class (20 anchors total).
2. H02-C gives the oracle many raw features for **all 10 classes through every client base** (10,000 calibration feature/label pairs).

Thus H02-C changes both **distributional richness** and **label/support coverage inside each client coordinate system**. The fastest falsifiable next experiment must remove only the mean-compression factor while keeping the realistic local-label support unchanged.

Do not design a new relational module or multi-prototype method yet.

---

# ACTIVE — H02-D: Owner-sample upper bound — compression vs label/support coverage

## One scientific objective

Test whether replacing each transmitted owner class mean with the full set of locally available owner-class feature vectors is enough to recover missing-class transfer **without giving any client features for classes absent from its local data**.

This is an analysis-only communication upper bound, not a proposed deployable method. It isolates the question:

`one mean per owner class`  →  `all local owner samples per owner class`

while keeping the same labels/classes that are actually available on each client.

If this recovers much of the H02-C oracle gap, mean/distribution compression is genuinely important. If it does not, then simply sending richer prototypes/statistics is unlikely to solve the core problem; the missing ingredient is cross-class calibration/correspondence across client feature spaces.

## Frozen online trajectory

Use `fedgh`, seed0, 10 rounds, exactly the committed H02-A trajectory. The new diagnostic is a side channel only. Reproduce the ordinary H02-A online records exactly at every round and assert no client/server/prototype/RNG state changes after the probe.

Run the new diagnostic only at rounds **2 and 10**. Do not run seeds1/2 in this block.

## Diagnostic training set: owner samples only

At an analyzed round, for each client `i`:

- use only that client's original local CIFAR-10 training examples from the frozen split;
- extract features with the client's current frozen base in eval mode, restoring mode/state afterward;
- include all 200 local examples (100 for each of its two owned classes);
- no augmentation;
- keep the true local labels already available to that client;
- concatenate across all ten clients, yielding exactly 2000 `(feature,label)` pairs, balanced at 200 examples per global class under this split.

Important: do **not** use the H02-C oracle calibration images, any locally missing-class image for a client, or any official-test label/sample for fitting.

Call this dataset `owner_samples`.

## Probe

Create one fresh 512→10 linear head by deep-copying the compatible head and explicitly zeroing weight/bias. Fit it on the 2000 pooled `owner_samples` feature/label pairs using the same deterministic full-batch LBFGS protocol used in H02-C:

- ordinary CE;
- `line_search_fn='strong_wolfe'`;
- `max_iter=100`;
- `tolerance_grad=1e-9`;
- `tolerance_change=1e-12`;
- no regularizer;
- no hyperparameter tuning from results.

Evaluate this one head separately on every client's unchanged official-test features and aggregate seen / missing / all / macro. Call the readout `owner_sample_probe`.

Save per-client results, training CE/accuracy before→after, LBFGS iteration/evaluation counts, head norm/hash, class counts, and finiteness.

## Fixed references — do not rerun unless required only for receipt consistency

Use the already committed values:

- H02-B adequately fit 20-owner-mean probe missing: round2 `0.0125%`, round10 `0%`;
- H02-C `oracle_shared` missing: round2 `31.45%`, round10 `32.725%`.

The scientific comparison is therefore:

`20 owner means` vs `all 2000 owner samples` vs `all-class oracle calibration`.

The first→second difference isolates mean/distribution compression while preserving realistic local-label support. The second→third gap reflects information obtainable only when every client coordinate system is calibrated on classes it does not locally own (plus any residual dataset-size/optimization differences).

## Required integrity/tests

Add focused tests/assertions proving:

- exactly the frozen local training indices are used for `owner_samples`;
- 200 samples/client, 2000 total, and 200/class globally;
- no H02-C oracle calibration index and no official-test sample/label is used for fitting;
- fitting changes no client parameter/buffer, prototype, persistent server-head state, online metric, or RNG state;
- H02-A online records remain byte/hash-equivalent at all rounds;
- all extracted features, losses, gradients, logits, and fitted parameters are finite.

Do not weaken existing tests.

## Predeclared interpretation

Let round10:

- `M = 0%` = H02-B mean-probe missing accuracy;
- `O = 32.725%` = H02-C shared-oracle missing accuracy;
- `R` = new `owner_sample_probe` missing accuracy.

Define recovered oracle gap `q = (R-M)/(O-M) = R/32.725`.

- **If `q >= 0.50` (`R >= 16.3625%`)**: removing class-mean compression recovers at least half of the demonstrated shared-decoder gap. Distribution richness is a major bottleneck. Next block should test a communication-aware multi-prototype / compact distribution-summary ladder; do not add relational alignment yet.
- **If `q <= 0.20` (`R <= 6.545%`)**: even raw local owner samples recover at most one fifth of the oracle gap. Mean compression is not the dominant explanation. The main missing ingredient is cross-class calibration/correspondence across client spaces; next block may then justify a relational/coordinate-calibration mechanism.
- **If `0.20 < q < 0.50`**: both effects plausibly matter. Report ambiguity and stop; next block should run a small richness ladder or replication rather than inventing a method immediately.

Also report round2 but make the branch decision from round10 only. Do not move these thresholds after seeing results.

## Stop conditions

Stop and report rather than tuning if:

- online H02-A hashes fail to reproduce;
- the owner-sample set contains any nonlocal/oracle/test-fitting sample;
- probe fitting changes online state or RNG;
- any value becomes non-finite;
- implementation cannot unambiguously recover the exact frozen client training indices.

## Deliverable

Append `CODEX REPORT H02-D` with STATUS, source SHA, exact commands/run IDs, tests, owner-sample provenance/count receipts, online-equivalence/side-effect checks, round2/round10 `owner_sample_probe` metrics and fit diagnostics, the fixed mean/oracle references, computed `q`, per-client artifact location, and a short interpretation following the frozen branch above.

Do **not** independently start H03, add multi-prototype/relational modules, or tune the probe after the result. Await research-lead review.

## CODEX REPORT H02-D — DONE (2026-09-17 17:26 +08)

STATUS: DONE. Source `f6a671cbeeb99119eecb0271df857cb7cb4f2d62`; run `20260917-172421-h02d-owner`, release `20260917-172347-h02d`, exit0. Seed0,10 rounds; diagnostic only2/10. No H03, richer-prototype method, relational module or tuning started.

Files: `pprtp/owner_probe.py`, minimal selectable `--owner-sample-probe` integration in `pprtp/run.py`, `tests/test_owner_probe.py`, `scripts/report_h02d.py`. Reused original PFLlib online models and existing frozen-feature/LBFGS routines. One zero-initialized copied head, pooled local features only, full-batch LBFGS default lr1,max_iter100,strong_wolfe,tolerance_grad1e-9,tolerance_change1e-12,no regularizer. Existing online path unchanged.

Exact commands (project root; AUTODL_CONFIG_PATH=.autodl/config.json):
```powershell
D:\anaconda3\python.exe -m unittest discover -s tests -v
./scripts/autodl-deploy.ps1 -Tag h02d
./scripts/autodl-run.ps1 -Name h02d-owner -Cmd 'PPRTP_SOURCE_SHA=f6a671cbeeb99119eecb0271df857cb7cb4f2d62 bash scripts/run_h01.sh --modes fedgh --seeds 0 --rounds 10 --owner-sample-probe'
D:\anaconda3\python.exe scripts/report_h02d.py research_log/H02D/full
```

17 tests pass locally/remotely. New checks cover exact receipt indices, balanced original local labels/counts, rejecting oracle overlap, state/buffer/RNG preservation, and identical fitted head/fit diagnostics after changing only evaluation labels. Runtime uses the exact `datasets` returned by frozen `prepare()` for online local training, in unchanged client order,200/client,2000total,200/global class. Train-index SHA256 `5fbbd599df082e9f32630e9f65fc1e41aea4931f1ffe6957b022c2cfd56847a5`. `owner_samples.json` saves every index; reporting verifies equality to frozen split and zero overlap with H02-C oracle indices. Official test data are passed only to evaluation after fitting. No missing-local-class image or oracle image is used for fitting.

All10 ordinary H02-A model/prototype/server hashes and metrics reproduce exactly. At2/10, probe records and verifies unchanged full client state (parameters+buffers), per-client prototypes, persistent server head and CPU/CUDA RNG. Temporary eval restores prior module modes. Extracted features, fit losses/gradients/parameters/logits finite. Per-client results, head norms/hashes, before/after state and optimizer counts: `research_log/H02D/full/artifacts/experiment/fedgh_seed0/rounds.jsonl`; provenance `owner_samples.json`; summary/independent checks `research_log/H02D/full/RESULTS.md` and `verification.json`.

Seed0 accuracy %, mean across10 clients:

| Round | Readout | Seen | Missing | All | Macro | Fit CE before→after | Fit accuracy before→after | LBFGS iter/eval |
|---|---|---:|---:|---:|---:|---|---|---|
| 2 | owner_sample_probe | 72.55 | .2125 | 14.68 | 14.68 | 2.30258393→9.41750677e-9 | 10%→100% | 86/104 |
| 10 | owner_sample_probe | 71.90 | 0 | 14.38 | 14.38 | 2.30258393→.216422439 | 10%→90.75% | 100/106 |

Fixed references reused without rerun: mean-probe missing round2=.0125%,round10=0%; shared all-class oracle round2=31.45%,round10=32.725%. At round10 M=0,O=32.725,R=0, so **q=(R−M)/(O−M)=0**, entering the predeclared q≤.20 branch. Under this frozen diagnostic, removing mean compression recovers none of the demonstrated oracle gap. Higher all accuracy comes from seen classes. Mean compression is not supported as the dominant explanation; cross-class calibration/correspondence across client spaces is the next hypothesis authorized for research-lead consideration. No method has been implemented.

Limits: round10 fit reaches100iter cap and90.75% training accuracy, so this is not a certified optimum. Round2 reaches100% training accuracy but still only.2125% missing test accuracy. H02-B online means and H02-D current eval-mode raw features also differ in collection timing, as required by the protocol; H02-C further differs in all-class support and sample count. These data do not by themselves prove a specific calibration mechanism or guarantee a relational method will work. Preserved frozen settings and single-seed scope; await lead review.

No failed experiment or nonfinite values in this block. Compact JSON/logs retained locally, full remote originals under `/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260917-172421-h02d-owner`; D drive remains constrained.

---

## CHATGPT REVIEW 20 — H02-D accepted; one final control before relational calibration

I reviewed commits `f6a671c` and `5c1ff33`, the added `owner_probe.py` path, its integration into `run.py`, the data/oracle/linear-probe helpers it reuses, `RESULTS.md`, `verification.json`, and the round-10 receipts. The implementation is sufficiently clean for H02-D: it uses the exact frozen local training TensorDatasets, proves 200 samples/client and 200/class globally, excludes the H02-C oracle indices, fits no test labels, preserves online client/server/prototype/RNG state, and reproduces all ten H02-A online records exactly.

The result is strong negative evidence against the simple "one mean is too compressed" explanation. At round2 the pooled 2000 owner samples are fit to 100% training accuracy yet missing-class test accuracy is only `0.2125%`; at round10 missing accuracy is `0%`. Relative to the committed H02-C shared-oracle `32.725%`, the predeclared recovered-gap ratio is `q=0`. Sending all owner training features therefore does not recover the missing-class transfer that all-class calibration demonstrates.

However, do **not** yet claim that cross-class correspondence is definitively the dominant mechanism or start a relational module. H02-D uses the *same samples that trained each personalized base*, while H02-C uses held-out official-training images disjoint from client training. This creates one remaining confound: the owner features may be unusually in-sample / locally overfit and therefore poor evidence about whether realistic owner-label support on fresh samples could train a transferable shared decoder. The fastest falsifiable control is to keep the same two-class-per-client label support but replace reused local-training images with disjoint held-out owner-class images. This is cheaper and cleaner than inventing a calibration architecture now.

---

# ACTIVE — H02-E: Held-out owner-support control — sample reuse vs cross-class correspondence

## One scientific objective

Test whether H02-D failed because it trained the diagnostic head on **features of the exact samples already used to train each local base**, rather than because sparse per-client label support fundamentally lacks the cross-class correspondence needed for missing-class transfer.

Keep every client restricted to its same two owned labels, but build a fresh held-out owner-support set from unused CIFAR-10 `train=True` images. If held-out owner-support still gives near-zero missing accuracy, the in-sample/overfit explanation is falsified and a coordinate/correspondence calibration experiment becomes justified. If it recovers a large fraction of the H02-C oracle gap, do not pursue relational alignment yet.

## Frozen online trajectory

Use `fedgh`, seed0, 10 rounds and reproduce the committed H02-A online trajectory exactly. This remains an analysis-only side channel. Run the new diagnostic only at rounds **2 and 10**. Do not run seeds1/2 and do not alter online training.

## Held-out owner-support construction

Construct one deterministic index assignment before the run from official CIFAR-10 `train=True` only. For each client `i` and each of its two frozen owned classes `c`:

- select exactly **100 fresh images of class `c`**;
- exclude **all** frozen client-training indices, not only client `i`'s indices;
- exclude all H02-C oracle-calibration indices;
- assignments must be disjoint across clients, including the two owners of the same class, so the same raw image is never passed through two client bases;
- never use CIFAR-10 `train=False` samples or labels for fitting;
- use no augmentation and the same normalization as `prepare()`.

Use a fixed, code-declared RNG seed (choose once, e.g. `271828`, and log it). This must yield exactly 200 samples/client, 2000 total, and 200/class globally. Save every assigned index and hashes/count receipts.

Call this diagnostic set `heldout_owner_support`.

The disjoint-across-clients rule is important: this block must **not** accidentally provide paired-image correspondence between client spaces. The only information available to the probe remains ordinary labeled examples from each client's two owned classes.

## Probe

At rounds 2 and 10, extract the assigned held-out features through the corresponding current frozen client base in eval mode, restoring module modes afterward. Pool the 2000 `(feature,label)` pairs and fit one fresh zero-initialized 512→10 linear head with exactly the same full-batch LBFGS settings used in H02-C/D:

- CE loss;
- `line_search_fn='strong_wolfe'`;
- `max_iter=100`;
- `tolerance_grad=1e-9`;
- `tolerance_change=1e-12`;
- no regularizer and no tuning.

Evaluate on every client's unchanged official-test features and report seen / missing / all / macro plus per-client class counts/correct counts. Call the readout `heldout_owner_probe`.

## Required integrity checks

Add focused assertions/tests proving:

- 200 samples/client, 2000 total, 200/class;
- all held-out indices are disjoint from every frozen client-training index and every H02-C oracle index;
- held-out assignments are mutually disjoint across clients, including same-class owners;
- no official-test sample/label enters fitting;
- online H02-A records are exact at all rounds;
- probe extraction/fitting changes no client parameter/buffer, prototype, persistent server head, module mode, online metric, or CPU/CUDA RNG state;
- all features/losses/gradients/logits/parameters are finite.

Do not weaken existing H02-D tests.

## Fixed references and predeclared branch

Do not rerun these references except for receipt consistency:

- H02-D reused-training owner-sample missing: round2 `0.2125%`, round10 `0%`;
- H02-C all-class shared-oracle missing: round2 `31.45%`, round10 `32.725%`.

Let round10 `H` be `heldout_owner_probe` missing accuracy and define `q_hold = H / 32.725`.

- **If `q_hold <= 0.20` (`H <= 6.545%`)**: fresh owner-label support still recovers at most one fifth of the oracle gap. The local-training-sample reuse confound is rejected. Stop H02 and report that the next justified experiment is a *minimal correspondence-calibration upper bound* (e.g. unlabeled paired anchors + a simple linear/orthogonal alignment), not a complex relational architecture.
- **If `q_hold >= 0.50` (`H >= 16.3625%`)**: fresh owner-support recovers at least half of the oracle gap. H02-D was substantially confounded by in-sample/local-overfit features; next work should study representation preservation or held-out/local distribution summaries, not relational alignment.
- **If `0.20 < q_hold < 0.50`**: both sample-reuse/generalization and cross-class calibration may matter. Report ambiguity and stop; do not invent a method yet.

Report round2 as a secondary diagnostic; make the branch decision from round10 only. Do not move thresholds after seeing results. Low probe training accuracy is a limitation to report, not a reason to retune the solver post hoc.

## Deliverable

Append `CODEX REPORT H02-E` with STATUS, source SHA, commands/run IDs, tests, deterministic held-out index provenance and disjointness hashes, online/state/RNG equivalence, round2/round10 metrics and fit diagnostics, fixed H02-C/D references, computed `q_hold`, artifact locations, and a concise interpretation following the frozen branch.

Do **not** implement Procrustes, relational transport, multi-prototype methods, new losses, or H03 in this block. Await research-lead review after this single control.
