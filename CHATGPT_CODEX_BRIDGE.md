# ChatGPT ↔ Codex Bridge

This file is the persistent coordination log. Do not delete history. ChatGPT writes the active scientific task; Codex executes it and appends evidence/results.

---

## ACTIVE — Block H01: Bootstrap + falsify the simplest GPC hypothesis

### Scientific hypothesis

FedProto's class-missing weakness may come less from the limited capacity of prototypes and more from **how prototypes are injected into local optimization**. Original FedProto mainly uses a same-class attraction term. A Global Prototype Classifier (GPC) instead uses every global class prototype in one softmax, so locally missing classes become explicit competitors in every local classification gradient.

For normalized local feature `z` and normalized global class prototype `p_c`:

```text
logit_c = s * cosine(z, p_c)
L_gpc = CE([logit_1, ..., logit_C], y)
```

All global classes must appear in the softmax denominator, including classes absent from the current client's local dataset.

### Objective of this block

Create the smallest reproducible codebase that can run a **mechanism test**, not a publication-scale benchmark.

### Required implementation

1. Bootstrap a compact PyTorch FL harness with configuration-driven algorithm selection.
2. Dataset: CIFAR-10 as the first real dataset. If download/runtime blocks execution, add a deterministic synthetic classification dataset only as an implementation smoke test; do not substitute synthetic results for the CIFAR-10 mechanism test.
3. Client split: implement a severe label-subset split where each client owns only `K` of the `C` classes. Make `K` configurable; initial target `K=2`, `C=10`. Log the class set of every client.
4. Model: use one simple common backbone first. We are isolating class-missing heterogeneity before model heterogeneity.
5. Implement three selectable modes: `local`, `fedproto`, and `gpc`.
6. Global prototype aggregation: class-wise, sample-count-aware aggregation over clients that observed that class. Track a validity mask for classes not observed in a round.
7. GPC: L2-normalize feature and valid global prototypes, use cosine logits, configurable scale, detached global prototypes, and include all valid classes in the denominator.
8. Metrics: seen-class accuracy, locally-missing-class accuracy, all-class accuracy, macro-per-class accuracy, convergence/instability.
9. Unit test the key mechanism: for a client lacking class `c`, changing global prototype `p_c` must change GPC loss/feature gradient but not the pure same-class FedProto alignment term.

### Experimental fairness

FedProto and GPC use the same client prototypes, aggregation, backbone, split, optimizer/local epochs/rounds, and seeds. The intended causal difference is knowledge injection: same-class attraction vs all-class competition.

### Stop / escalation

Do not add FedRE, relational GPC, learnable prototypes, ETF anchors, server-side head training, or model heterogeneity in H01. Those are later stages contingent on falsification of the simple mechanism.

---

## CODEX PROGRESS H01 — 2026-09-16 bootstrap / PFLlib requirement

User explicitly requested Jianqing Zhang's PFLlib. Pinned official TsingZ0/PFLlib `0169ba7e412c9856a08bb3faefab1e35f538a3c1` as unchanged submodule. A6000 root created at `/home/wenchang/asdasdsad/wjq/PPRTP`. Upstream FedProto inference uses raw nearest-prototype L2 whereas GPC uses cosine; to isolate training injection, every arm is evaluated with local head plus common cosine and L2 prototype readouts. Sample-count-aware aggregation is used identically for FedProto/GPC.

---

## CHATGPT REVIEW 01 — H01 bootstrap accepted; real-data evidence still missing

Commits `74a3219` and `59af148` pin upstream PFLlib unchanged, add a narrow H01 client, keep official CIFAR-10 train/test separated, implement severe `K=2, C=10`, and add common head/cosine/L2 readouts. Five tests passed, including two-round parity with unmodified PFLlib and the absent-class mechanism test. No CIFAR result had yet been committed.

Required checks before interpretation: measure loss-scale confound via local-CE and lambda-scaled knowledge-gradient norms; exploit exact round-1 pairing because no global knowledge is active in round 1; interpret missing-class accuracy conservatively; keep prototype aggregation identical.

---

## ACTIVE — Block H01-B: Paired CIFAR-10 causal mechanism run

Frozen protocol: `local / fedproto / gpc`, 10 clients, `K=2`, 100 train samples/class, 10 rounds, 1 local epoch, SGD lr .01, lambda 1, GPC scale 10. Add per-round client/prototype hashes and first-active-round gradient diagnostics. Round 1 must be exactly paired across arms. Report round-2 and round-10 head/cosine/L2 seen/missing/all/macro, gradient strengths, prototype norms, runtime, and preserve raw evidence. If seed 0 is stable, run seeds 1/2. Do not add new algorithms.

---

## CHATGPT REVIEW 02 — No new Codex evidence; H01-B remains active

No new evidence after the initial bootstrap; H01-B stayed active unchanged until the real run was committed.

---

## CODEX REPORT H01-B — 2026-09-17 — DONE

Official PFLlib pin unchanged. Implementation source `52a6c8f3d167b7b7827386221238e1576e1eaea8`. Real CIFAR-10 subset: 2000 training / 1000 official-test examples, 10 clients, K=2, seeds 0/1/2, 10 rounds, 1 local epoch, SGD lr .01, batch32, lambda1, scale10; PFLlib CNN512D. No post-hoc tuning. Remote A6000 runs all exited 0; final suite 8 tests passed.

Round-1 pairing passed for all three seeds: every client model hash and global prototype hash identical across all arms. Round-2 hashes diverged as expected.

Round10 common cosine, mean ± sample SD across 3 seeds (percent):

| Training | Seen | Missing | All | Macro |
|---|---:|---:|---:|---:|
| Local, posthoc prototype probe | 66.55 ± 6.51 | 0.00 ± 0.00 | 13.31 ± 1.30 | 13.31 ± 1.30 |
| FedProto | 66.85 ± 5.57 | 0.00 ± 0.00 | 13.37 ± 1.11 | 13.37 ± 1.11 |
| GPC | 69.80 ± 4.93 | 0.00 ± 0.00 | 13.96 ± 0.99 | 13.96 ± 0.99 |

Local-only head all-class accuracy: 12.77±0.46%.

First active global-loss round (round2), client0 first-batch gradients into base:

| Seed | Local norm | FedProto knowledge norm / ratio | GPC knowledge norm / ratio |
|---|---:|---:|---:|
| 0 | 1.21295 | .00947862 / .00781455 | 5.75848 / 4.74752 |
| 1 | 1.12790 | .00800925 / .00710101 | 3.72995 / 3.30698 |
| 2 | 1.24102 | .0109992 / .00886303 | 3.79103 / 3.05477 |

All losses/prototypes finite. GPC exceeded FedProto by only .59 pp in final common-cosine all-class accuracy, entirely from seen classes; both had exactly 0% missing-class accuracy in every seed. Equal lambda created a major strength imbalance: GPC knowledge gradients were 3.05–4.75× local CE while FedProto was below 1% of local CE. Therefore the simple strong missing-class claim is unsupported and objective type remained confounded with optimization strength.

A useful clue: in seed0, after round1 (before any global knowledge loss acts), common-prototype missing-class accuracy was nonzero (cosine ≈11.84%, L2 ≈14.63%); by round2 it collapsed to approximately zero. This suggested possible latent-coordinate drift, but required direct measurement.

---

## CHATGPT REVIEW 03 — H01-B accepted; simple missing-class claim is not supported

H01-B was accepted as a trustworthy negative result. The next experiment was restricted to strength matching and a denominator ablation, plus cross-client same-class prototype cosine as a coordinate-compatibility diagnostic.

### ACTIVE — Block H01-C: Strength-match GPC and isolate the missing-class denominator

Frozen H01-B setup. Keep FedProto lambda 1. Add only:

```text
gpc_all_match: cosine CE over ALL valid global classes, lambda = 0.002, scale = 10
gpc_seen_match: same cosine CE and lambda = 0.002, denominator restricted to the current client's local class_set
```

`lambda=.002` was predeclared from H01-B gradient diagnostics, not tuned on accuracy. Unit tests must prove a missing prototype affects all-class GPC but has exactly no effect on seen-only GPC. Before aggregation each round, log cosine similarity between the two owners' local prototype for every class. First run a seed0 round-2 gate; both GPC arms must be within 0.5×–2× FedProto scaled knowledge-gradient norm. If both pass, run seeds0/1/2 to round10 and report rounds 1/2/5/10 owner-cosine trajectory. If either fails, stop without retuning in H01-C.

---

## CHATGPT REVIEWS 04–12 — H01-C remained active with no new Codex evidence

Across the hourly checks after H01-C assignment, there were no new Codex commits/results until the 20-minute Codex heartbeat was enabled. The research direction was intentionally not changed; the same predeclared H01-C gate remained active.

---

## CODEX REPORT H01-C — 2026-09-17 — DONE (predeclared stop)

STATUS: DONE for implementation and mandatory first-seed gate; extended matrix NOT RUN because the declared strength check failed. Source SHA `c744b1b655955c556728f09d5a5b7769ac0771e4`; run `20260917-102726-h01c-gate`; A6000 CUDA0; exit0. Local and remote suites: **10 tests passed**. New tests establish absent-prototype sensitivity for all-class GPC, exact invariance for seen-only GPC, detached prototype targets, and known-answer owner cosine. Pairing passed: the three arms have identical round-1 client and global-prototype hashes, and rerun FedProto hashes match H01-B in both rounds.

Round2, client0, same first batch:

| Arm | Local CE gradient | Scaled knowledge gradient | Knowledge/local | Knowledge/FedProto | Gate |
|---|---:|---:|---:|---:|---|
| FedProto | 1.21294522 | .00947861932 | .00781454891 | 1 | Reference |
| GPC all match | 1.21294522 | .0115169547 | .00949503295 | 1.21504561 | PASS |
| GPC seen match | 1.21294522 | .000658496167 | .00054289028 | .0694717389 | FAIL |

The required [0.5,2] window had to pass for both GPC arms. Seen-only failed, so Codex correctly stopped after seed0/round2 without tuning. Round2 common cosine discrete accuracies were identical across all three arms: seen 63.65%, missing 0%, all/macro 12.73%; these are descriptive gate-run values, not an accepted causal comparison.

Pre-aggregation same-class owner cosine mean [min,max]:

- Round1 all arms: .989056653 [.987980545,.991554499]
- Round2 FedProto: .908854854 [.887295842,.926129818]
- Round2 GPC-all: .908422846 [.886698127,.925801039]
- Round2 GPC-seen: .908556002 [.886922359,.925877571]

Thus owner-space compatibility falls sharply in one round while common-cosine missing recognition falls from ≈11.84% to zero. This is strong correlational evidence for rapid coordinate drift, but two rounds do not establish causality. Full H01-C artifacts are under `research_log/H01C/` and remote run `20260917-102726-h01c-gate`.

---

## CHATGPT REVIEW 13 — H01-C accepted; match GPC-all vs GPC-seen directly before moving on

### Research-lead verdict

H01-C is correctly implemented and the predeclared stop was handled properly. The failed gate does **not** refute the denominator hypothesis: at the same lambda, the seen-only cosine CE produces only 6.95% of FedProto's scaled knowledge gradient, while all-class GPC is 1.215× FedProto. Therefore an all-vs-seen accuracy comparison at lambda=.002 would mostly compare optimization strength rather than information in missing-class negatives.

The strength gap is itself mechanistically informative: adding the eight locally missing prototypes massively increases the GPC gradient. But we still need to determine whether those prototypes contribute a **useful new gradient direction** or merely a larger force. The owner-cosine drop `.989 -> .909` in one round, simultaneous with missing-class accuracy collapsing to zero, keeps latent-coordinate drift as the leading failure hypothesis, but remains correlational.

### ACTIVE — Block H01-D: Direct gradient-matched denominator ablation

Spend the next work block only on this final simple-GPC falsification. Do not add FedGH, FedRE, relational modules, ETF anchors, trainable prototypes, model heterogeneity, or larger datasets yet.

#### Frozen setup

Keep H01-B/H01-C data, split, model, optimizer, batch order, rounds, scale=10, prototype construction/aggregation, and evaluation unchanged. Use:

```text
fedproto:       lambda = 1.0
gpc_all_match:  lambda = 0.00200
gpc_seen_match: lambda = 0.03498
```

`0.03498` is **predeclared from H01-C gradient norms only**, not from accuracy: `0.002 * (0.0115169547 / 0.000658496167) ≈ 0.03498`. It is chosen so the seed0/round2 seen-only knowledge gradient matches GPC-all, isolating denominator content rather than magnitude. Do not sweep or retune these lambdas in this block.

#### Mandatory seed0 gate

Run seed0 for 2 rounds first. Round1 hashes must again match exactly. On client0 first batch of round2 report scaled knowledge-gradient norms for all three arms. Require `gpc_seen_match / gpc_all_match` to lie in `[0.8, 1.25]`. If this fails, stop and report; do not tune again.

Also, on the **same round2 seed0 feature tensor and prototype bank before the optimizer update**, compute both GPC losses regardless of the arm and report:

- cosine similarity between flattened `dL_all/dz` and `dL_seen/dz`;
- their unscaled norm ratio;
- cosine similarity after norm matching is of course unchanged, so do not duplicate it;
- optionally the norm of the residual after projecting `g_all` onto `g_seen` if trivial to add.

This diagnostic is important: if gradient cosine is near 1, missing-class prototypes mainly amplify an existing direction; if substantially lower, they inject genuinely different optimization information.

#### If the gate passes

Run seeds `0,1,2` for 10 rounds with the frozen settings. Report mean ± sample SD at rounds 2 and 10 for common-cosine seen/missing/all/macro; preserve head/L2 in artifacts. For rounds 1/2/5/10 report cross-client same-class owner-prototype cosine mean/min/max. Also report the client0 first-batch knowledge-gradient ratio `seen/all` at rounds 2/5/10 for each seed so we know whether the initial strength match stays approximately valid.

No accuracy-driven hyperparameter search is allowed.

### Decision rule after H01-D

- If matched `gpc_all_match` materially exceeds matched `gpc_seen_match`, especially on missing/all behavior, and their gradient directions differ, missing-class competition has a real signal worth refining.
- If the two matched GPC variants are essentially indistinguishable and missing accuracy remains near zero, **reject the simple GPC denominator thesis**. The next block should then compare prototype-based transfer against a shared global decision head (FedGH-style) to test whether the key missing ingredient is a shared decision coordinate system.
- If owner same-class cosine continues to collapse in parallel with missing recognition, treat coordinate drift as a primary mechanism to test next; do not jump directly to a relational architecture before a shared-head control.
- Do not interpret seen-only gains as missing-class transfer.

### Deliverable

Append `CODEX REPORT H01-D` with source SHA, exact commands, tests, pairing gate, gradient-direction diagnostic, full accepted result table if the gate passes, owner-cosine trajectory, any suspicious behavior, and a concise recommendation. Do not independently start H02.
