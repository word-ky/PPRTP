# H16-B one-owner CIFAR100 stochastic replication

| Seed | Arm | Seen % | Missing % | All % | Macro % | Aggregate coverage |
|---|---|---:|---:|---:|---:|---:|
| 0 | local | 51.070001 | 0.000000 | 5.107000 | 5.107000 | 100 |
| 0 | fedproto | 51.400000 | 0.000000 | 5.140000 | 5.140000 | 100 |
| 0 | fedgh | 24.820000 | 0.000000 | 2.482000 | 2.482000 | 62 |
| 0 | fedavg | 9.810000 | 9.810000 | 9.810000 | 9.809999 | 94 |
| 0 | paired_h07 | 23.320000 | 7.767778 | 9.323000 | 9.323000 | 100 |
| 0 | pair_broken_h07 | 44.160000 | 0.256667 | 4.647000 | 4.647000 | 100 |
| 0 | native_control | 46.910000 | 0.000000 | 4.691000 | 4.691000 | 100 |
| 1 | local | 51.220000 | 0.000000 | 5.122000 | 5.122000 | 100 |
| 1 | fedproto | 50.500001 | 0.000000 | 5.050000 | 5.050000 | 100 |
| 1 | fedgh | 25.340000 | 0.000000 | 2.534000 | 2.534000 | 61 |
| 1 | fedavg | 10.070000 | 10.070000 | 10.070000 | 10.070000 | 94 |
| 1 | paired_h07 | 23.880000 | 8.306667 | 9.864000 | 9.864000 | 100 |
| 1 | pair_broken_h07 | 44.520000 | 0.236667 | 4.665000 | 4.665000 | 100 |
| 1 | native_control | 47.840000 | 0.000000 | 4.784000 | 4.784000 | 100 |
| 2 | local | 52.630000 | 0.000000 | 5.263000 | 5.263000 | 100 |
| 2 | fedproto | 51.410000 | 0.000000 | 5.141000 | 5.141000 | 100 |
| 2 | fedgh | 24.720000 | 0.000000 | 2.472000 | 2.472000 | 63 |
| 2 | fedavg | 10.360000 | 10.360000 | 10.360000 | 10.360000 | 91 |
| 2 | paired_h07 | 23.580000 | 7.993333 | 9.552000 | 9.552000 | 100 |
| 2 | pair_broken_h07 | 44.150000 | 0.250000 | 4.640000 | 4.640000 | 100 |
| 2 | native_control | 47.830000 | 0.000000 | 4.783000 | 4.783000 | 100 |

Mean +/- sample standard deviation (n=3, ddof=1), percentage points.

| Arm | Seen | Missing | All |
|---|---:|---:|---:|
| local | 51.640000 +/- 0.860639 | 0.000000 +/- 0.000000 | 5.164000 +/- 0.086064 |
| fedproto | 51.103333 +/- 0.522525 | 0.000000 +/- 0.000000 | 5.110333 +/- 0.052253 |
| fedgh | 24.960000 +/- 0.332866 | 0.000000 +/- 0.000000 | 2.496000 +/- 0.033287 |
| fedavg | 10.080000 +/- 0.275136 | 10.080000 +/- 0.275136 | 10.080000 +/- 0.275137 |
| paired_h07 | 23.593333 +/- 0.280238 | 8.022593 +/- 0.270633 | 9.579667 +/- 0.271559 |
| pair_broken_h07 | 44.276667 +/- 0.210792 | 0.247778 +/- 0.010184 | 4.650667 +/- 0.012897 |
| native_control | 47.526667 +/- 0.534072 | 0.000000 +/- 0.000000 | 4.752667 +/- 0.053407 |

Frozen verdict: 3/3 STRONG.
Seedwise verdicts: {"0": "STRONG", "1": "STRONG", "2": "STRONG"}
Seedwise gates: {"0": {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "aggregate_coverage_at_least90": true, "mean_client_coverage_at_least80": true}, "1": {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "aggregate_coverage_at_least90": true, "mean_client_coverage_at_least80": true}, "2": {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "aggregate_coverage_at_least90": true, "mean_client_coverage_at_least80": true}}
Gap values (pp): {"paired_minus_broken_missing_pp": [7.51111107529141, 8.07000009343028, 7.7433333377121025], "paired_minus_native_missing_pp": [7.767777740955354, 8.306666761636734, 7.993333339691162], "paired_minus_fedavg_missing_pp": [-2.042222246527671, -1.7633331567049022, -2.3666667193174353], "paired_minus_fedavg_all_pp": [-0.48700012266635895, -0.205999836325646, -0.8080002665519712]}
Gap mean +/- sample SD: {"paired_minus_broken_missing_pp": {"mean": 7.774814835477931, "std": 0.28077134322299097}, "paired_minus_native_missing_pp": {"mean": 8.022592614094417, "std": 0.2706333737858064}, "paired_minus_fedavg_missing_pp": {"mean": -2.0574073741833363, "std": 0.30195328781141295}, "paired_minus_fedavg_all_pp": {"mean": -0.5003334085146587, "std": 0.30122161589401997}}
FedAvg missing/all domination warnings seed0/1/2: [true, true, true]

Exact same H16-A split, owners1/order120100,256 anchors and49744 non-anchor examples. Initial hashes differ across all3seeds; actual first-round minibatch hashes differ for every client across3seeds and pair among4arms within eachseed. Fourarms15590localsteps each; readouts share frozen finalstate/rawmeans/counts and anchorfeature multisets. No labels enter transport fitting.
Initial hashes seed0/1/2: ["45f199c1b832adf0f877a365c0d96fca9521efbd03ed5f13cc09ce49bfe321e0", "953f01f3b69eb39d67d6c42d5fa50b4a30d4383541bb8e10f67f4fe92c623070", "6e60ed1a1a7ceeb4ecbf0d9335264b394d66d1407f9475c151e1053995efbec1"]
Per-seed coverage, runtime, communication, class counts/correct counts, transformations and integrity receipts remain in individual reports/raw artifacts. Seed0 reused unchanged; no seed selection, tuning, rerun or new dataset.
FedAvg is a homogeneous shared full-model baseline without correspondence information. PPRTP is personalized post-hoc and uses extra same-image anchors; preserve the seen/missing tradeoff and FedAvg comparison, do not claim universal accuracy or communication superiority.
Stop after reporting both seeds regardless of outcome; await lead decision before Tiny-ImageNet.
