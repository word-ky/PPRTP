# H12-B CIFAR100 fixed-split stochastic replication

| Seed | Readout | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | local | 32.440000 | 0.000000 | 6.488000 | 6.488000 | 100 |
| 0 | fedproto | 33.900000 | 0.000000 | 6.780000 | 6.780000 | 100 |
| 0 | fedgh | 16.180000 | 0.000000 | 3.236000 | 3.236000 | 57 |
| 0 | paired_h07 | 15.895000 | 9.345000 | 10.655000 | 10.655000 | 100 |
| 0 | pair_broken_h07 | 25.050000 | 0.270000 | 5.226000 | 5.226000 | 100 |
| 0 | native_control | 27.220000 | 0.000000 | 5.444000 | 5.444000 | 100 |
| 1 | local | 33.780000 | 0.000000 | 6.756000 | 6.756000 | 100 |
| 1 | fedproto | 34.285000 | 0.001250 | 6.858000 | 6.858000 | 100 |
| 1 | fedgh | 16.080000 | 0.000000 | 3.216000 | 3.216000 | 60 |
| 1 | paired_h07 | 17.110000 | 9.588750 | 11.093000 | 11.093000 | 99 |
| 1 | pair_broken_h07 | 25.180000 | 0.288750 | 5.267000 | 5.267000 | 100 |
| 1 | native_control | 27.665000 | 0.000000 | 5.533000 | 5.533000 | 99 |
| 2 | local | 34.405000 | 0.000000 | 6.881000 | 6.881000 | 100 |
| 2 | fedproto | 35.390000 | 0.000000 | 7.078000 | 7.078000 | 100 |
| 2 | fedgh | 14.730000 | 0.000000 | 2.946000 | 2.946000 | 55 |
| 2 | paired_h07 | 17.085000 | 9.415000 | 10.949000 | 10.949000 | 100 |
| 2 | pair_broken_h07 | 25.465000 | 0.288750 | 5.324000 | 5.324000 | 100 |
| 2 | native_control | 27.640000 | 0.000000 | 5.528000 | 5.528000 | 94 |

Mean +/- sample SD (n=3,ddof=1), percentage points:

| Readout | Seen | Missing | All |
|---|---:|---:|---:|
| local | 33.541667 +/- 1.003947 | 0.000000 +/- 0.000000 | 6.708333 +/- 0.200789 |
| fedproto | 34.525000 +/- 0.773450 | 0.000417 +/- 0.000722 | 6.905333 +/- 0.154536 |
| fedgh | 15.663333 +/- 0.809835 | 0.000000 +/- 0.000000 | 3.132667 +/- 0.161967 |
| paired_h07 | 16.696667 +/- 0.694376 | 9.449583 +/- 0.125501 | 10.899000 +/- 0.223240 |
| pair_broken_h07 | 25.231667 +/- 0.212270 | 0.282500 +/- 0.010825 | 5.272333 +/- 0.049217 |
| native_control | 27.508333 +/- 0.250016 | 0.000000 +/- 0.000000 | 5.501667 +/- 0.050003 |

Frozen verdict: 3/3 STRONG: accept CIFAR100 fixed-split stochastic replication.
Seedwise strong: [True, True, True].
Seedwise paired-minus-broken missing gaps (pp): [9.075000012526289, 9.299999930663034, 9.12625007564202].
Seedwise paired all gains vs better preregistered FedProto/FedGH (pp): [3.875000141561032, 4.235000088810921, 3.871000036597251].
Gap means +/- sampleSD: {"paired_minus_broken_missing_pp": {"mean": 9.167083339610448, "std": 0.11792691539730044}, "paired_all_gain_vs_best_FL_pp": {"mean": 3.993666755656401, "std": 0.20901036691641173}}

Exact same H12-A split, ownershipgraph120100, anchors161803 and allocation110001 acrossallseeds. This is initialization/training-shuffle stochastic replication, NOT ownership-graph replication. Allthree initial model hashes are distinct; within each seed allthree arms share initialization and round1 pairing. No tuning/seed selection.
Initial model SHA256 seed0/1/2: ["45f199c1b832adf0f877a365c0d96fca9521efbd03ed5f13cc09ce49bfe321e0", "953f01f3b69eb39d67d6c42d5fa50b4a30d4383541bb8e10f67f4fe92c623070", "6e60ed1a1a7ceeb4ecbf0d9335264b394d66d1407f9475c151e1053995efbec1"]
Exact common split-file SHA256: a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0

Seed0 is unchanged committed H12-A; seeds1/2 rawoutputs and reports include full residuals, classwise/perclient counts, coverage, steps, hashes, permutation receipts, communication and warnings. Aggregatecoverage does not imply everyclient predicts everyclass. Seen/missing tradeoff remains. No newnumericalablation, architecture, method, online-training or communication-efficiency claim.
