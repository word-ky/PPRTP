# H12-A CIFAR100 seed0 portability stress test

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 32.440000 | 0.000000 | 6.488000 | 6.488000 | 100 |
| fedproto | 33.900000 | 0.000000 | 6.780000 | 6.780000 | 100 |
| fedgh | 16.180000 | 0.000000 | 3.236000 | 3.236000 | 57 |
| paired_h07 | 15.895000 | 9.345000 | 10.655000 | 10.655000 | 100 |
| pair_broken_h07 | 25.050000 | 0.270000 | 5.226000 | 5.226000 | 100 |
| native_control | 27.220000 | 0.000000 | 5.444000 | 5.444000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 9.075000 pp; paired-minus-native: 9.345000 pp; all gain vs best FedProto/FedGH: 3.875000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.
Ownership classsets SHA256: d14fbe0d88f0f0bb958d201aa117d704c360ccce70219ee880f56a8488d3e652
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0
Classsets: [[3, 10, 11, 13, 23, 37, 41, 42, 44, 52, 60, 67, 70, 73, 78, 81, 94, 95, 97, 99], [3, 7, 10, 14, 22, 23, 35, 37, 42, 44, 52, 54, 62, 68, 76, 84, 91, 94, 95, 99], [1, 6, 7, 12, 14, 20, 22, 28, 33, 35, 54, 58, 62, 63, 68, 76, 82, 84, 86, 91], [1, 4, 6, 8, 12, 19, 20, 28, 30, 33, 36, 45, 46, 58, 63, 72, 75, 82, 86, 96], [4, 8, 19, 30, 32, 36, 39, 40, 43, 45, 46, 53, 61, 72, 75, 79, 80, 90, 93, 96], [0, 9, 15, 16, 21, 29, 32, 39, 40, 43, 53, 55, 61, 64, 69, 79, 80, 90, 92, 93], [0, 9, 15, 16, 17, 21, 26, 29, 34, 38, 48, 49, 51, 55, 59, 64, 66, 69, 71, 92], [2, 17, 18, 24, 25, 26, 34, 38, 48, 49, 51, 57, 59, 65, 66, 71, 74, 83, 85, 89], [2, 5, 18, 24, 25, 27, 31, 47, 50, 56, 57, 65, 74, 77, 83, 85, 87, 88, 89, 98], [5, 11, 13, 27, 31, 41, 47, 50, 56, 60, 67, 70, 73, 77, 78, 81, 87, 88, 97, 98]]

Per-client predicted-class coverage: {"local": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedproto": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedgh": [10, 11, 14, 9, 7, 8, 12, 9, 7, 8], "paired_h07": [96, 96, 98, 99, 99, 99, 97, 97, 98, 98], "pair_broken_h07": [86, 77, 77, 78, 68, 67, 77, 79, 76, 78], "native_control": [18, 15, 20, 20, 20, 19, 20, 20, 19, 20]}
Runtime/steps: {"local": {"elapsed_seconds": 122.3267240524292, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 126.4438259601593, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 318.40789461135864, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 157.53215837478638
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4982, 4972, 4964, 4976, 4977, 4978, 4969, 4974, 4979, 4973], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 52.269709 | 230.016351 |
| 2 | 66.224053 | 227.162935 |
| 3 | 74.226026 | 223.639736 |
| 4 | 78.897059 | 228.002096 |
| 5 | 75.556632 | 218.146904 |
| 6 | 75.871303 | 220.665208 |
| 7 | 76.355540 | 218.337443 |
| 8 | 77.281129 | 213.149710 |
| 9 | 51.997931 | 222.556350 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.3243999987840652, "missing": 0.0, "all": 0.06488000005483627, "macro": 0.06487999968230725}, "cosine": {"seen": 0.29619999825954435, "missing": 0.0, "all": 0.05923999957740307, "macro": 0.05923999957740307}, "l2": {"seen": 0.3133499979972839, "missing": 0.0, "all": 0.06267000064253807, "macro": 0.06266999877989292}}
fedproto: {"head": {"seen": 0.3449499994516373, "missing": 0.0, "all": 0.06898999996483327, "macro": 0.06899000108242034}, "cosine": {"seen": 0.32069999873638155, "missing": 0.0, "all": 0.06413999907672405, "macro": 0.06414000019431114}, "l2": {"seen": 0.33899999856948854, "missing": 0.0, "all": 0.06779999919235706, "macro": 0.06780000142753125}}
fedgh: {"cosine": {"seen": 0.2703000009059906, "missing": 0.0, "all": 0.05406000018119812, "macro": 0.05405999906361103}, "l2": {"seen": 0.28569999784231187, "missing": 0.0, "all": 0.05713999941945076, "macro": 0.0571399986743927}, "local_head_pre_server": {"seen": 0.29594999700784685, "missing": 0.0, "all": 0.059189999103546144, "macro": 0.0591899998486042}, "global_head_post_server": {"seen": 0.16179999932646752, "missing": 0.0, "all": 0.03235999960452318, "macro": 0.032359999418258664}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.
