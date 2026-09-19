# H16-A CIFAR100 nested one-owner seed1 stress test

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 51.220000 | 0.000000 | 5.122000 | 5.122000 | 100 |
| fedproto | 50.500001 | 0.000000 | 5.050000 | 5.050000 | 100 |
| fedgh | 25.340000 | 0.000000 | 2.534000 | 2.534000 | 61 |
| fedavg | 10.070000 | 10.070000 | 10.070000 | 10.070000 | 94 |
| paired_h07 | 23.880000 | 8.306667 | 9.864000 | 9.864000 | 100 |
| pair_broken_h07 | 44.520000 | 0.236667 | 4.665000 | 4.665000 | 100 |
| native_control | 47.840000 | 0.000000 | 4.784000 | 4.784000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 8.070000 pp; paired-minus-native: 8.306667 pp; all gain vs best FedProto/FedGH: 4.814000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "aggregate_coverage_at_least90": true, "mean_client_coverage_at_least80": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.10classes/client,exactly1owner/class; historical ownership order retained.
Ownership classsets SHA256: c65e33d8ffc2a08af3706b0005bedc75df402190eb868803d32074800170a7ce
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: 7a657188e9d5c769a5f0cafb7c7ae06ace4bf2a7d3586a04adf0ce1d458e6d1c
Classsets: [[3, 10, 23, 37, 42, 44, 52, 94, 95, 99], [7, 14, 22, 35, 54, 62, 68, 76, 84, 91], [1, 6, 12, 20, 28, 33, 58, 63, 82, 86], [4, 8, 19, 30, 36, 45, 46, 72, 75, 96], [32, 39, 40, 43, 53, 61, 79, 80, 90, 93], [0, 9, 15, 16, 21, 29, 55, 64, 69, 92], [17, 26, 34, 38, 48, 49, 51, 59, 66, 71], [2, 18, 24, 25, 57, 65, 74, 83, 85, 89], [5, 27, 31, 47, 50, 56, 77, 87, 88, 98], [11, 13, 41, 60, 67, 70, 73, 78, 81, 97]]

Per-client predicted-class coverage: {"local": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "fedproto": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "fedgh": [8, 5, 8, 7, 4, 6, 6, 7, 4, 6], "fedavg": [94, 94, 94, 94, 94, 94, 94, 94, 94, 94], "paired_h07": [97, 98, 95, 98, 98, 98, 98, 93, 97, 98], "pair_broken_h07": [85, 66, 65, 60, 50, 61, 67, 55, 67, 58], "native_control": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
Runtime/steps: {"local": {"elapsed_seconds": 138.12724351882935, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 127.46508622169495, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 302.0286681652069, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedavg": {"elapsed_seconds": 157.6665461063385, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 157.8438482284546
Communication: {"semantic_uplink_bytes": 206400, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4980, 4958, 4973, 4977, 4980, 4971, 4972, 4974, 4982, 4977], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 84.170321 | 228.787871 |
| 2 | 89.637674 | 211.297369 |
| 3 | 101.111952 | 235.837384 |
| 4 | 85.452469 | 217.252641 |
| 5 | 89.692881 | 209.102632 |
| 6 | 87.813791 | 220.883114 |
| 7 | 99.953742 | 212.202853 |
| 8 | 89.345041 | 218.082682 |
| 9 | 87.387460 | 214.119616 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.5122000008821488, "missing": 0.0, "all": 0.05122000016272068, "macro": 0.05121999979019165}, "cosine": {"seen": 0.46509999930858614, "missing": 0.0, "all": 0.046509999781847, "macro": 0.04651000052690506}, "l2": {"seen": 0.47480000257492067, "missing": 0.0, "all": 0.04748000018298626, "macro": 0.047480000928044316}}
fedproto: {"head": {"seen": 0.5185999989509582, "missing": 0.0, "all": 0.05186000019311905, "macro": 0.051859999820590016}, "cosine": {"seen": 0.49380000233650206, "missing": 0.0, "all": 0.049380000308156015, "macro": 0.049379998818039894}, "l2": {"seen": 0.5050000071525573, "missing": 0.0, "all": 0.05050000064074993, "macro": 0.05049999989569187}}
fedgh: {"cosine": {"seen": 0.4620000034570694, "missing": 0.0, "all": 0.04619999974966049, "macro": 0.046199999004602435}, "l2": {"seen": 0.466200003027916, "missing": 0.0, "all": 0.046619999781250955, "macro": 0.046620000153779984}, "local_head_pre_server": {"seen": 0.4649000018835068, "missing": 0.0, "all": 0.04649000018835068, "macro": 0.046489999443292615}, "global_head_post_server": {"seen": 0.25340000092983245, "missing": 0.0, "all": 0.02533999988809228, "macro": 0.02533999951556325}}
fedavg: {"head": {"seen": 0.36590000092983244, "missing": 0.0, "all": 0.03659000005573034, "macro": 0.036589999496936795}, "cosine": {"seen": 0.38299999535083773, "missing": 0.0, "all": 0.03830000050365925, "macro": 0.038300000317394736}, "l2": {"seen": 0.3981000006198883, "missing": 0.0, "all": 0.03981000017374754, "macro": 0.03981000017374754}, "global_model_post_server": {"seen": 0.10070000104606151, "missing": 0.10069999918341636, "all": 0.1006999984383583, "macro": 0.1006999984383583}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.

## Frozen two-owner comparison

| Arm | H12 seen % | H12 missing % | H12 all % |
|---|---:|---:|---:|
| local | 32.440000 | 0.000000 | 6.488000 |
| fedproto | 33.900000 | 0.000000 | 6.780000 |
| fedgh | 16.180000 | 0.000000 | 3.236000 |
| paired_h07 | 15.895000 | 9.345000 | 10.655000 |

Partition comparison: {"h12_train_counts": [4982, 4972, 4964, 4976, 4977, 4978, 4969, 4974, 4979, 4973], "h16_train_counts": [4980, 4958, 4973, 4977, 4980, 4971, 4972, 4974, 4982, 4977], "owners_per_class": [2, 1], "local_label_support": [0.2, 0.1], "historical_order_exact": true, "anchors_exact": true, "train_pool_exact": true}
Cumulative training communication bytes: {"local": {"training_network_bytes": 0, "prototype_metrics_diagnostic_only": true}, "fedproto": {"upload_vectors": 2048000, "upload_counts": 8000, "download_vectors": 20480000}, "fedgh": {"upload_vectors": 2048000, "upload_labels": 8000, "downloaded_head_per_client": 2052000, "downloaded_head_all_clients": 20520000}, "fedavg": {"upload_model_all_clients": 369883200, "download_model_all_clients": 369883200}}
FedAvg missing/all domination positioning warning: True
FedAvg directly reuses pinned PFLlib full-parameter sample-count-weighted aggregation. Local CE SGD equivalence tested; all clients participate, homogeneous CNN has no BN buffers. Matched loaders shuffle with no drop_last; exactly10 updates and final post-aggregation global evaluation. Prototype computations for FedAvg are diagnostic only and never sent/injected. No equal-information claim: PPRTP has same-image anchor side information.
Mechanism gate uses missing and coverage only; comparisons against training baselines are descriptive. Stop after seed0, await lead replication decision.
