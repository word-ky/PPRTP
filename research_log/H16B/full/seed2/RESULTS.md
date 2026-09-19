# H16-A CIFAR100 nested one-owner seed2 stress test

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 52.630000 | 0.000000 | 5.263000 | 5.263000 | 100 |
| fedproto | 51.410000 | 0.000000 | 5.141000 | 5.141000 | 100 |
| fedgh | 24.720000 | 0.000000 | 2.472000 | 2.472000 | 63 |
| fedavg | 10.360000 | 10.360000 | 10.360000 | 10.360000 | 91 |
| paired_h07 | 23.580000 | 7.993333 | 9.552000 | 9.552000 | 100 |
| pair_broken_h07 | 44.150000 | 0.250000 | 4.640000 | 4.640000 | 100 |
| native_control | 47.830000 | 0.000000 | 4.783000 | 4.783000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 7.743333 pp; paired-minus-native: 7.993333 pp; all gain vs best FedProto/FedGH: 4.411000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "aggregate_coverage_at_least90": true, "mean_client_coverage_at_least80": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.10classes/client,exactly1owner/class; historical ownership order retained.
Ownership classsets SHA256: c65e33d8ffc2a08af3706b0005bedc75df402190eb868803d32074800170a7ce
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: 7a657188e9d5c769a5f0cafb7c7ae06ace4bf2a7d3586a04adf0ce1d458e6d1c
Classsets: [[3, 10, 23, 37, 42, 44, 52, 94, 95, 99], [7, 14, 22, 35, 54, 62, 68, 76, 84, 91], [1, 6, 12, 20, 28, 33, 58, 63, 82, 86], [4, 8, 19, 30, 36, 45, 46, 72, 75, 96], [32, 39, 40, 43, 53, 61, 79, 80, 90, 93], [0, 9, 15, 16, 21, 29, 55, 64, 69, 92], [17, 26, 34, 38, 48, 49, 51, 59, 66, 71], [2, 18, 24, 25, 57, 65, 74, 83, 85, 89], [5, 27, 31, 47, 50, 56, 77, 87, 88, 98], [11, 13, 41, 60, 67, 70, 73, 78, 81, 97]]

Per-client predicted-class coverage: {"local": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "fedproto": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "fedgh": [8, 5, 9, 7, 5, 5, 5, 7, 6, 6], "fedavg": [91, 91, 91, 91, 91, 91, 91, 91, 91, 91], "paired_h07": [98, 98, 98, 98, 98, 98, 98, 93, 98, 96], "pair_broken_h07": [86, 68, 66, 55, 53, 60, 66, 59, 61, 56], "native_control": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
Runtime/steps: {"local": {"elapsed_seconds": 119.77108526229858, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 144.18640995025635, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 316.23562502861023, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedavg": {"elapsed_seconds": 173.57649612426758, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 143.325448513031
Communication: {"semantic_uplink_bytes": 206400, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4980, 4958, 4973, 4977, 4980, 4971, 4972, 4974, 4982, 4977], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 84.164159 | 236.279043 |
| 2 | 93.762120 | 222.862688 |
| 3 | 95.192407 | 238.241055 |
| 4 | 86.877063 | 221.044339 |
| 5 | 89.246920 | 219.866796 |
| 6 | 90.343334 | 226.336319 |
| 7 | 100.314665 | 216.004644 |
| 8 | 90.489328 | 227.562380 |
| 9 | 93.539971 | 221.446859 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.5263000011444092, "missing": 0.0, "all": 0.052629999443888666, "macro": 0.052629999443888666}, "cosine": {"seen": 0.4690999984741211, "missing": 0.0, "all": 0.04690999984741211, "macro": 0.04690999947488308}, "l2": {"seen": 0.4769000023603439, "missing": 0.0, "all": 0.04769000001251698, "macro": 0.04768999963998795}}
fedproto: {"head": {"seen": 0.5358999997377396, "missing": 0.0, "all": 0.053590000793337825, "macro": 0.05359000004827976}, "cosine": {"seen": 0.5076999962329865, "missing": 0.0, "all": 0.050769999995827673, "macro": 0.05077000074088574}, "l2": {"seen": 0.5141000002622604, "missing": 0.0, "all": 0.05140999965369701, "macro": 0.051410000398755075}}
fedgh: {"cosine": {"seen": 0.46330000162124635, "missing": 0.0, "all": 0.04632999934256077, "macro": 0.046330000087618825}, "l2": {"seen": 0.4689999967813492, "missing": 0.0, "all": 0.04689999967813492, "macro": 0.04690000005066395}, "local_head_pre_server": {"seen": 0.49270000755786897, "missing": 0.0, "all": 0.04927000030875206, "macro": 0.04926999993622303}, "global_head_post_server": {"seen": 0.24719999879598617, "missing": 0.0, "all": 0.024719999730587007, "macro": 0.024720000196248294}}
fedavg: {"head": {"seen": 0.40900000035762785, "missing": 0.0, "all": 0.040899999998509885, "macro": 0.04090000055730343}, "cosine": {"seen": 0.3849000006914139, "missing": 0.0, "all": 0.038490000553429125, "macro": 0.03848999999463558}, "l2": {"seen": 0.4041000008583069, "missing": 0.0, "all": 0.040409999899566174, "macro": 0.0404100002720952}, "global_model_post_server": {"seen": 0.10360000021755696, "missing": 0.10360000059008598, "all": 0.10360000282526016, "macro": 0.10359999537467957}}

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
