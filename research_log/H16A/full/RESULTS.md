# H16-A CIFAR100 nested one-owner seed0 stress test

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 51.070001 | 0.000000 | 5.107000 | 5.107000 | 100 |
| fedproto | 51.400000 | 0.000000 | 5.140000 | 5.140000 | 100 |
| fedgh | 24.820000 | 0.000000 | 2.482000 | 2.482000 | 62 |
| fedavg | 9.810000 | 9.810000 | 9.810000 | 9.809999 | 94 |
| paired_h07 | 23.320000 | 7.767778 | 9.323000 | 9.323000 | 100 |
| pair_broken_h07 | 44.160000 | 0.256667 | 4.647000 | 4.647000 | 100 |
| native_control | 46.910000 | 0.000000 | 4.691000 | 4.691000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 7.511111 pp; paired-minus-native: 7.767778 pp; all gain vs best FedProto/FedGH: 4.183000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "aggregate_coverage_at_least90": true, "mean_client_coverage_at_least80": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.10classes/client,exactly1owner/class; historical ownership order retained.
Ownership classsets SHA256: c65e33d8ffc2a08af3706b0005bedc75df402190eb868803d32074800170a7ce
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: 7a657188e9d5c769a5f0cafb7c7ae06ace4bf2a7d3586a04adf0ce1d458e6d1c
Classsets: [[3, 10, 23, 37, 42, 44, 52, 94, 95, 99], [7, 14, 22, 35, 54, 62, 68, 76, 84, 91], [1, 6, 12, 20, 28, 33, 58, 63, 82, 86], [4, 8, 19, 30, 36, 45, 46, 72, 75, 96], [32, 39, 40, 43, 53, 61, 79, 80, 90, 93], [0, 9, 15, 16, 21, 29, 55, 64, 69, 92], [17, 26, 34, 38, 48, 49, 51, 59, 66, 71], [2, 18, 24, 25, 57, 65, 74, 83, 85, 89], [5, 27, 31, 47, 50, 56, 77, 87, 88, 98], [11, 13, 41, 60, 67, 70, 73, 78, 81, 97]]

Per-client predicted-class coverage: {"local": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "fedproto": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "fedgh": [8, 6, 9, 5, 3, 6, 7, 7, 4, 7], "fedavg": [94, 94, 94, 94, 94, 94, 94, 94, 94, 94], "paired_h07": [97, 99, 100, 97, 98, 97, 97, 93, 98, 98], "pair_broken_h07": [80, 68, 63, 52, 47, 54, 62, 51, 58, 55], "native_control": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
Runtime/steps: {"local": {"elapsed_seconds": 138.34135174751282, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 132.43120431900024, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 322.5044662952423, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedavg": {"elapsed_seconds": 181.30979180335999, "optimizer_steps_total": 15590, "steps_per_client_round": [156, 155, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 146.05774307250977
Communication: {"semantic_uplink_bytes": 206400, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4980, 4958, 4973, 4977, 4980, 4971, 4972, 4974, 4982, 4977], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 82.248653 | 231.874790 |
| 2 | 98.989027 | 219.959928 |
| 3 | 96.829930 | 235.506125 |
| 4 | 83.604682 | 220.703554 |
| 5 | 81.408256 | 214.496368 |
| 6 | 88.881254 | 223.492046 |
| 7 | 102.763749 | 213.112042 |
| 8 | 80.123529 | 220.241126 |
| 9 | 88.128587 | 216.594712 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.5107000052928925, "missing": 0.0, "all": 0.05106999985873699, "macro": 0.05107000060379505}, "cosine": {"seen": 0.4680000007152557, "missing": 0.0, "all": 0.04680000022053719, "macro": 0.046799999102950095}, "l2": {"seen": 0.4757000029087067, "missing": 0.0, "all": 0.04757000021636486, "macro": 0.04757000058889389}}
fedproto: {"head": {"seen": 0.5233999907970428, "missing": 0.0, "all": 0.05234000012278557, "macro": 0.05233999863266945}, "cosine": {"seen": 0.5088999986648559, "missing": 0.0, "all": 0.05089000016450882, "macro": 0.05088999979197979}, "l2": {"seen": 0.5139999955892562, "missing": 0.0, "all": 0.05140000022947788, "macro": 0.05139999911189079}}
fedgh: {"cosine": {"seen": 0.4610999941825867, "missing": 0.0, "all": 0.04611000046133995, "macro": 0.04611000120639801}, "l2": {"seen": 0.46849999725818636, "missing": 0.0, "all": 0.04685000032186508, "macro": 0.04685000032186508}, "local_head_pre_server": {"seen": 0.4652999997138977, "missing": 0.0, "all": 0.046530000120401385, "macro": 0.04652999937534332}, "global_head_post_server": {"seen": 0.24819999933242798, "missing": 0.0, "all": 0.024819999933242798, "macro": 0.024819999281316996}}
fedavg: {"head": {"seen": 0.3760000020265579, "missing": 0.0, "all": 0.03759999983012676, "macro": 0.03760000020265579}, "cosine": {"seen": 0.38039999902248384, "missing": 0.0, "all": 0.038039999827742574, "macro": 0.038039999268949035}, "l2": {"seen": 0.39959999918937683, "missing": 3.333333297632635e-05, "all": 0.03999000005424023, "macro": 0.0399899996817112}, "global_model_post_server": {"seen": 0.09809999987483024, "missing": 0.09809999987483024, "all": 0.09809999912977219, "macro": 0.09809999167919159}}

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
