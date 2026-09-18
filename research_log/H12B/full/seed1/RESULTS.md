# H12-A CIFAR100 seed1 portability stress test

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 33.780000 | 0.000000 | 6.756000 | 6.756000 | 100 |
| fedproto | 34.285000 | 0.001250 | 6.858000 | 6.858000 | 100 |
| fedgh | 16.080000 | 0.000000 | 3.216000 | 3.216000 | 60 |
| paired_h07 | 17.110000 | 9.588750 | 11.093000 | 11.093000 | 99 |
| pair_broken_h07 | 25.180000 | 0.288750 | 5.267000 | 5.267000 | 100 |
| native_control | 27.665000 | 0.000000 | 5.533000 | 5.533000 | 99 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 9.300000 pp; paired-minus-native: 9.588750 pp; all gain vs best FedProto/FedGH: 4.235000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.
Ownership classsets SHA256: d14fbe0d88f0f0bb958d201aa117d704c360ccce70219ee880f56a8488d3e652
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0
Classsets: [[3, 10, 11, 13, 23, 37, 41, 42, 44, 52, 60, 67, 70, 73, 78, 81, 94, 95, 97, 99], [3, 7, 10, 14, 22, 23, 35, 37, 42, 44, 52, 54, 62, 68, 76, 84, 91, 94, 95, 99], [1, 6, 7, 12, 14, 20, 22, 28, 33, 35, 54, 58, 62, 63, 68, 76, 82, 84, 86, 91], [1, 4, 6, 8, 12, 19, 20, 28, 30, 33, 36, 45, 46, 58, 63, 72, 75, 82, 86, 96], [4, 8, 19, 30, 32, 36, 39, 40, 43, 45, 46, 53, 61, 72, 75, 79, 80, 90, 93, 96], [0, 9, 15, 16, 21, 29, 32, 39, 40, 43, 53, 55, 61, 64, 69, 79, 80, 90, 92, 93], [0, 9, 15, 16, 17, 21, 26, 29, 34, 38, 48, 49, 51, 55, 59, 64, 66, 69, 71, 92], [2, 17, 18, 24, 25, 26, 34, 38, 48, 49, 51, 57, 59, 65, 66, 71, 74, 83, 85, 89], [2, 5, 18, 24, 25, 27, 31, 47, 50, 56, 57, 65, 74, 77, 83, 85, 87, 88, 89, 98], [5, 11, 13, 27, 31, 41, 47, 50, 56, 60, 67, 70, 73, 77, 78, 81, 87, 88, 97, 98]]

Per-client predicted-class coverage: {"local": [20, 20, 18, 20, 20, 20, 20, 20, 20, 20], "fedproto": [20, 20, 20, 20, 20, 20, 20, 20, 20, 21], "fedgh": [11, 10, 13, 9, 9, 9, 10, 8, 9, 5], "paired_h07": [98, 99, 98, 99, 99, 98, 97, 97, 97, 98], "pair_broken_h07": [94, 73, 84, 87, 62, 75, 78, 84, 74, 78], "native_control": [18, 16, 19, 20, 20, 20, 20, 20, 18, 18]}
Runtime/steps: {"local": {"elapsed_seconds": 123.33317923545837, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 128.9828999042511, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 307.7901141643524, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 145.1019265651703
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4982, 4972, 4964, 4976, 4977, 4978, 4969, 4974, 4979, 4973], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 54.076729 | 225.219882 |
| 2 | 65.158956 | 219.149016 |
| 3 | 71.019887 | 213.862338 |
| 4 | 70.647997 | 218.709878 |
| 5 | 72.929046 | 211.136341 |
| 6 | 69.812012 | 213.392147 |
| 7 | 69.766690 | 212.631880 |
| 8 | 74.968998 | 209.193577 |
| 9 | 47.802259 | 214.059015 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.3378000020980835, "missing": 0.0, "all": 0.06755999960005284, "macro": 0.06756000071763993}, "cosine": {"seen": 0.30550000071525574, "missing": 0.0, "all": 0.06110000051558018, "macro": 0.06109999977052212}, "l2": {"seen": 0.31714999973773955, "missing": 0.0, "all": 0.06343000121414662, "macro": 0.06342999897897243}}
fedproto: {"head": {"seen": 0.35239999890327456, "missing": 0.0, "all": 0.07048000022768974, "macro": 0.07047999911010265}, "cosine": {"seen": 0.3300000011920929, "missing": 0.0, "all": 0.06600000001490117, "macro": 0.06600000075995922}, "l2": {"seen": 0.342849999666214, "missing": 1.2500000593718141e-05, "all": 0.06857999935746192, "macro": 0.0685799989849329}}
fedgh: {"cosine": {"seen": 0.2675000041723251, "missing": 0.0, "all": 0.05350000001490116, "macro": 0.05350000187754631}, "l2": {"seen": 0.28010000139474867, "missing": 0.0, "all": 0.05602000057697296, "macro": 0.05601999945938587}, "local_head_pre_server": {"seen": 0.3034999966621399, "missing": 0.0, "all": 0.06069999933242798, "macro": 0.06069999970495701}, "global_head_post_server": {"seen": 0.16080000177025794, "missing": 0.0, "all": 0.032160000130534175, "macro": 0.032160000130534175}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.
