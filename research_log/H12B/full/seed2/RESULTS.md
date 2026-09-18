# H12-A CIFAR100 seed2 portability stress test

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 34.405000 | 0.000000 | 6.881000 | 6.881000 | 100 |
| fedproto | 35.390000 | 0.000000 | 7.078000 | 7.078000 | 100 |
| fedgh | 14.730000 | 0.000000 | 2.946000 | 2.946000 | 55 |
| paired_h07 | 17.085000 | 9.415000 | 10.949000 | 10.949000 | 100 |
| pair_broken_h07 | 25.465000 | 0.288750 | 5.324000 | 5.324000 | 100 |
| native_control | 27.640000 | 0.000000 | 5.528000 | 5.528000 | 94 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 9.126250 pp; paired-minus-native: 9.415000 pp; all gain vs best FedProto/FedGH: 3.871000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.
Ownership classsets SHA256: d14fbe0d88f0f0bb958d201aa117d704c360ccce70219ee880f56a8488d3e652
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0
Classsets: [[3, 10, 11, 13, 23, 37, 41, 42, 44, 52, 60, 67, 70, 73, 78, 81, 94, 95, 97, 99], [3, 7, 10, 14, 22, 23, 35, 37, 42, 44, 52, 54, 62, 68, 76, 84, 91, 94, 95, 99], [1, 6, 7, 12, 14, 20, 22, 28, 33, 35, 54, 58, 62, 63, 68, 76, 82, 84, 86, 91], [1, 4, 6, 8, 12, 19, 20, 28, 30, 33, 36, 45, 46, 58, 63, 72, 75, 82, 86, 96], [4, 8, 19, 30, 32, 36, 39, 40, 43, 45, 46, 53, 61, 72, 75, 79, 80, 90, 93, 96], [0, 9, 15, 16, 21, 29, 32, 39, 40, 43, 53, 55, 61, 64, 69, 79, 80, 90, 92, 93], [0, 9, 15, 16, 17, 21, 26, 29, 34, 38, 48, 49, 51, 55, 59, 64, 66, 69, 71, 92], [2, 17, 18, 24, 25, 26, 34, 38, 48, 49, 51, 57, 59, 65, 66, 71, 74, 83, 85, 89], [2, 5, 18, 24, 25, 27, 31, 47, 50, 56, 57, 65, 74, 77, 83, 85, 87, 88, 89, 98], [5, 11, 13, 27, 31, 41, 47, 50, 56, 60, 67, 70, 73, 77, 78, 81, 87, 88, 97, 98]]

Per-client predicted-class coverage: {"local": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedproto": [20, 20, 20, 20, 20, 20, 20, 20, 20, 21], "fedgh": [9, 10, 11, 10, 7, 7, 11, 10, 9, 6], "paired_h07": [97, 97, 98, 99, 98, 98, 97, 99, 98, 99], "pair_broken_h07": [91, 76, 86, 84, 75, 72, 81, 79, 84, 84], "native_control": [14, 11, 14, 20, 20, 20, 20, 19, 14, 20]}
Runtime/steps: {"local": {"elapsed_seconds": 125.54839706420898, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 134.38167929649353, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 301.51780939102173, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 140.50966787338257
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4982, 4972, 4964, 4976, 4977, 4978, 4969, 4974, 4979, 4973], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 55.830805 | 230.902412 |
| 2 | 62.560472 | 221.182216 |
| 3 | 70.134302 | 222.502806 |
| 4 | 66.530903 | 220.197158 |
| 5 | 70.796098 | 217.610368 |
| 6 | 68.349579 | 223.200598 |
| 7 | 75.321413 | 213.978813 |
| 8 | 76.435892 | 213.802790 |
| 9 | 58.857060 | 222.477241 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.3440500020980835, "missing": 0.0, "all": 0.068809999153018, "macro": 0.06880999989807605}, "cosine": {"seen": 0.3119500011205673, "missing": 0.0, "all": 0.06238999925553799, "macro": 0.062390001490712166}, "l2": {"seen": 0.3254499971866608, "missing": 0.0, "all": 0.06509000025689601, "macro": 0.06509000025689601}}
fedproto: {"head": {"seen": 0.36099999845027925, "missing": 0.0, "all": 0.07219999879598618, "macro": 0.07219999991357326}, "cosine": {"seen": 0.33919999897480013, "missing": 0.0, "all": 0.06783999912440777, "macro": 0.06783999986946583}, "l2": {"seen": 0.35390000343322753, "missing": 0.0, "all": 0.07078000009059907, "macro": 0.07077999897301197}}
fedgh: {"cosine": {"seen": 0.26955000311136246, "missing": 0.0, "all": 0.05390999987721443, "macro": 0.05390999987721443}, "l2": {"seen": 0.279399998486042, "missing": 0.0, "all": 0.055880000814795494, "macro": 0.055879998579621316}, "local_head_pre_server": {"seen": 0.3087000012397766, "missing": 0.0, "all": 0.061740000173449515, "macro": 0.06173999942839146}, "global_head_post_server": {"seen": 0.14730000048875808, "missing": 0.0, "all": 0.029460000060498714, "macro": 0.029459999687969685}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.
