# H14-A CIFAR100 ownership seed1 / training seed0 replication

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 33.380000 | 0.000000 | 6.676000 | 6.676000 | 100 |
| fedproto | 33.210000 | 0.000000 | 6.642000 | 6.642000 | 100 |
| fedgh | 14.335000 | 0.000000 | 2.867000 | 2.867000 | 60 |
| paired_h07 | 16.620000 | 9.035000 | 10.552000 | 10.552000 | 100 |
| pair_broken_h07 | 23.525000 | 0.375000 | 5.005000 | 5.005000 | 100 |
| native_control | 27.070000 | 0.000000 | 5.414000 | 5.414000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 8.660000 pp; paired-minus-native: 9.035000 pp; all gain vs best FedProto/FedGH: 3.910000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.
Ownership classsets SHA256: 462f14367ae84761514d70b1b33445e966c23f65797e413c262094fe965e8d83
Ownership order SHA256: 6860bb34e44af88356db6107db927bcc7759137dd7d415b49d1135b02f1510cb
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: df7f7f5a2228489bd62d2415be8cf3229437e01729639751f4484daea1bdd7f2
Classsets: [[4, 20, 22, 23, 30, 34, 37, 39, 45, 48, 51, 58, 59, 63, 65, 74, 80, 86, 93, 95], [1, 8, 14, 20, 22, 26, 29, 30, 31, 36, 37, 45, 48, 51, 63, 67, 73, 86, 90, 93], [1, 8, 11, 14, 16, 25, 26, 29, 31, 36, 52, 56, 62, 64, 67, 69, 73, 79, 90, 91], [9, 11, 12, 13, 16, 25, 27, 44, 52, 56, 62, 64, 69, 70, 72, 79, 81, 85, 91, 96], [3, 6, 9, 12, 13, 27, 32, 44, 53, 57, 70, 71, 72, 75, 81, 83, 85, 87, 88, 96], [3, 6, 17, 32, 41, 50, 53, 57, 66, 68, 71, 75, 78, 83, 87, 88, 94, 97, 98, 99], [0, 2, 17, 24, 28, 35, 38, 41, 46, 50, 54, 60, 66, 68, 78, 92, 94, 97, 98, 99], [0, 2, 5, 21, 24, 28, 33, 35, 38, 40, 42, 43, 46, 47, 49, 54, 60, 61, 82, 92], [5, 7, 10, 15, 18, 19, 21, 33, 40, 42, 43, 47, 49, 55, 61, 76, 77, 82, 84, 89], [4, 7, 10, 15, 18, 19, 23, 34, 39, 55, 58, 59, 65, 74, 76, 77, 80, 84, 89, 95]]

Per-client predicted-class coverage: {"local": [20, 20, 20, 20, 20, 20, 19, 20, 20, 20], "fedproto": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedgh": [13, 13, 7, 9, 9, 12, 10, 10, 5, 6], "paired_h07": [97, 98, 98, 99, 100, 98, 98, 97, 99, 98], "pair_broken_h07": [88, 73, 81, 82, 78, 72, 80, 71, 73, 75], "native_control": [20, 19, 17, 20, 20, 20, 20, 20, 19, 19]}
Runtime/steps: {"local": {"elapsed_seconds": 116.29366040229797, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 125.78542447090149, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 248.41395592689514, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 117.27929520606995
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4983, 4976, 4973, 4973, 4973, 4968, 4971, 4978, 4974, 4975], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 54.386441 | 235.437176 |
| 2 | 79.947111 | 234.011297 |
| 3 | 95.673879 | 231.828557 |
| 4 | 87.980167 | 226.736758 |
| 5 | 78.653620 | 230.729323 |
| 6 | 85.060982 | 226.576305 |
| 7 | 95.091398 | 235.338856 |
| 8 | 88.920840 | 225.231513 |
| 9 | 61.138075 | 228.673874 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.3338000029325485, "missing": 0.0, "all": 0.0667600005865097, "macro": 0.06675999909639359}, "cosine": {"seen": 0.2878000020980835, "missing": 0.0, "all": 0.057559999823570254, "macro": 0.05755999907851219}, "l2": {"seen": 0.30370000302791594, "missing": 0.0, "all": 0.060739998891949656, "macro": 0.06074000000953674}}
fedproto: {"head": {"seen": 0.349850007891655, "missing": 0.0, "all": 0.06997000016272067, "macro": 0.06996999941766262}, "cosine": {"seen": 0.318599995970726, "missing": 0.0, "all": 0.06371999979019165, "macro": 0.06372000128030778}, "l2": {"seen": 0.3321000039577484, "missing": 0.0, "all": 0.06642000004649162, "macro": 0.06641999930143357}}
fedgh: {"cosine": {"seen": 0.26299999803304674, "missing": 0.0, "all": 0.05260000042617321, "macro": 0.05260000005364418}, "l2": {"seen": 0.27694999873638154, "missing": 0.0, "all": 0.055390000343322754, "macro": 0.05539000071585178}, "local_head_pre_server": {"seen": 0.3021499991416931, "missing": 0.0, "all": 0.06043000034987926, "macro": 0.06043000034987926}, "global_head_post_server": {"seen": 0.14335000216960908, "missing": 0.0, "all": 0.02867000047117472, "macro": 0.028670000098645686}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.

## Ownership replication receipts

H14-A verdict: STRONG. All frozen metric definitions and gates unchanged.
Paired all minus Local all: 3.876000 pp (diagnostic, not replacement gate).
Ownership RNG seed1; training seed0; first balanced cyclic construction, no graph search. Labeled class-client graph changes; unlabeled client-neighbor cycle topology is retained.
Ownership distance: {"old_incidence_count": 200, "new_incidence_count": 200, "removed": 165, "added": 165, "changed_binary_incidence_entries": 330, "changed_fraction_of_1000_entries": 0.33, "removed_fraction_of_200_old_edges": 0.825, "classes_with_changed_owner_pair": 92, "per_client_jaccard": [0.08108108108108109, 0.08108108108108109, 0.1111111111111111, 0.08108108108108109, 0.14285714285714285, 0.05263157894736842, 0.14285714285714285, 0.1111111111111111, 0.14285714285714285, 0.02564102564102564], "mean_client_jaccard": 0.09723094986252881}
Per-class owner pairs: {"0": [6, 7], "1": [1, 2], "10": [8, 9], "11": [2, 3], "12": [3, 4], "13": [3, 4], "14": [1, 2], "15": [8, 9], "16": [2, 3], "17": [5, 6], "18": [8, 9], "19": [8, 9], "2": [6, 7], "20": [0, 1], "21": [7, 8], "22": [0, 1], "23": [0, 9], "24": [6, 7], "25": [2, 3], "26": [1, 2], "27": [3, 4], "28": [6, 7], "29": [1, 2], "3": [4, 5], "30": [0, 1], "31": [1, 2], "32": [4, 5], "33": [7, 8], "34": [0, 9], "35": [6, 7], "36": [1, 2], "37": [0, 1], "38": [6, 7], "39": [0, 9], "4": [0, 9], "40": [7, 8], "41": [5, 6], "42": [7, 8], "43": [7, 8], "44": [3, 4], "45": [0, 1], "46": [6, 7], "47": [7, 8], "48": [0, 1], "49": [7, 8], "5": [7, 8], "50": [5, 6], "51": [0, 1], "52": [2, 3], "53": [4, 5], "54": [6, 7], "55": [8, 9], "56": [2, 3], "57": [4, 5], "58": [0, 9], "59": [0, 9], "6": [4, 5], "60": [6, 7], "61": [7, 8], "62": [2, 3], "63": [0, 1], "64": [2, 3], "65": [0, 9], "66": [5, 6], "67": [1, 2], "68": [5, 6], "69": [2, 3], "7": [8, 9], "70": [3, 4], "71": [4, 5], "72": [3, 4], "73": [1, 2], "74": [0, 9], "75": [4, 5], "76": [8, 9], "77": [8, 9], "78": [5, 6], "79": [2, 3], "8": [1, 2], "80": [0, 9], "81": [3, 4], "82": [7, 8], "83": [4, 5], "84": [8, 9], "85": [3, 4], "86": [0, 1], "87": [4, 5], "88": [4, 5], "89": [8, 9], "9": [3, 4], "90": [1, 2], "91": [2, 3], "92": [6, 7], "93": [0, 1], "94": [5, 6], "95": [0, 9], "96": [3, 4], "97": [5, 6], "98": [5, 6], "99": [5, 6]}

Exact historical anchors/non-anchor pool/test indices; historical seed0 initialmodel hash retained. Actual round1 batches pair acrossall3arms. This is one new graph and one trainingseed, not general graph/topology robustness. No tuning or alternate reference.
