# H17-A Tiny-ImageNet one-owner seed0 stress test

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 43.270000 | 0.000000 | 4.327000 | 4.327000 | 200 |
| fedproto | 42.750000 | 0.000000 | 4.275000 | 4.275000 | 200 |
| fedgh | 22.040000 | 0.000000 | 2.204000 | 2.204000 | 165 |
| fedavg | 8.550000 | 8.550000 | 8.550000 | 8.550000 | 180 |
| paired_h07 | 21.430000 | 6.013333 | 7.555000 | 7.555000 | 200 |
| pair_broken_h07 | 37.730000 | 0.034444 | 3.804000 | 3.804000 | 200 |
| native_control | 39.150000 | 0.000000 | 3.915000 | 3.915000 | 200 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 5.978889 pp; paired-minus-native: 6.013333 pp; all gain vs best FedProto/FedGH: 3.280000 pp.
Gates: {"missing_at_least2": true, "native_gap_at_least1_5": true, "broken_gap_at_least1": true, "aggregate_coverage_at_least160": true, "mean_client_coverage_at_least120": true}

Split/initialization/all-arm round1 hashes identical. Exact100000-index coverage:99744 disjoint clienttrain+256 label-blind anchors;10000 official validation-only evaluation.20classes/client,exactly1owner/class; Tiny ownership order120200, one graph only.
Ownership classsets SHA256: e0fb7960f966a7af62fb83b7547157725ec3cf36ba903c54b6a9804fd206c743
Ownership order SHA256: f91bddeeaec559537050cf64b98424ef7b5abe33adeab7af8087adfda0b20619
Anchor SHA256: a0c9e30eebe32b9c14bf31ca2c4eda354874ac65408fbd4ee191a44d5ef1a9ab
Split-file SHA256: b5d1f5c015e5dfdec1a1311319d2cb49643b74a93dd86df60b3f87e02e02d3c4
Classsets: [[12, 19, 45, 51, 54, 71, 80, 84, 103, 107, 114, 123, 128, 130, 180, 181, 187, 189, 192, 197], [9, 18, 32, 39, 44, 66, 73, 85, 88, 93, 104, 109, 126, 129, 135, 137, 140, 175, 178, 198], [2, 14, 46, 53, 59, 69, 76, 105, 106, 113, 115, 148, 150, 154, 168, 169, 174, 176, 190, 193], [11, 16, 25, 34, 38, 48, 52, 60, 87, 100, 118, 120, 125, 139, 144, 145, 155, 161, 164, 166], [7, 23, 35, 40, 49, 50, 70, 77, 83, 94, 101, 110, 116, 121, 143, 158, 159, 160, 165, 179], [0, 20, 21, 29, 58, 68, 79, 91, 97, 111, 112, 122, 124, 133, 134, 151, 167, 171, 182, 183], [3, 6, 13, 17, 22, 31, 41, 55, 75, 86, 95, 119, 132, 149, 153, 170, 172, 177, 195, 196], [1, 28, 33, 37, 42, 47, 61, 72, 78, 81, 82, 90, 92, 127, 142, 163, 188, 191, 194, 199], [4, 5, 15, 24, 26, 27, 30, 62, 65, 89, 96, 98, 99, 102, 141, 146, 147, 152, 157, 184], [8, 10, 36, 43, 56, 57, 63, 64, 67, 74, 108, 117, 131, 136, 138, 156, 162, 173, 185, 186]]

Per-client predicted-class coverage: {"local": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedproto": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedgh": [18, 19, 16, 15, 15, 15, 20, 15, 17, 15], "fedavg": [180, 180, 180, 180, 180, 180, 180, 180, 180, 180], "paired_h07": [194, 197, 198, 191, 193, 191, 192, 194, 193, 189], "pair_broken_h07": [134, 80, 83, 79, 104, 114, 86, 95, 75, 76], "native_control": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]}
Runtime/steps: {"local": {"elapsed_seconds": 265.679132938385, "optimizer_steps_total": 31200, "steps_per_client_round": [312, 312, 312, 312, 312, 312, 312, 312, 312, 312]}, "fedproto": {"elapsed_seconds": 326.86575412750244, "optimizer_steps_total": 31200, "steps_per_client_round": [312, 312, 312, 312, 312, 312, 312, 312, 312, 312]}, "fedgh": {"elapsed_seconds": 539.8007516860962, "optimizer_steps_total": 31200, "steps_per_client_round": [312, 312, 312, 312, 312, 312, 312, 312, 312, 312]}, "fedavg": {"elapsed_seconds": 351.0275356769562, "optimizer_steps_total": 31200, "steps_per_client_round": [312, 312, 312, 312, 312, 312, 312, 312, 312, 312]}}
Readout diagnostic seconds: 171.6620466709137
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 409600, "global_vectors_downlink_total": 4096000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..199; no separate IDs transmitted", "learned_head_downlink_per_client": 410400, "learned_head_downlink_total": 4104000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [9975, 9979, 9972, 9978, 9965, 9978, 9973, 9971, 9971, 9982], "pprtp_total": 102304, "matched_native_extra_refresh_total": 99744}
Broken control additionally refreshes2560anchor features+99744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 124.036626 | 258.310328 |
| 2 | 120.575863 | 261.218353 |
| 3 | 125.676653 | 255.413148 |
| 4 | 133.117437 | 282.847720 |
| 5 | 136.990037 | 289.944210 |
| 6 | 117.785318 | 268.054497 |
| 7 | 120.527819 | 273.693876 |
| 8 | 123.575010 | 257.948888 |
| 9 | 117.317329 | 257.216804 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.43269999921321867, "missing": 0.0, "all": 0.04327000044286251, "macro": 0.04326999969780445}, "cosine": {"seen": 0.3848000019788742, "missing": 0.0, "all": 0.038479999825358394, "macro": 0.03847999945282936}, "l2": {"seen": 0.3818000018596649, "missing": 0.0, "all": 0.038180000707507135, "macro": 0.038179999217391014}}
fedproto: {"head": {"seen": 0.44159999787807463, "missing": 0.0, "all": 0.04416000060737133, "macro": 0.04415999874472618}, "cosine": {"seen": 0.4261000037193298, "missing": 0.0, "all": 0.04261000081896782, "macro": 0.04260999895632267}, "l2": {"seen": 0.4274999976158142, "missing": 0.0, "all": 0.04275000020861626, "macro": 0.042750001326203345}}
fedgh: {"cosine": {"seen": 0.3796999990940094, "missing": 0.0, "all": 0.03797000013291836, "macro": 0.03797000013291836}, "l2": {"seen": 0.37900000214576723, "missing": 0.0, "all": 0.03790000006556511, "macro": 0.03790000081062317}, "local_head_pre_server": {"seen": 0.4100999981164932, "missing": 0.0, "all": 0.04101000018417835, "macro": 0.04101000018417835}, "global_head_post_server": {"seen": 0.22040000110864638, "missing": 0.0, "all": 0.02203999999910593, "macro": 0.022039999812841417}}
fedavg: {"head": {"seen": 0.3372000053524971, "missing": 0.0, "all": 0.03372000008821487, "macro": 0.033719999343156816}, "cosine": {"seen": 0.3233999967575073, "missing": 0.0, "all": 0.03233999982476234, "macro": 0.03233999889343977}, "l2": {"seen": 0.32660000026226044, "missing": 1.1111111234640703e-05, "all": 0.03267000000923872, "macro": 0.03267000038176775}, "global_model_post_server": {"seen": 0.08550000041723252, "missing": 0.08549999967217445, "all": 0.08550000190734863, "macro": 0.08550000190734863}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.

Local seen readiness >=10%: True; actual 43.270000%. Global chance0.5%. Readiness precedes mechanism classification.
Training client counts: [9975, 9979, 9972, 9978, 9965, 9978, 9973, 9971, 9971, 9982]
Raw structure 200classes x500train/50officialval; exact lexical ImageFolder class mapping. Per-file SHA256 and ordered-file/index hashes in split.json; val_annotations.txt parsed against train mapping, no official test data used.
Cumulative training communication bytes: {"local": {"training_network_bytes": 0, "prototype_metrics_diagnostic_only": true}, "fedproto": {"upload_vectors": 4096000, "upload_counts": 16000, "download_vectors": 40960000}, "fedgh": {"upload_vectors": 4096000, "upload_labels": 16000, "downloaded_head_per_client": 4104000, "downloaded_head_all_clients": 41040000}, "fedavg": {"upload_model_all_clients": 2277840000, "download_model_all_clients": 2277840000}}
FedAvg missing/all domination positioning warning: True
Pinned PFLlib FedAvgCNN dim10816,64x64RGB,512Dbase/200head,random initialization; ToTensor + Normalize(.5),no resize/augmentation/pretraining. Matched batch32/lr.01/localepoch1/10cycles. PPRTP extra unlabeled same-image correspondence remains. No universal superiority or communication-efficiency claim.
Stop after seed0 regardless of verdict; no result-conditioned tuning or extra rounds.
