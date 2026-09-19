# H13-A CIFAR100 seed2 mixed-backbone falsifier

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 39.645000 | 0.000000 | 7.929000 | 7.929000 | 100 |
| fedproto | 38.185000 | 0.000000 | 7.637000 | 7.637000 | 100 |
| fedgh | 30.305000 | 0.002500 | 6.063000 | 6.063000 | 100 |
| paired_h07 | 27.555000 | 6.363750 | 10.602000 | 10.602000 | 100 |
| pair_broken_h07 | 28.655000 | 0.406250 | 6.056000 | 6.056000 | 100 |
| native_control | 34.615000 | 0.001250 | 6.924000 | 6.924000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 5.957500 pp; paired-minus-native: 6.362500 pp; all gain vs best FedProto/FedGH: 2.965000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.
Ownership classsets SHA256: d14fbe0d88f0f0bb958d201aa117d704c360ccce70219ee880f56a8488d3e652
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0
Classsets: [[3, 10, 11, 13, 23, 37, 41, 42, 44, 52, 60, 67, 70, 73, 78, 81, 94, 95, 97, 99], [3, 7, 10, 14, 22, 23, 35, 37, 42, 44, 52, 54, 62, 68, 76, 84, 91, 94, 95, 99], [1, 6, 7, 12, 14, 20, 22, 28, 33, 35, 54, 58, 62, 63, 68, 76, 82, 84, 86, 91], [1, 4, 6, 8, 12, 19, 20, 28, 30, 33, 36, 45, 46, 58, 63, 72, 75, 82, 86, 96], [4, 8, 19, 30, 32, 36, 39, 40, 43, 45, 46, 53, 61, 72, 75, 79, 80, 90, 93, 96], [0, 9, 15, 16, 21, 29, 32, 39, 40, 43, 53, 55, 61, 64, 69, 79, 80, 90, 92, 93], [0, 9, 15, 16, 17, 21, 26, 29, 34, 38, 48, 49, 51, 55, 59, 64, 66, 69, 71, 92], [2, 17, 18, 24, 25, 26, 34, 38, 48, 49, 51, 57, 59, 65, 66, 71, 74, 83, 85, 89], [2, 5, 18, 24, 25, 27, 31, 47, 50, 56, 57, 65, 74, 77, 83, 85, 87, 88, 89, 98], [5, 11, 13, 27, 31, 41, 47, 50, 56, 60, 67, 70, 73, 77, 78, 81, 87, 88, 97, 98]]

Per-client predicted-class coverage: {"local": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedproto": [19, 24, 20, 27, 20, 22, 20, 27, 19, 31], "fedgh": [7, 32, 12, 28, 10, 25, 14, 26, 9, 26], "paired_h07": [97, 100, 95, 100, 90, 100, 95, 100, 95, 100], "pair_broken_h07": [68, 57, 72, 60, 53, 52, 66, 68, 59, 61], "native_control": [17, 29, 19, 35, 20, 29, 20, 24, 16, 30]}
Runtime/steps: {"local": {"elapsed_seconds": 274.3668911457062, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 298.3946342468262, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 496.2998332977295, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 165.06491565704346
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4982, 4972, 4964, 4976, 4977, 4978, 4969, 4974, 4979, 4973], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 306.442536 | 372.412597 |
| 2 | 75.244133 | 233.362218 |
| 3 | 316.650409 | 381.232349 |
| 4 | 76.318210 | 225.738877 |
| 5 | 309.623558 | 369.967132 |
| 6 | 70.273963 | 227.663355 |
| 7 | 306.442520 | 362.005951 |
| 8 | 76.657121 | 218.592052 |
| 9 | 308.231566 | 366.164980 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.39644999504089357, "missing": 0.0, "all": 0.07929000034928321, "macro": 0.07928999811410904}, "cosine": {"seen": 0.346000000834465, "missing": 0.00010000000474974513, "all": 0.06928000040352345, "macro": 0.06927999891340733}, "l2": {"seen": 0.352400004863739, "missing": 0.00015000000275904312, "all": 0.07060000076889991, "macro": 0.07060000263154506}}
fedproto: {"head": {"seen": 0.41099999845027924, "missing": 0.0, "all": 0.08219999969005584, "macro": 0.08219999819993973}, "cosine": {"seen": 0.37024999856948854, "missing": 0.0, "all": 0.07405000030994416, "macro": 0.07404999956488609}, "l2": {"seen": 0.38184999823570254, "missing": 0.0, "all": 0.07637000009417534, "macro": 0.07636999897658825}}
fedgh: {"cosine": {"seen": 0.3518999993801117, "missing": 2.5000001187436283e-05, "all": 0.07039999961853027, "macro": 0.07039999850094318}, "l2": {"seen": 0.35305000096559525, "missing": 2.5000001187436283e-05, "all": 0.0706300001591444, "macro": 0.07062999717891216}, "local_head_pre_server": {"seen": 0.37820000052452085, "missing": 0.0, "all": 0.0756400004029274, "macro": 0.07563999965786934}, "global_head_post_server": {"seen": 0.3030500002205372, "missing": 2.5000001187436283e-05, "all": 0.06063000001013279, "macro": 0.06062999963760376}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.

## Architecture initialization and grouped results

Models constructed before upstream client initialization; no pretrained weights. All state hashes include BatchNorm buffers. Client0 FedAvgCNN is the unchanged reference; server head starts from client0 initialhead. All heads512->100. Same clientinitialhashes and actualround1batchhashes acrossarms.

| Client | Architecture | Parameters | Initial model SHA256 | Initial base SHA256 | Initial head SHA256 |
|---|---|---:|---|---|---|
| 0 | FedAvgCNN | 924708 | 6e60ed1a1a7ceeb4ecbf0d9335264b394d66d1407f9475c151e1053995efbec1 | 5299a7c5112386dad4673a01288dc77f30f930a0ee42e003ac482bbe1a8848cd | a8056e967dc15d4ce6b47b0ff0c155072a235efd703677c41ceec0bf1f363cef |
| 1 | ResNet18 | 11227812 | 4eee5c52376a55d62f6b4510edf274e84fd7f319e2726c4aafd2980f5c856bd3 | 6ba3d7807b7c609bdfcf0e94faa754b516633412db7480eb1293de2d1ed23e9e | ea4c0f73f355621c0cfb64fcfbc5884725124f584568ad6fdbf6311c242bc607 |
| 2 | FedAvgCNN | 924708 | cd22291fdae411479607828c19e3e278d383d0a0eaa962a92fea1cc0733a2261 | 57957460cc9f7da9426dd39df22e496aed242719b6a5e099f25ec32175925992 | 8b2eeaf2d163899f0ad1e450fe8a41c3d5f5189c6d184dd065c66d95b1c5d2bd |
| 3 | ResNet18 | 11227812 | e4d97d9351834e5d7bb2757e42f34d4826c4bd232aeccf9098a95d7584e55c66 | 4a9760ea8dc86110032bea1f4e594e316a8eb0cbc664ff04645a39bae84d7e14 | 9ef95712a5fd01f06e7767ecdceb2081e711c6da3275c8b7cba0194e6184f40f |
| 4 | FedAvgCNN | 924708 | 3495b24f79e80f238868600c769a4faa7379427deadb7e4da4bf2102c0b028c6 | ab88ac642ad17411df980c9726205e60cc0634b84fd662836f85f03c944c1667 | 0e895a26cc4db13465cb304eb1166cea16b58c0e45e0c64dd9928ec34b78de5a |
| 5 | ResNet18 | 11227812 | 2329c0751cc8eb53036534f7ec1557785940cff1a07793efccb70274cab06f5a | 164645bbf0da7bcb3cf00ae6c58c95511e59c04afaf006cb84969633cf0c037d | 094798ebfc73838f4820bc15934fc08580fa2bd9643678fe439bd6097613bb83 |
| 6 | FedAvgCNN | 924708 | 573541dfcfb08de2d6b3929bbd2c2e3f4aa0fa1f2e61037f875312ea9a4b39dc | a80a780addbb07070646ae87513c2a445ad1b14823aad4ecdb96e764f5eb2c09 | 087f18f554ad8b7ff1ddb9a8031a1b07990af5942b34982508e26d30db3a2129 |
| 7 | ResNet18 | 11227812 | 1db7eebafac08a06253de8bcfb17dfb59ad56625a06ea64f35fc34cea88a06d8 | 8dc2cbace66b00f25f61edd3849915ed8d02b98eb51ff6ba2507ca657f0ac9a2 | 9e6feadb713fbda1c2178ba520cdb8b3c617c392ccf77c7636b572b8c92b9272 |
| 8 | FedAvgCNN | 924708 | c73b33b7d07c8a4ce95f7f838e841a79d33f40cf5ad060807b7be01e94cb3a0f | a2fa2c393a2070638bc7904f635ee8446388b44cf27a40d3bf118b028ee6e99a | 8198c6e524cc0ad280121985c85e7cf82a074314256a2e71a2593c81d85610de |
| 9 | ResNet18 | 11227812 | 0449ce995ec56a4a81a732960efcd6ed4d5bbd6be28b99098cf641e53fc17cd3 | 6651ee683bfcc9d822adaf6795cecc5219f4535baafc5ad2f07c61e4335a2802 | e9433e048e6f12f5b5b52ceccaa3b6e6513b078e4f504e710fc7458d6e2beff6 |

| Arm | Backbone | Seen % | Missing % | All % | Macro % |
|---|---|---:|---:|---:|---:|
| local | FedAvgCNN | 34.489999 | 0.000000 | 6.898000 | 6.898000 |
| local | ResNet18 | 44.800000 | 0.000000 | 8.960000 | 8.960000 |
| fedproto | FedAvgCNN | 31.520000 | 0.000000 | 6.304000 | 6.304000 |
| fedproto | ResNet18 | 44.850000 | 0.000000 | 8.970000 | 8.970000 |
| fedgh | FedAvgCNN | 15.650000 | 0.000000 | 3.130000 | 3.130000 |
| fedgh | ResNet18 | 44.960000 | 0.005000 | 8.996000 | 8.996000 |
| paired_h07 | FedAvgCNN | 14.570000 | 7.755000 | 9.118000 | 9.118000 |
| paired_h07 | ResNet18 | 40.540000 | 4.972500 | 12.086000 | 12.086000 |
| pair_broken_h07 | FedAvgCNN | 11.430000 | 0.785000 | 2.914000 | 2.914000 |
| pair_broken_h07 | ResNet18 | 45.880000 | 0.027500 | 9.198000 | 9.198000 |
| native_control | FedAvgCNN | 23.890000 | 0.000000 | 4.778000 | 4.778000 |
| native_control | ResNet18 | 45.340000 | 0.002500 | 9.070000 | 9.070000 |

| Arm | Client | Backbone | Seen % | Missing % | All % | Macro % |
|---|---:|---|---:|---:|---:|---:|
| local | 0 | FedAvgCNN | 34.650001 | 0.000000 | 6.930000 | 6.930000 |
| local | 1 | ResNet18 | 49.500000 | 0.000000 | 9.900000 | 9.900001 |
| local | 2 | FedAvgCNN | 38.049999 | 0.000000 | 7.610000 | 7.610000 |
| local | 3 | ResNet18 | 45.249999 | 0.000000 | 9.050000 | 9.050000 |
| local | 4 | FedAvgCNN | 33.399999 | 0.000000 | 6.680000 | 6.680000 |
| local | 5 | ResNet18 | 42.449999 | 0.000000 | 8.490000 | 8.490000 |
| local | 6 | FedAvgCNN | 35.749999 | 0.000000 | 7.150000 | 7.150000 |
| local | 7 | ResNet18 | 45.199999 | 0.000000 | 9.040000 | 9.039999 |
| local | 8 | FedAvgCNN | 30.599999 | 0.000000 | 6.120000 | 6.119999 |
| local | 9 | ResNet18 | 41.600001 | 0.000000 | 8.320000 | 8.320000 |
| fedproto | 0 | FedAvgCNN | 36.199999 | 0.000000 | 7.240000 | 7.240000 |
| fedproto | 1 | ResNet18 | 49.800000 | 0.000000 | 9.960000 | 9.959999 |
| fedproto | 2 | FedAvgCNN | 35.200000 | 0.000000 | 7.040000 | 7.040000 |
| fedproto | 3 | ResNet18 | 46.399999 | 0.000000 | 9.280000 | 9.280000 |
| fedproto | 4 | FedAvgCNN | 29.550001 | 0.000000 | 5.910000 | 5.910000 |
| fedproto | 5 | ResNet18 | 42.399999 | 0.000000 | 8.480000 | 8.480000 |
| fedproto | 6 | FedAvgCNN | 31.250000 | 0.000000 | 6.250000 | 6.250000 |
| fedproto | 7 | ResNet18 | 44.250000 | 0.000000 | 8.850000 | 8.849999 |
| fedproto | 8 | FedAvgCNN | 25.400001 | 0.000000 | 5.080000 | 5.080000 |
| fedproto | 9 | ResNet18 | 41.400000 | 0.000000 | 8.280000 | 8.280001 |
| fedgh | 0 | FedAvgCNN | 16.800000 | 0.000000 | 3.360000 | 3.360000 |
| fedgh | 1 | ResNet18 | 48.550001 | 0.012500 | 9.720000 | 9.719999 |
| fedgh | 2 | FedAvgCNN | 16.050000 | 0.000000 | 3.210000 | 3.210000 |
| fedgh | 3 | ResNet18 | 46.750000 | 0.000000 | 9.350000 | 9.350000 |
| fedgh | 4 | FedAvgCNN | 16.550000 | 0.000000 | 3.310000 | 3.310000 |
| fedgh | 5 | ResNet18 | 45.950001 | 0.000000 | 9.190000 | 9.190000 |
| fedgh | 6 | FedAvgCNN | 16.750000 | 0.000000 | 3.350000 | 3.350000 |
| fedgh | 7 | ResNet18 | 42.699999 | 0.000000 | 8.540000 | 8.540000 |
| fedgh | 8 | FedAvgCNN | 12.100000 | 0.000000 | 2.420000 | 2.420000 |
| fedgh | 9 | ResNet18 | 40.849999 | 0.012500 | 8.180000 | 8.180001 |
| paired_h07 | 0 | FedAvgCNN | 14.700000 | 9.250000 | 10.340000 | 10.340000 |
| paired_h07 | 1 | ResNet18 | 43.900001 | 4.600000 | 12.460000 | 12.460000 |
| paired_h07 | 2 | FedAvgCNN | 17.299999 | 7.300000 | 9.300000 | 9.300000 |
| paired_h07 | 3 | ResNet18 | 41.650000 | 5.362500 | 12.620001 | 12.619999 |
| paired_h07 | 4 | FedAvgCNN | 14.650001 | 6.750000 | 8.330000 | 8.330000 |
| paired_h07 | 5 | ResNet18 | 40.349999 | 4.987500 | 12.060000 | 12.059999 |
| paired_h07 | 6 | FedAvgCNN | 13.500001 | 7.812500 | 8.950000 | 8.950001 |
| paired_h07 | 7 | ResNet18 | 39.199999 | 4.912500 | 11.770000 | 11.770000 |
| paired_h07 | 8 | FedAvgCNN | 12.700000 | 7.662500 | 8.670000 | 8.670000 |
| paired_h07 | 9 | ResNet18 | 37.599999 | 5.000000 | 11.520000 | 11.520001 |
| pair_broken_h07 | 0 | FedAvgCNN | 11.250000 | 0.862500 | 2.940000 | 2.940000 |
| pair_broken_h07 | 1 | ResNet18 | 49.149999 | 0.012500 | 9.840000 | 9.839999 |
| pair_broken_h07 | 2 | FedAvgCNN | 14.250000 | 0.687500 | 3.400000 | 3.400000 |
| pair_broken_h07 | 3 | ResNet18 | 47.250000 | 0.037500 | 9.480000 | 9.480000 |
| pair_broken_h07 | 4 | FedAvgCNN | 11.550000 | 0.525000 | 2.730000 | 2.730000 |
| pair_broken_h07 | 5 | ResNet18 | 45.150000 | 0.012500 | 9.040000 | 9.040001 |
| pair_broken_h07 | 6 | FedAvgCNN | 10.400000 | 1.125000 | 2.980000 | 2.980000 |
| pair_broken_h07 | 7 | ResNet18 | 46.250001 | 0.012500 | 9.260000 | 9.260000 |
| pair_broken_h07 | 8 | FedAvgCNN | 9.700000 | 0.725000 | 2.520000 | 2.520000 |
| pair_broken_h07 | 9 | ResNet18 | 41.600001 | 0.062500 | 8.370000 | 8.370001 |
| native_control | 0 | FedAvgCNN | 22.149999 | 0.000000 | 4.430000 | 4.430000 |
| native_control | 1 | ResNet18 | 48.249999 | 0.012500 | 9.660000 | 9.660000 |
| native_control | 2 | FedAvgCNN | 24.300000 | 0.000000 | 4.860000 | 4.860000 |
| native_control | 3 | ResNet18 | 46.799999 | 0.000000 | 9.360000 | 9.360000 |
| native_control | 4 | FedAvgCNN | 23.150000 | 0.000000 | 4.630000 | 4.630000 |
| native_control | 5 | ResNet18 | 45.550001 | 0.000000 | 9.110000 | 9.110001 |
| native_control | 6 | FedAvgCNN | 27.800000 | 0.000000 | 5.560000 | 5.560000 |
| native_control | 7 | ResNet18 | 44.949999 | 0.000000 | 8.990000 | 8.989999 |
| native_control | 8 | FedAvgCNN | 22.050001 | 0.000000 | 4.410000 | 4.410000 |
| native_control | 9 | ResNet18 | 41.150001 | 0.000000 | 8.230000 | 8.229999 |

Mean centered alignment residual after transform, grouped over non-reference clients (client0 identity excluded):
FedAvgCNN: {"paired": 74.62335689809318, "broken": 226.33912564863726}
ResNet18: {"paired": 309.4781180455053, "broken": 370.3566017528846}

Residual magnitudes depend on feature scale; this grouping is descriptive, not proof of a failure mechanism. Full per-client residuals/orthogonality, classwise counts and histograms remain in final.json. The frozen overallgate is unchanged; groupedmetrics are not selected as alternativegates. Same512Dpayloads asH12; modelparametercounts differ, with no modelweight exchange in these prototype/head baselines. One mixedseed only; do not claim multi-seed architectureheterogeneity yet.
