# H13-A CIFAR100 seed1 mixed-backbone falsifier

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 39.100000 | 0.000000 | 7.820000 | 7.820000 | 100 |
| fedproto | 36.140001 | 0.001250 | 7.229000 | 7.229000 | 100 |
| fedgh | 30.590000 | 0.003750 | 6.121000 | 6.121000 | 100 |
| paired_h07 | 27.565000 | 6.383750 | 10.620000 | 10.620000 | 100 |
| pair_broken_h07 | 29.270000 | 0.363750 | 6.145000 | 6.145000 | 100 |
| native_control | 35.050000 | 0.002500 | 7.012000 | 7.012000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 6.020000 pp; paired-minus-native: 6.381250 pp; all gain vs best FedProto/FedGH: 3.391000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.
Ownership classsets SHA256: d14fbe0d88f0f0bb958d201aa117d704c360ccce70219ee880f56a8488d3e652
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0
Classsets: [[3, 10, 11, 13, 23, 37, 41, 42, 44, 52, 60, 67, 70, 73, 78, 81, 94, 95, 97, 99], [3, 7, 10, 14, 22, 23, 35, 37, 42, 44, 52, 54, 62, 68, 76, 84, 91, 94, 95, 99], [1, 6, 7, 12, 14, 20, 22, 28, 33, 35, 54, 58, 62, 63, 68, 76, 82, 84, 86, 91], [1, 4, 6, 8, 12, 19, 20, 28, 30, 33, 36, 45, 46, 58, 63, 72, 75, 82, 86, 96], [4, 8, 19, 30, 32, 36, 39, 40, 43, 45, 46, 53, 61, 72, 75, 79, 80, 90, 93, 96], [0, 9, 15, 16, 21, 29, 32, 39, 40, 43, 53, 55, 61, 64, 69, 79, 80, 90, 92, 93], [0, 9, 15, 16, 17, 21, 26, 29, 34, 38, 48, 49, 51, 55, 59, 64, 66, 69, 71, 92], [2, 17, 18, 24, 25, 26, 34, 38, 48, 49, 51, 57, 59, 65, 66, 71, 74, 83, 85, 89], [2, 5, 18, 24, 25, 27, 31, 47, 50, 56, 57, 65, 74, 77, 83, 85, 87, 88, 89, 98], [5, 11, 13, 27, 31, 41, 47, 50, 56, 60, 67, 70, 73, 77, 78, 81, 87, 88, 97, 98]]

Per-client predicted-class coverage: {"local": [20, 20, 19, 20, 20, 20, 20, 20, 20, 20], "fedproto": [20, 21, 20, 26, 19, 23, 20, 37, 20, 27], "fedgh": [10, 25, 11, 32, 10, 25, 13, 28, 10, 27], "paired_h07": [96, 98, 98, 100, 90, 97, 91, 98, 90, 96], "pair_broken_h07": [76, 61, 80, 60, 58, 57, 67, 63, 71, 53], "native_control": [20, 26, 19, 33, 19, 35, 19, 30, 20, 27]}
Runtime/steps: {"local": {"elapsed_seconds": 255.15855884552002, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 275.28959345817566, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 485.99683260917664, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 171.16046142578125
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4982, 4972, 4964, 4976, 4977, 4978, 4969, 4974, 4979, 4973], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 307.283121 | 374.101984 |
| 2 | 75.395669 | 228.916456 |
| 3 | 313.519040 | 371.484716 |
| 4 | 71.207758 | 226.592315 |
| 5 | 308.549549 | 366.943039 |
| 6 | 69.751249 | 219.708976 |
| 7 | 311.626373 | 370.172636 |
| 8 | 80.514024 | 217.563600 |
| 9 | 308.030352 | 369.595106 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.39099999964237214, "missing": 0.0, "all": 0.07819999977946282, "macro": 0.07820000201463699}, "cosine": {"seen": 0.34484999626874924, "missing": 3.7500001781154424e-05, "all": 0.06900000050663949, "macro": 0.06900000087916851}, "l2": {"seen": 0.35080000162124636, "missing": 6.250000151339919e-05, "all": 0.07020999975502491, "macro": 0.070210000872612}}
fedproto: {"head": {"seen": 0.39375, "missing": 0.0, "all": 0.07875000089406967, "macro": 0.07874999940395355}, "cosine": {"seen": 0.35764999985694884, "missing": 1.2500000593718141e-05, "all": 0.07154000028967858, "macro": 0.07154000028967858}, "l2": {"seen": 0.3614000052213669, "missing": 1.2500000593718141e-05, "all": 0.07228999957442284, "macro": 0.07229000106453895}}
fedgh: {"cosine": {"seen": 0.3436000034213066, "missing": 6.25000029685907e-05, "all": 0.0687699992209673, "macro": 0.06877000033855438}, "l2": {"seen": 0.34649999886751176, "missing": 5.0000002374872565e-05, "all": 0.06933999955654144, "macro": 0.06934000067412853}, "local_head_pre_server": {"seen": 0.36549999490380286, "missing": 0.0, "all": 0.07310000136494636, "macro": 0.07309999987483025}, "global_head_post_server": {"seen": 0.3059000037610531, "missing": 3.7500001781154424e-05, "all": 0.06120999902486801, "macro": 0.06120999753475189}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.

## Architecture initialization and grouped results

Models constructed before upstream client initialization; no pretrained weights. All state hashes include BatchNorm buffers. Client0 FedAvgCNN is the unchanged reference; server head starts from client0 initialhead. All heads512->100. Same clientinitialhashes and actualround1batchhashes acrossarms.

| Client | Architecture | Parameters | Initial model SHA256 | Initial base SHA256 | Initial head SHA256 |
|---|---|---:|---|---|---|
| 0 | FedAvgCNN | 924708 | 953f01f3b69eb39d67d6c42d5fa50b4a30d4383541bb8e10f67f4fe92c623070 | 0e29bc0e49677c4414e7d0d7cab2192cbd031481e336adf24f6dd282e47f4941 | 17c8557c4cecf09f07cbfafd4f6be53ccb9e08737f96b9678e14c0ee38b2428d |
| 1 | ResNet18 | 11227812 | bf898b0e89f8d2d1cbfea169086b9b06f6841822f9d8a804bb8dcb05b35ebf98 | 0fa73357a274ef59eefe5ea133300384a2f8d2f2942a499976e3fb687e6fc843 | c37545b97ace94f3f51659fd17098bcee27160d25b668e963ab38328e3982380 |
| 2 | FedAvgCNN | 924708 | 81cd60be15bb9c5936675a4b6e7bf3e1d4d8d4e5a42e6d074c7f0f0d0450cd57 | 5676c88ea6a0fa0989ed3be7b70daac83d875aec92616a782cb859a92c3ad0f4 | 88e38658bfc40648d3d7f702082f60746b2c20a438f51d4a6de98e7236dcbde3 |
| 3 | ResNet18 | 11227812 | 4d92966ebc2ddbbdd6f4c0a18efef0c397044155459797800c8520949547e42d | 299cf21a6afd89f0ed10a23c729df25da05db250e35904677edd41b37432922d | 264cf5e0136b059c32c8cbaba504fa50ea9f42fe7d7c90962a87e8ab0039b634 |
| 4 | FedAvgCNN | 924708 | c64afa49d6d1ea66452371a21f06cd7f8417f1df882755cbe53fbca028f48a5a | 8f4f2528a9475c48f51cd557e87e7d0ed3a753cb0d914aec998520622d75b54b | 4107bf6458ee28645aad2028c953d44701fee7cc53e9a719add8fd92a1e77150 |
| 5 | ResNet18 | 11227812 | f2c6a46427e8d89a91b6151d63aac1ec8ccba948787fb080dc2ea01f570e301d | 27e9d0740cd2e11bc01c219d782d23a5c7c4817fada03447a537616120ad0f79 | 1eb0b51fcd703419b6e361c7f5602e308113f155b2aa140434ea481159e2fb4a |
| 6 | FedAvgCNN | 924708 | 71d9e3e2b5a9898fc42d24388d68a961dc9a6bb0207e2a97055f18b03049745f | b42c7596312a6bcee1ff4dc957a3d8c479c8a2862368bf025e2409fff54c64cb | ef61eacac6f69b157b0da4f2208a5703c74790efc0c77ead841ef5ba15c8dd2a |
| 7 | ResNet18 | 11227812 | 87b0d993dfc75c4e72d9771a136b1b9f900e7bf17bd33f0b9ac14a0beec10404 | 0cf7dfc79f9b7fb4d34f399cf5c000983189d35d3f517e1223a9cd29e26eb8d2 | 238592265f9f88dd5395504e5f058b4f3c63c021555585a7b7099cc61b86877f |
| 8 | FedAvgCNN | 924708 | c8af2c666759ef3bd01253d852ff961757bd94da7923456fb086433ec6dd1ba5 | 319c25d0538c386202c939ae6f28b55cbbbc5419e6c3205f6c7f7677e23b0636 | 0049b363d10b826990cd82785d48f377b6d308d890dfc8a82c151b3ed48e4548 |
| 9 | ResNet18 | 11227812 | bd39f161b1c6294265671c4da86c0d2b6cfe2cdb5a1e64cd4028fe42d6a73eff | 795b1e4fc8fd567904d86039b323588ae181254d757b0e013b96c3f353f7f87b | e136dd146d4ff19955a8936ceab0d22e00b056f17655ac87fdada5f034030b90 |

| Arm | Backbone | Seen % | Missing % | All % | Macro % |
|---|---|---:|---:|---:|---:|
| local | FedAvgCNN | 33.250000 | 0.000000 | 6.650000 | 6.650000 |
| local | ResNet18 | 44.949999 | 0.000000 | 8.990000 | 8.990000 |
| fedproto | FedAvgCNN | 29.390000 | 0.000000 | 5.878000 | 5.878000 |
| fedproto | ResNet18 | 42.890001 | 0.002500 | 8.580000 | 8.580000 |
| fedgh | FedAvgCNN | 16.980000 | 0.000000 | 3.396000 | 3.396000 |
| fedgh | ResNet18 | 44.200001 | 0.007500 | 8.846000 | 8.846000 |
| paired_h07 | FedAvgCNN | 14.410000 | 7.990000 | 9.274000 | 9.274000 |
| paired_h07 | ResNet18 | 40.719999 | 4.777500 | 11.966000 | 11.966000 |
| pair_broken_h07 | FedAvgCNN | 12.500000 | 0.710000 | 3.068000 | 3.068000 |
| pair_broken_h07 | ResNet18 | 46.040000 | 0.017500 | 9.222000 | 9.222000 |
| native_control | FedAvgCNN | 24.710000 | 0.000000 | 4.942000 | 4.942000 |
| native_control | ResNet18 | 45.390000 | 0.005000 | 9.082000 | 9.082000 |

| Arm | Client | Backbone | Seen % | Missing % | All % | Macro % |
|---|---:|---|---:|---:|---:|---:|
| local | 0 | FedAvgCNN | 41.450000 | 0.000000 | 8.290000 | 8.290000 |
| local | 1 | ResNet18 | 52.749997 | 0.000000 | 10.550000 | 10.550001 |
| local | 2 | FedAvgCNN | 27.050000 | 0.000000 | 5.410000 | 5.410000 |
| local | 3 | ResNet18 | 45.449999 | 0.000000 | 9.090000 | 9.090000 |
| local | 4 | FedAvgCNN | 33.500001 | 0.000000 | 6.700000 | 6.700000 |
| local | 5 | ResNet18 | 41.549999 | 0.000000 | 8.310000 | 8.310001 |
| local | 6 | FedAvgCNN | 31.900001 | 0.000000 | 6.380000 | 6.380000 |
| local | 7 | ResNet18 | 43.900001 | 0.000000 | 8.780000 | 8.780000 |
| local | 8 | FedAvgCNN | 32.350001 | 0.000000 | 6.470000 | 6.470000 |
| local | 9 | ResNet18 | 41.100001 | 0.000000 | 8.220000 | 8.220001 |
| fedproto | 0 | FedAvgCNN | 33.149999 | 0.000000 | 6.630000 | 6.630000 |
| fedproto | 1 | ResNet18 | 50.050002 | 0.000000 | 10.010000 | 10.010001 |
| fedproto | 2 | FedAvgCNN | 30.350000 | 0.000000 | 6.070000 | 6.070000 |
| fedproto | 3 | ResNet18 | 47.600001 | 0.012500 | 9.530000 | 9.530000 |
| fedproto | 4 | FedAvgCNN | 27.050000 | 0.000000 | 5.410000 | 5.410000 |
| fedproto | 5 | ResNet18 | 39.300001 | 0.000000 | 7.860000 | 7.860000 |
| fedproto | 6 | FedAvgCNN | 29.300001 | 0.000000 | 5.860000 | 5.860000 |
| fedproto | 7 | ResNet18 | 41.400000 | 0.000000 | 8.280000 | 8.280000 |
| fedproto | 8 | FedAvgCNN | 27.100000 | 0.000000 | 5.420000 | 5.420000 |
| fedproto | 9 | ResNet18 | 36.100000 | 0.000000 | 7.220000 | 7.220000 |
| fedgh | 0 | FedAvgCNN | 19.000000 | 0.000000 | 3.800000 | 3.800000 |
| fedgh | 1 | ResNet18 | 49.000001 | 0.000000 | 9.800000 | 9.799999 |
| fedgh | 2 | FedAvgCNN | 21.850000 | 0.000000 | 4.370000 | 4.370000 |
| fedgh | 3 | ResNet18 | 46.799999 | 0.000000 | 9.360000 | 9.360000 |
| fedgh | 4 | FedAvgCNN | 16.949999 | 0.000000 | 3.390000 | 3.390000 |
| fedgh | 5 | ResNet18 | 42.050001 | 0.025000 | 8.430000 | 8.430000 |
| fedgh | 6 | FedAvgCNN | 15.250000 | 0.000000 | 3.050000 | 3.050000 |
| fedgh | 7 | ResNet18 | 44.100001 | 0.012500 | 8.830000 | 8.830000 |
| fedgh | 8 | FedAvgCNN | 11.850000 | 0.000000 | 2.370000 | 2.370000 |
| fedgh | 9 | ResNet18 | 39.050001 | 0.000000 | 7.810000 | 7.810000 |
| paired_h07 | 0 | FedAvgCNN | 16.550000 | 9.525000 | 10.930000 | 10.930000 |
| paired_h07 | 1 | ResNet18 | 44.499999 | 4.975000 | 12.880000 | 12.879999 |
| paired_h07 | 2 | FedAvgCNN | 14.350000 | 8.575000 | 9.730000 | 9.729999 |
| paired_h07 | 3 | ResNet18 | 43.849999 | 4.450000 | 12.330000 | 12.330000 |
| paired_h07 | 4 | FedAvgCNN | 15.449999 | 6.787500 | 8.520000 | 8.520000 |
| paired_h07 | 5 | ResNet18 | 36.600000 | 4.075000 | 10.580000 | 10.580000 |
| paired_h07 | 6 | FedAvgCNN | 13.600001 | 7.825000 | 8.980000 | 8.979999 |
| paired_h07 | 7 | ResNet18 | 41.049999 | 5.050000 | 12.250000 | 12.250001 |
| paired_h07 | 8 | FedAvgCNN | 12.100000 | 7.237500 | 8.210000 | 8.210000 |
| paired_h07 | 9 | ResNet18 | 37.599999 | 5.337500 | 11.790000 | 11.790000 |
| pair_broken_h07 | 0 | FedAvgCNN | 8.550000 | 0.787500 | 2.340000 | 2.340000 |
| pair_broken_h07 | 1 | ResNet18 | 50.900000 | 0.000000 | 10.180000 | 10.179999 |
| pair_broken_h07 | 2 | FedAvgCNN | 11.750000 | 1.137500 | 3.260000 | 3.260000 |
| pair_broken_h07 | 3 | ResNet18 | 48.800001 | 0.012500 | 9.770000 | 9.769999 |
| pair_broken_h07 | 4 | FedAvgCNN | 19.800000 | 0.212500 | 4.130000 | 4.130000 |
| pair_broken_h07 | 5 | ResNet18 | 42.449999 | 0.012500 | 8.500000 | 8.500000 |
| pair_broken_h07 | 6 | FedAvgCNN | 10.450000 | 0.737500 | 2.680000 | 2.680000 |
| pair_broken_h07 | 7 | ResNet18 | 45.899999 | 0.012500 | 9.190000 | 9.190000 |
| pair_broken_h07 | 8 | FedAvgCNN | 11.950000 | 0.675000 | 2.930000 | 2.930000 |
| pair_broken_h07 | 9 | ResNet18 | 42.150000 | 0.050000 | 8.470000 | 8.470000 |
| native_control | 0 | FedAvgCNN | 28.500000 | 0.000000 | 5.700000 | 5.700000 |
| native_control | 1 | ResNet18 | 49.900001 | 0.000000 | 9.980000 | 9.980000 |
| native_control | 2 | FedAvgCNN | 24.699999 | 0.000000 | 4.940000 | 4.940000 |
| native_control | 3 | ResNet18 | 47.450000 | 0.012500 | 9.500000 | 9.500000 |
| native_control | 4 | FedAvgCNN | 26.350001 | 0.000000 | 5.270000 | 5.270000 |
| native_control | 5 | ResNet18 | 41.949999 | 0.012500 | 8.400000 | 8.400001 |
| native_control | 6 | FedAvgCNN | 21.150000 | 0.000000 | 4.230000 | 4.230000 |
| native_control | 7 | ResNet18 | 45.449999 | 0.000000 | 9.090000 | 9.090000 |
| native_control | 8 | FedAvgCNN | 22.849999 | 0.000000 | 4.570000 | 4.570000 |
| native_control | 9 | ResNet18 | 42.199999 | 0.000000 | 8.440000 | 8.440000 |

Mean centered alignment residual after transform, grouped over non-reference clients (client0 identity excluded):
FedAvgCNN: {"paired": 74.21717499042956, "broken": 223.19533683418518}
ResNet18: {"paired": 309.8016869604895, "broken": 370.45949613167863}

Residual magnitudes depend on feature scale; this grouping is descriptive, not proof of a failure mechanism. Full per-client residuals/orthogonality, classwise counts and histograms remain in final.json. The frozen overallgate is unchanged; groupedmetrics are not selected as alternativegates. Same512Dpayloads asH12; modelparametercounts differ, with no modelweight exchange in these prototype/head baselines. One mixedseed only; do not claim multi-seed architectureheterogeneity yet.
