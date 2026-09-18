# H13-A CIFAR100 seed0 mixed-backbone falsifier

| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---:|---:|---:|---:|---:|
| local | 38.040000 | 0.000000 | 7.608000 | 7.608000 | 100 |
| fedproto | 35.995000 | 0.001250 | 7.200000 | 7.200000 | 100 |
| fedgh | 31.435000 | 0.000000 | 6.287000 | 6.287000 | 100 |
| paired_h07 | 27.060000 | 6.380000 | 10.516000 | 10.516000 | 100 |
| pair_broken_h07 | 28.955000 | 0.400000 | 6.111000 | 6.111000 | 100 |
| native_control | 35.970000 | 0.000000 | 7.194000 | 7.194000 | 100 |

Frozen verdict: STRONG.
Missing paired-minus-broken: 5.980000 pp; paired-minus-native: 6.380000 pp; all gain vs best FedProto/FedGH: 3.316000 pp.
Gates: {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}

Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.
Ownership classsets SHA256: d14fbe0d88f0f0bb958d201aa117d704c360ccce70219ee880f56a8488d3e652
Ownership order SHA256: 39aa9e4178ffc4ac098e636d5bb7a93c45d1adc32813d7adfb29d29d83532271
Anchor SHA256: 5ef034047d3bb6532a854912e7eccfc876d52d15053b0465d9f42741199983b4
Split-file SHA256: a843a6d67be6cf1ced31ff3e344c42c6d7be27c8c93e163a5b242b0df6bcf7a0
Classsets: [[3, 10, 11, 13, 23, 37, 41, 42, 44, 52, 60, 67, 70, 73, 78, 81, 94, 95, 97, 99], [3, 7, 10, 14, 22, 23, 35, 37, 42, 44, 52, 54, 62, 68, 76, 84, 91, 94, 95, 99], [1, 6, 7, 12, 14, 20, 22, 28, 33, 35, 54, 58, 62, 63, 68, 76, 82, 84, 86, 91], [1, 4, 6, 8, 12, 19, 20, 28, 30, 33, 36, 45, 46, 58, 63, 72, 75, 82, 86, 96], [4, 8, 19, 30, 32, 36, 39, 40, 43, 45, 46, 53, 61, 72, 75, 79, 80, 90, 93, 96], [0, 9, 15, 16, 21, 29, 32, 39, 40, 43, 53, 55, 61, 64, 69, 79, 80, 90, 92, 93], [0, 9, 15, 16, 17, 21, 26, 29, 34, 38, 48, 49, 51, 55, 59, 64, 66, 69, 71, 92], [2, 17, 18, 24, 25, 26, 34, 38, 48, 49, 51, 57, 59, 65, 66, 71, 74, 83, 85, 89], [2, 5, 18, 24, 25, 27, 31, 47, 50, 56, 57, 65, 74, 77, 83, 85, 87, 88, 89, 98], [5, 11, 13, 27, 31, 41, 47, 50, 56, 60, 67, 70, 73, 77, 78, 81, 87, 88, 97, 98]]

Per-client predicted-class coverage: {"local": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "fedproto": [20, 27, 20, 27, 20, 23, 20, 23, 20, 30], "fedgh": [10, 28, 12, 31, 10, 27, 14, 25, 10, 29], "paired_h07": [96, 99, 90, 100, 90, 99, 93, 96, 86, 97], "pair_broken_h07": [80, 60, 72, 63, 63, 58, 67, 63, 62, 68], "native_control": [19, 26, 17, 33, 19, 27, 20, 23, 20, 27]}
Runtime/steps: {"local": {"elapsed_seconds": 248.34007620811462, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedproto": {"elapsed_seconds": 271.66890048980713, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}, "fedgh": {"elapsed_seconds": 474.4915511608124, "optimizer_steps_total": 15600, "steps_per_client_round": [156, 156, 156, 156, 156, 156, 156, 156, 156, 156]}}
Readout diagnostic seconds: 172.79538798332214
Communication: {"semantic_uplink_bytes": 412800, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 204800, "global_vectors_downlink_total": 2048000, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..99; no separate IDs transmitted", "learned_head_downlink_per_client": 205200, "learned_head_downlink_total": 2052000, "naive_affine_downlink_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672], "naive_affine_downlink_total": 10526720, "includes_redundant_identity_reference": true}
Forward examples paired/native: {"anchor_per_client": [256, 256, 256, 256, 256, 256, 256, 256, 256, 256], "prototype_refresh_per_client": [4982, 4972, 4964, 4976, 4977, 4978, 4969, 4974, 4979, 4973], "pprtp_total": 52304, "matched_native_extra_refresh_total": 49744}
Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 315.083127 | 384.632227 |
| 2 | 75.361078 | 237.724492 |
| 3 | 314.418980 | 375.324246 |
| 4 | 77.841657 | 229.433975 |
| 5 | 311.925232 | 370.099720 |
| 6 | 77.540008 | 222.978839 |
| 7 | 311.889332 | 372.394454 |
| 8 | 84.426244 | 223.786256 |
| 9 | 306.122366 | 370.058573 |

All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.
Common baseline readouts (diagnostic only):
local: {"head": {"seen": 0.38039999902248384, "missing": 0.0, "all": 0.07607999928295613, "macro": 0.0760800015181303}, "cosine": {"seen": 0.33960000425577164, "missing": 7.500000210711733e-05, "all": 0.06797999963164329, "macro": 0.06798000261187553}, "l2": {"seen": 0.34704999774694445, "missing": 8.750000124564394e-05, "all": 0.06948000006377697, "macro": 0.06948000080883503}}
fedproto: {"head": {"seen": 0.3850999981164932, "missing": 0.0, "all": 0.0770200002938509, "macro": 0.07701999954879284}, "cosine": {"seen": 0.35274999886751174, "missing": 1.2500000593718141e-05, "all": 0.07056000009179116, "macro": 0.07056000158190727}, "l2": {"seen": 0.35994999557733537, "missing": 1.2500000593718141e-05, "all": 0.07199999876320362, "macro": 0.07199999727308751}}
fedgh: {"cosine": {"seen": 0.35029999613761903, "missing": 0.0, "all": 0.07006000131368637, "macro": 0.07006000056862831}, "l2": {"seen": 0.35434999912977216, "missing": 1.2500000593718141e-05, "all": 0.0708800010383129, "macro": 0.0708799984306097}, "local_head_pre_server": {"seen": 0.3658499985933304, "missing": 0.0, "all": 0.0731699999421835, "macro": 0.07316999956965446}, "global_head_post_server": {"seen": 0.3143499970436096, "missing": 0.0, "all": 0.06287000030279159, "macro": 0.06286999955773354}}

One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.

## Architecture initialization and grouped results

Models constructed before upstream client initialization; no pretrained weights. All state hashes include BatchNorm buffers. Client0 FedAvgCNN is the unchanged reference; server head starts from client0 initialhead. All heads512->100. Same clientinitialhashes and actualround1batchhashes acrossarms.

| Client | Architecture | Parameters | Initial model SHA256 | Initial base SHA256 | Initial head SHA256 |
|---|---|---:|---|---|---|
| 0 | FedAvgCNN | 924708 | 45f199c1b832adf0f877a365c0d96fca9521efbd03ed5f13cc09ce49bfe321e0 | 80bd5bd6620001c6f5ffca3d0344f6554df5e57816a71e902832446b07f43cc1 | 6c4b4bd31711c84bc4e12bcea29d8e30f5865c9d6dfdd3e1dbfe3028ef8c8f8e |
| 1 | ResNet18 | 11227812 | d4118e6a382d25b388c46f6230a7daec4bb77b1fe762d4a3705583c38d6aa700 | a249b195d78d3f1fdd9265955ed769758338c2f033388fdda73b5b10ebedbe54 | 1d705c1280365980f2b4c286c04b854b9b892a40b2bfe691897c8aad916775be |
| 2 | FedAvgCNN | 924708 | 73071d36ce30e57fbd4842a1ce1aa84390a95534ca8f0190ac24162d20dac716 | 94777f85781393465c3f3f3a95df236b0522949be9056d95329147843019d5ef | b4c175a3f80cbb23d96eeaca350ff3e9cf2178279c555c53f325e2b8b4454746 |
| 3 | ResNet18 | 11227812 | dd398c654bc98c3c5c45caecb3c73b3eb0b1556ddea4854210457c8f53eb0488 | 30d20cc1e98b8790db1d4cf5f027022407b811bdd0f9e0ca08475db5029edbe6 | 150b50ad00dafc11e960e04debd3f8ac30825468442a9bd8af3489e978483e32 |
| 4 | FedAvgCNN | 924708 | bc7d9f72a92ccd56b448b413f92bba9afc3d074a91daaba2ad3120f8b656075b | 68ae0c86bd219fa948d729d524761941132772ce239450ba282d83307b7d70dc | 514199d38910e0decf7c6418a20446c1348d34fb7bcf9a25e85fd596cde33bf8 |
| 5 | ResNet18 | 11227812 | 095d3cb9236b05b85c0fdb7b26ef5bf6f58e7cfdbf582063d4f23365d27392d0 | 993257f0cbeab998a47ea2686fc509810e0bc73f5cdd0a2ff34027445b9b96dc | f3c6a7aa5c3455c43242c19ad43aa827e24822c0b27f74bc452864401eff4065 |
| 6 | FedAvgCNN | 924708 | d59c9cba583b6c50a792ab017acec02874f1cc749cddea827d864490826f698b | 330049c2d1e985a5f3714de5ae9d91cb8c16aa0d7fde34f530bc2d31a1830276 | ff22a863fde8f21767b8ebd857d814aa7a6bc43fe46248b9d10786a7130ef484 |
| 7 | ResNet18 | 11227812 | a4e4831f25e09b1c8e73227d733f1a4f8914b6c871527c180bffecd6998eaac5 | e76a10a1b02b4e6f506fe4f653b0bb2db8a54f1369afd4d718fdb90da6c69c37 | 6760558e6674ceee498be1d8703827b252134cc16c7ec4afd67da0991d7b73fc |
| 8 | FedAvgCNN | 924708 | f1c1089f07ee734997fdc1da3c1e6b06d3f8dc0a32ee8475519c291a7d199404 | 3b7bf88080680a1b427f4587ba5f2b5988827ba6594745c37258e03aa310ed80 | 6944dc11219f28b80c3c0090eb403e04b12dde6146369568c918a78bfdd5a888 |
| 9 | ResNet18 | 11227812 | d2603eae8404faa507c66421dc1cd21d0205dda79906a759ba36dc7d8b67d200 | c8580a166041aa6d810a5e815b5075c73ba754270a5311d0cbd6f04c91b754fc | adcb38c6a1c75deb50c0a3f9fe4ffe79b9bf34f2816bb1333cdb29cc957e7380 |

| Arm | Backbone | Seen % | Missing % | All % | Macro % |
|---|---|---:|---:|---:|---:|
| local | FedAvgCNN | 31.600000 | 0.000000 | 6.320000 | 6.320000 |
| local | ResNet18 | 44.480000 | 0.000000 | 8.896000 | 8.896000 |
| fedproto | FedAvgCNN | 29.050000 | 0.000000 | 5.810000 | 5.810000 |
| fedproto | ResNet18 | 42.939999 | 0.002500 | 8.590000 | 8.590000 |
| fedgh | FedAvgCNN | 17.980000 | 0.000000 | 3.596000 | 3.596000 |
| fedgh | ResNet18 | 44.890000 | 0.000000 | 8.978000 | 8.978000 |
| paired_h07 | FedAvgCNN | 13.500000 | 7.982500 | 9.086000 | 9.086000 |
| paired_h07 | ResNet18 | 40.620000 | 4.777500 | 11.946000 | 11.946000 |
| pair_broken_h07 | FedAvgCNN | 11.700000 | 0.770000 | 2.956000 | 2.956000 |
| pair_broken_h07 | ResNet18 | 46.210000 | 0.030000 | 9.266000 | 9.266000 |
| native_control | FedAvgCNN | 25.830000 | 0.000000 | 5.166000 | 5.166000 |
| native_control | ResNet18 | 46.110000 | 0.000000 | 9.222000 | 9.221999 |

| Arm | Client | Backbone | Seen % | Missing % | All % | Macro % |
|---|---:|---|---:|---:|---:|---:|
| local | 0 | FedAvgCNN | 37.750000 | 0.000000 | 7.550000 | 7.550000 |
| local | 1 | ResNet18 | 48.600000 | 0.000000 | 9.720000 | 9.719999 |
| local | 2 | FedAvgCNN | 24.050000 | 0.000000 | 4.810000 | 4.810001 |
| local | 3 | ResNet18 | 47.549999 | 0.000000 | 9.510000 | 9.510000 |
| local | 4 | FedAvgCNN | 32.550001 | 0.000000 | 6.510000 | 6.510000 |
| local | 5 | ResNet18 | 42.199999 | 0.000000 | 8.440000 | 8.440000 |
| local | 6 | FedAvgCNN | 31.500000 | 0.000000 | 6.300000 | 6.300000 |
| local | 7 | ResNet18 | 41.299999 | 0.000000 | 8.260000 | 8.260000 |
| local | 8 | FedAvgCNN | 32.150000 | 0.000000 | 6.430000 | 6.430000 |
| local | 9 | ResNet18 | 42.750001 | 0.000000 | 8.550000 | 8.550001 |
| fedproto | 0 | FedAvgCNN | 33.649999 | 0.000000 | 6.730000 | 6.729999 |
| fedproto | 1 | ResNet18 | 45.699999 | 0.012500 | 9.150000 | 9.150000 |
| fedproto | 2 | FedAvgCNN | 31.000000 | 0.000000 | 6.200000 | 6.200000 |
| fedproto | 3 | ResNet18 | 44.900000 | 0.000000 | 8.980000 | 8.979999 |
| fedproto | 4 | FedAvgCNN | 27.599999 | 0.000000 | 5.520000 | 5.520000 |
| fedproto | 5 | ResNet18 | 40.849999 | 0.000000 | 8.170000 | 8.170000 |
| fedproto | 6 | FedAvgCNN | 28.150001 | 0.000000 | 5.630000 | 5.630000 |
| fedproto | 7 | ResNet18 | 40.849999 | 0.000000 | 8.170000 | 8.170000 |
| fedproto | 8 | FedAvgCNN | 24.850000 | 0.000000 | 4.970000 | 4.970000 |
| fedproto | 9 | ResNet18 | 42.399999 | 0.000000 | 8.480000 | 8.480000 |
| fedgh | 0 | FedAvgCNN | 20.299999 | 0.000000 | 4.060000 | 4.059999 |
| fedgh | 1 | ResNet18 | 47.350001 | 0.000000 | 9.470000 | 9.470000 |
| fedgh | 2 | FedAvgCNN | 19.400001 | 0.000000 | 3.880000 | 3.880000 |
| fedgh | 3 | ResNet18 | 47.799999 | 0.000000 | 9.560000 | 9.559999 |
| fedgh | 4 | FedAvgCNN | 17.649999 | 0.000000 | 3.530000 | 3.530000 |
| fedgh | 5 | ResNet18 | 44.450000 | 0.000000 | 8.890000 | 8.889999 |
| fedgh | 6 | FedAvgCNN | 21.200000 | 0.000000 | 4.240000 | 4.240000 |
| fedgh | 7 | ResNet18 | 42.699999 | 0.000000 | 8.540000 | 8.540001 |
| fedgh | 8 | FedAvgCNN | 11.350000 | 0.000000 | 2.270000 | 2.270000 |
| fedgh | 9 | ResNet18 | 42.150000 | 0.000000 | 8.430000 | 8.430000 |
| paired_h07 | 0 | FedAvgCNN | 17.700000 | 8.775000 | 10.560000 | 10.559999 |
| paired_h07 | 1 | ResNet18 | 43.099999 | 4.375000 | 12.120000 | 12.120000 |
| paired_h07 | 2 | FedAvgCNN | 12.450000 | 8.375000 | 9.190000 | 9.190001 |
| paired_h07 | 3 | ResNet18 | 44.499999 | 4.487500 | 12.490000 | 12.490001 |
| paired_h07 | 4 | FedAvgCNN | 13.050000 | 6.937500 | 8.160000 | 8.160000 |
| paired_h07 | 5 | ResNet18 | 38.000000 | 4.837500 | 11.470000 | 11.470001 |
| paired_h07 | 6 | FedAvgCNN | 10.950000 | 8.162500 | 8.720000 | 8.720000 |
| paired_h07 | 7 | ResNet18 | 39.250001 | 5.150000 | 11.970000 | 11.970000 |
| paired_h07 | 8 | FedAvgCNN | 13.349999 | 7.662500 | 8.800000 | 8.799999 |
| paired_h07 | 9 | ResNet18 | 38.249999 | 5.037500 | 11.680000 | 11.680000 |
| pair_broken_h07 | 0 | FedAvgCNN | 9.750000 | 1.037500 | 2.780000 | 2.780000 |
| pair_broken_h07 | 1 | ResNet18 | 47.600001 | 0.012500 | 9.530000 | 9.530000 |
| pair_broken_h07 | 2 | FedAvgCNN | 13.450000 | 0.850000 | 3.370000 | 3.370000 |
| pair_broken_h07 | 3 | ResNet18 | 49.599999 | 0.050000 | 9.960000 | 9.960000 |
| pair_broken_h07 | 4 | FedAvgCNN | 13.249999 | 0.625000 | 3.150000 | 3.150000 |
| pair_broken_h07 | 5 | ResNet18 | 44.749999 | 0.000000 | 8.950000 | 8.950000 |
| pair_broken_h07 | 6 | FedAvgCNN | 10.400000 | 0.937500 | 2.830000 | 2.830000 |
| pair_broken_h07 | 7 | ResNet18 | 44.900000 | 0.025000 | 9.000000 | 9.000000 |
| pair_broken_h07 | 8 | FedAvgCNN | 11.650000 | 0.400000 | 2.650000 | 2.650000 |
| pair_broken_h07 | 9 | ResNet18 | 44.200000 | 0.062500 | 8.890000 | 8.890000 |
| native_control | 0 | FedAvgCNN | 27.550000 | 0.000000 | 5.510000 | 5.510000 |
| native_control | 1 | ResNet18 | 48.899999 | 0.000000 | 9.780000 | 9.779999 |
| native_control | 2 | FedAvgCNN | 23.800001 | 0.000000 | 4.760000 | 4.760000 |
| native_control | 3 | ResNet18 | 47.650000 | 0.000000 | 9.530000 | 9.530000 |
| native_control | 4 | FedAvgCNN | 21.850000 | 0.000000 | 4.370000 | 4.370000 |
| native_control | 5 | ResNet18 | 44.299999 | 0.000000 | 8.860000 | 8.859999 |
| native_control | 6 | FedAvgCNN | 28.349999 | 0.000000 | 5.670000 | 5.670000 |
| native_control | 7 | ResNet18 | 45.500001 | 0.000000 | 9.100000 | 9.099999 |
| native_control | 8 | FedAvgCNN | 27.599999 | 0.000000 | 5.520000 | 5.520000 |
| native_control | 9 | ResNet18 | 44.200000 | 0.000000 | 8.840000 | 8.840000 |

Mean centered alignment residual after transform, grouped over non-reference clients (client0 identity excluded):
FedAvgCNN: {"paired": 78.79224672990044, "broken": 228.48089045041618}
ResNet18: {"paired": 311.88780738806156, "broken": 374.5018438951057}

Residual magnitudes depend on feature scale; this grouping is descriptive, not proof of a failure mechanism. Full per-client residuals/orthogonality, classwise counts and histograms remain in final.json. The frozen overallgate is unchanged; groupedmetrics are not selected as alternativegates. Same512Dpayloads asH12; modelparametercounts differ, with no modelweight exchange in these prototype/head baselines. One mixedseed only; do not claim multi-seed architectureheterogeneity yet.
