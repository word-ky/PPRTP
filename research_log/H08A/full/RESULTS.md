# H08-A lag-1 aligned-GPC causal gate

| Arm | Round | Seen % | Missing % | All % | Macro % | Predicted classes |
|---|---:|---:|---:|---:|---:|---:|
| fedgh_posthoc_reference | 10 | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| pprtp_all_lag1 | 2 | 31.150000 | 30.425000 | 30.570000 | 30.569999 | 10 |
| pprtp_all_lag1 | 5 | 26.800000 | 24.450000 | 24.920000 | 24.919999 | 10 |
| pprtp_all_lag1 | 10 | 27.900000 | 22.212500 | 23.350000 | 23.349999 | 10 |
| pprtp_seen_lag1 | 2 | 31.550000 | 30.500000 | 30.710000 | 30.709999 | 10 |
| pprtp_seen_lag1 | 5 | 26.850000 | 24.425000 | 24.910000 | 24.909999 | 10 |
| pprtp_seen_lag1 | 10 | 28.000000 | 22.262500 | 23.410000 | 23.409999 | 10 |

Frozen verdict: B: online mechanism falsified at frozen strength; all-minus-seen missing=-0.050000pp, all=-0.060000pp.
Per bank: {"semantic_forward_examples": 2000, "anchor_forward_examples": 2560, "semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 5242880, "bank_bytes": 20480, "bank_downlink_total": 204800, "naive_transform_bytes_per_client": [1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672, 1052672]}
Ten builds per arm: nine training banks and one evaluation-only final bank; rounds2/5 reuse the fresh next-round bank for separate evaluation. Each arm refreshes20,000 local semantic and25,600 anchor forward examples. Naive affine transform includes both512-D means and512x512 matrix;10,526,720B total per build. This excludes image/reference distribution and unchanged FedGH traffic; no communication-efficiency claim.
All per-client/class counts, histograms, bank/transform hashes, losses and ordinary readouts are preserved in rounds.jsonl/final.json. Seen-arm missing_probability_on_seen records the counterfactual ALL-class softmax on the same tensor, not the masked training softmax (whose missing mass is zero).

pprtp_all_lag1, round 2
Ordinary readouts: {"cosine": {"seen": 0.6360000014305115, "missing": 0.00012499999720603228, "all": 0.12730000019073487, "macro": 0.12729999646544457}, "l2": {"seen": 0.6224999964237213, "missing": 0.0004999999888241291, "all": 0.12490000054240227, "macro": 0.12490000277757644}, "local_head_pre_server": {"seen": 0.5294999957084656, "missing": 0.0, "all": 0.10590000078082085, "macro": 0.10589999929070473}, "global_head_post_server": {"seen": 0.529499989748001, "missing": 0.0, "all": 0.10590000078082085, "macro": 0.10589999854564666}}
Mean losses: {"local": 2.1574703097343444, "knowledge": 2.1606847705841066}
Client0 first-batch diagnostics: {"feature_extractor_knowledge_grad_norm": 5.234928607940674, "local_grad_norm": 1.1623830795288086, "scaled_knowledge_grad_norm": 0.010469857603311539, "knowledge_local_grad_ratio": 0.009007235057651997, "missing_probability_on_seen": 0.771550178527832, "valid_mask": [true, true, true, true, true, true, true, true, true, true], "denominator_feature_gradients": {"cosine": 0.31342029571533203, "all_norm": 0.20234517753124237, "seen_norm": 0.07283154129981995, "missing_probability_mass": 0.771550178527832}, "preupdate_feature_hash": "065065d7d1709feabb4186ac133b493251f57095e66f2f9dcc2bf2edd95db4fc"}
Prediction histograms (class0..9): {"overall": [1290, 1041, 941, 377, 826, 1291, 2097, 309, 1051, 777], "seen": [244, 210, 199, 89, 161, 254, 409, 67, 209, 158], "missing": [1046, 831, 742, 288, 665, 1037, 1688, 242, 842, 619]}
Fresh bank SHA256: 641f75994e672921d13f4eb4074cd6f62e93685ff23efd6a333c9210d7e89635
Owner norm range: [1.393553376197815, 1.7727187871932983]
Global norm range: [1.411819338798523, 1.735795259475708]

pprtp_all_lag1, round 5
Ordinary readouts: {"cosine": {"seen": 0.5309999972581864, "missing": 0.0, "all": 0.10619999915361404, "macro": 0.10620000064373017}, "l2": {"seen": 0.5830000013113021, "missing": 0.0, "all": 0.11659999936819077, "macro": 0.11660000011324882}, "local_head_pre_server": {"seen": 0.5064999997615814, "missing": 0.0, "all": 0.10130000114440918, "macro": 0.10129999965429307}, "global_head_post_server": {"seen": 0.527999997138977, "missing": 0.0, "all": 0.10560000017285347, "macro": 0.10559999942779541}}
Mean losses: {"local": 1.1864572989940643, "knowledge": 2.272029369354248}
Client0 first-batch diagnostics: {"feature_extractor_knowledge_grad_norm": 1.004482626914978, "local_grad_norm": 2.626251220703125, "scaled_knowledge_grad_norm": 0.002008965238928795, "knowledge_local_grad_ratio": 0.0007649554754607379, "missing_probability_on_seen": 0.7965614199638367, "valid_mask": [true, true, true, true, true, true, true, true, true, true], "denominator_feature_gradients": {"cosine": 0.31859201192855835, "all_norm": 0.017844662070274353, "seen_norm": 0.005516957025974989, "missing_probability_mass": 0.7965614199638367}, "preupdate_feature_hash": "9fb9bac6430791be087a688bc062280ed384f46d42e3d6aa0214ac7a39437cbf"}
Prediction histograms (class0..9): {"overall": [1216, 1304, 935, 580, 1070, 1394, 1734, 330, 707, 730], "seen": [232, 245, 210, 130, 193, 297, 347, 74, 135, 137], "missing": [984, 1059, 725, 450, 877, 1097, 1387, 256, 572, 593]}
Fresh bank SHA256: 7ccff40ba575d801e525ed26d1510e76c6fdffdea571518d35eaf674159cbd17
Owner norm range: [7.277390956878662, 8.877798080444336]
Global norm range: [7.629063129425049, 8.842063903808594]

pprtp_all_lag1, round 10
Ordinary readouts: {"cosine": {"seen": 0.5314999938011169, "missing": 0.0, "all": 0.10630000159144401, "macro": 0.1063000001013279}, "l2": {"seen": 0.5604999959468842, "missing": 0.0, "all": 0.11210000067949295, "macro": 0.11210000142455101}, "local_head_pre_server": {"seen": 0.6129999995231629, "missing": 0.0, "all": 0.12260000184178352, "macro": 0.1226000003516674}, "global_head_post_server": {"seen": 0.5, "missing": 0.0, "all": 0.10000000149011612, "macro": 0.10000000149011612}}
Mean losses: {"local": 0.7713662459850312, "knowledge": 2.292378323554993}
Client0 first-batch diagnostics: {"feature_extractor_knowledge_grad_norm": 0.4530303478240967, "local_grad_norm": 5.3773345947265625, "scaled_knowledge_grad_norm": 0.0009060607408173382, "knowledge_local_grad_ratio": 0.00016849624807946384, "missing_probability_on_seen": 0.7982860803604126, "valid_mask": [true, true, true, true, true, true, true, true, true, true], "denominator_feature_gradients": {"cosine": 0.3995862305164337, "all_norm": 0.004518482368439436, "seen_norm": 0.001721168402582407, "missing_probability_mass": 0.7982860803604126}, "preupdate_feature_hash": "fa1d3baffb89d81c0e8ec762f9b6e68ae73974ee5ad55031aa821b8a60697693"}
Prediction histograms (class0..9): {"overall": [1110, 1430, 1037, 369, 1110, 1533, 1824, 206, 636, 745], "seen": [234, 267, 229, 83, 212, 333, 340, 47, 114, 141], "missing": [876, 1163, 808, 286, 898, 1200, 1484, 159, 522, 604]}
Fresh bank SHA256: 89740ec4f88e23b80e3130bfa0e5e979ec80ffbcddab6659719d763b1572691f
Owner norm range: [11.631319999694824, 14.9199857711792]
Global norm range: [12.41988754272461, 14.769025802612305]

pprtp_seen_lag1, round 2
Ordinary readouts: {"cosine": {"seen": 0.6350000023841857, "missing": 0.00012499999720603228, "all": 0.12710000053048134, "macro": 0.12709999606013297}, "l2": {"seen": 0.6224999964237213, "missing": 0.0004999999888241291, "all": 0.12490000054240227, "macro": 0.12490000277757644}, "local_head_pre_server": {"seen": 0.5284999966621399, "missing": 0.0, "all": 0.10570000112056732, "macro": 0.10569999888539314}, "global_head_post_server": {"seen": 0.529499989748001, "missing": 0.0, "all": 0.1059000015258789, "macro": 0.10589999854564666}}
Mean losses: {"local": 2.157491027832031, "knowledge": 0.654034467458725}
Client0 first-batch diagnostics: {"feature_extractor_knowledge_grad_norm": 0.23335465788841248, "local_grad_norm": 1.1623830795288086, "scaled_knowledge_grad_norm": 0.0004667093453463167, "knowledge_local_grad_ratio": 0.00040151079883798957, "missing_probability_on_seen": 0.771550178527832, "valid_mask": [true, true, true, true, true, true, true, true, true, true], "denominator_feature_gradients": {"cosine": 0.31342029571533203, "all_norm": 0.20234517753124237, "seen_norm": 0.07283154129981995, "missing_probability_mass": 0.771550178527832}, "preupdate_feature_hash": "065065d7d1709feabb4186ac133b493251f57095e66f2f9dcc2bf2edd95db4fc"}
Prediction histograms (class0..9): {"overall": [1293, 1041, 946, 372, 845, 1292, 2081, 307, 1048, 775], "seen": [245, 210, 200, 87, 168, 255, 402, 65, 210, 158], "missing": [1048, 831, 746, 285, 677, 1037, 1679, 242, 838, 617]}
Fresh bank SHA256: 2597c469d24f0238ba2763674d1069174c807eb2a294c6881df79b28fc9d5a22
Owner norm range: [1.3939158916473389, 1.7733299732208252]
Global norm range: [1.411752700805664, 1.7363767623901367]

pprtp_seen_lag1, round 5
Ordinary readouts: {"cosine": {"seen": 0.5299999982118606, "missing": 0.0, "all": 0.10600000023841857, "macro": 0.10600000023841857}, "l2": {"seen": 0.5830000013113021, "missing": 0.0, "all": 0.11659999936819077, "macro": 0.11660000160336495}, "local_head_pre_server": {"seen": 0.5055000007152557, "missing": 0.0, "all": 0.1011000007390976, "macro": 0.10109999999403954}, "global_head_post_server": {"seen": 0.527999997138977, "missing": 0.0, "all": 0.10560000017285347, "macro": 0.10559999942779541}}
Mean losses: {"local": 1.1868707003593444, "knowledge": 0.6875727670192718}
Client0 first-batch diagnostics: {"feature_extractor_knowledge_grad_norm": 0.0345684289932251, "local_grad_norm": 2.6252949237823486, "scaled_knowledge_grad_norm": 6.91368622938171e-05, "knowledge_local_grad_ratio": 2.6334893846069463e-05, "missing_probability_on_seen": 0.7965734004974365, "valid_mask": [true, true, true, true, true, true, true, true, true, true], "denominator_feature_gradients": {"cosine": 0.3184739053249359, "all_norm": 0.017820414155721664, "seen_norm": 0.005508308298885822, "missing_probability_mass": 0.7965734004974365}, "preupdate_feature_hash": "def0d81702418d8d0a2a06dcb1fac79888c902667871b7e5dc35046c0da84f89"}
Prediction histograms (class0..9): {"overall": [1215, 1305, 932, 579, 1068, 1396, 1735, 328, 710, 732], "seen": [232, 245, 207, 130, 193, 298, 346, 75, 137, 137], "missing": [983, 1060, 725, 449, 875, 1098, 1389, 253, 573, 595]}
Fresh bank SHA256: 935767fcab6aff5ee0d2d396f0a0d52fd471f60930b9cc85d3ff906b3c029c17
Owner norm range: [7.275350093841553, 8.87433910369873]
Global norm range: [7.626368522644043, 8.839091300964355]

pprtp_seen_lag1, round 10
Ordinary readouts: {"cosine": {"seen": 0.5314999938011169, "missing": 0.0, "all": 0.10630000159144401, "macro": 0.1063000001013279}, "l2": {"seen": 0.5604999959468842, "missing": 0.0, "all": 0.11210000067949295, "macro": 0.11210000142455101}, "local_head_pre_server": {"seen": 0.6120000004768371, "missing": 0.0, "all": 0.12240000218153, "macro": 0.12240000143647194}, "global_head_post_server": {"seen": 0.5, "missing": 0.0, "all": 0.10000000149011612, "macro": 0.10000000149011612}}
Mean losses: {"local": 0.7713138360977172, "knowledge": 0.6895285458564758}
Client0 first-batch diagnostics: {"feature_extractor_knowledge_grad_norm": 0.011020571924746037, "local_grad_norm": 5.3958306312561035, "scaled_knowledge_grad_norm": 2.2041145712137222e-05, "knowledge_local_grad_ratio": 4.084847660124069e-06, "missing_probability_on_seen": 0.7982965707778931, "valid_mask": [true, true, true, true, true, true, true, true, true, true], "denominator_feature_gradients": {"cosine": 0.4006684422492981, "all_norm": 0.004511385690420866, "seen_norm": 0.00172287761233747, "missing_probability_mass": 0.7982965707778931}, "preupdate_feature_hash": "d352c80e923450749de9fd8cc92eff55c7a818801232c8b15b4313ba0685ebbf"}
Prediction histograms (class0..9): {"overall": [1111, 1433, 1035, 371, 1111, 1531, 1823, 209, 633, 743], "seen": [232, 267, 230, 84, 212, 332, 336, 48, 117, 142], "missing": [879, 1166, 805, 287, 899, 1199, 1487, 161, 516, 601]}
Fresh bank SHA256: be0b59d1de82620e58cf3fce9b9fab59bc56109ec7fb3901b1c431f24ce75b7c
Owner norm range: [11.63077163696289, 14.918163299560547]
Global norm range: [12.418347358703613, 14.766951560974121]
