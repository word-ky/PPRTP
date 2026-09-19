# H13-B CIFAR100 mixed-backbone stochastic replication

| Seed | Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | local | 38.040000 | 0.000000 | 7.608000 | 7.608000 | 100 |
| 0 | fedproto | 35.995000 | 0.001250 | 7.200000 | 7.200000 | 100 |
| 0 | fedgh | 31.435000 | 0.000000 | 6.287000 | 6.287000 | 100 |
| 0 | paired_h07 | 27.060000 | 6.380000 | 10.516000 | 10.516000 | 100 |
| 0 | pair_broken_h07 | 28.955000 | 0.400000 | 6.111000 | 6.111000 | 100 |
| 0 | native_control | 35.970000 | 0.000000 | 7.194000 | 7.194000 | 100 |
| 1 | local | 39.100000 | 0.000000 | 7.820000 | 7.820000 | 100 |
| 1 | fedproto | 36.140001 | 0.001250 | 7.229000 | 7.229000 | 100 |
| 1 | fedgh | 30.590000 | 0.003750 | 6.121000 | 6.121000 | 100 |
| 1 | paired_h07 | 27.565000 | 6.383750 | 10.620000 | 10.620000 | 100 |
| 1 | pair_broken_h07 | 29.270000 | 0.363750 | 6.145000 | 6.145000 | 100 |
| 1 | native_control | 35.050000 | 0.002500 | 7.012000 | 7.012000 | 100 |
| 2 | local | 39.645000 | 0.000000 | 7.929000 | 7.929000 | 100 |
| 2 | fedproto | 38.185000 | 0.000000 | 7.637000 | 7.637000 | 100 |
| 2 | fedgh | 30.305000 | 0.002500 | 6.063000 | 6.063000 | 100 |
| 2 | paired_h07 | 27.555000 | 6.363750 | 10.602000 | 10.602000 | 100 |
| 2 | pair_broken_h07 | 28.655000 | 0.406250 | 6.056000 | 6.056000 | 100 |
| 2 | native_control | 34.615000 | 0.001250 | 6.924000 | 6.924000 | 100 |

Mean +/- sample SD (n=3,ddof=1), percentage points:

| Arm | Seen | Missing | All |
|---|---:|---:|---:|
| local | 38.928333 +/- 0.816154 | 0.000000 +/- 0.000000 | 7.785667 +/- 0.163231 |
| fedproto | 36.773333 +/- 1.224687 | 0.000833 +/- 0.000722 | 7.355333 +/- 0.244361 |
| fedgh | 30.776667 +/- 0.587672 | 0.002083 +/- 0.001909 | 6.157000 +/- 0.116258 |
| paired_h07 | 27.393333 +/- 0.288718 | 6.375833 +/- 0.010631 | 10.579333 +/- 0.055582 |
| pair_broken_h07 | 28.960000 +/- 0.307530 | 0.390000 +/- 0.022947 | 6.104000 +/- 0.044911 |
| native_control | 35.211667 +/- 0.691815 | 0.001250 +/- 0.001250 | 7.043333 +/- 0.137700 |

Frozen verdict: 3/3 STRONG: accept fixed-assignment stochastic mixed-backbone portability.
Seedwise strong: [True, True, True].
Seedwise frozen gate values: {"0": {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}, "1": {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}, "2": {"missing_at_least5": true, "native_gap_at_least4": true, "broken_gap_at_least3": true, "all_gain_at_least1": true}}
Seedwise gap values (pp): {"missing_gap_pp": [5.979999995179242, 6.019999988930066, 5.957499989162899], "native_gap_pp": [6.38000000268221, 6.381249991536605, 6.3624999883177225], "all_gap_pp": [3.316000141203404, 3.391000106930733, 2.9650001227855673]}
Gap mean +/- sampleSD (pp): {"missing_gap_pp": {"mean": 5.985833324424069, "std": 0.03165569917539177}, "native_gap_pp": {"mean": 6.3745833275121795, "std": 0.010483126089113348}, "all_gap_pp": {"mean": 3.2240001236399016, "std": 0.22741371780674083}}

## Backbone-group qualifications

| Seed | Backbone | Readout | Seen % | Missing % | All % | Group paired-minus-broken missing pp |
|---|---|---|---:|---:|---:|---:|
| 0 | FedAvgCNN | paired_h07 | 13.500000 | 7.982500 | 9.086000 | 7.212500 |
| 0 | FedAvgCNN | pair_broken_h07 | 11.700000 | 0.770000 | 2.956000 | 7.212500 |
| 0 | FedAvgCNN | native_control | 25.830000 | 0.000000 | 5.166000 | 7.212500 |
| 0 | ResNet18 | paired_h07 | 40.620000 | 4.777500 | 11.946000 | 4.747500 |
| 0 | ResNet18 | pair_broken_h07 | 46.210000 | 0.030000 | 9.266000 | 4.747500 |
| 0 | ResNet18 | native_control | 46.110000 | 0.000000 | 9.222000 | 4.747500 |
| 1 | FedAvgCNN | paired_h07 | 14.410000 | 7.990000 | 9.274000 | 7.280000 |
| 1 | FedAvgCNN | pair_broken_h07 | 12.500000 | 0.710000 | 3.068000 | 7.280000 |
| 1 | FedAvgCNN | native_control | 24.710000 | 0.000000 | 4.942000 | 7.280000 |
| 1 | ResNet18 | paired_h07 | 40.719999 | 4.777500 | 11.966000 | 4.760000 |
| 1 | ResNet18 | pair_broken_h07 | 46.040000 | 0.017500 | 9.222000 | 4.760000 |
| 1 | ResNet18 | native_control | 45.390000 | 0.005000 | 9.082000 | 4.760000 |
| 2 | FedAvgCNN | paired_h07 | 14.570000 | 7.755000 | 9.118000 | 6.970000 |
| 2 | FedAvgCNN | pair_broken_h07 | 11.430000 | 0.785000 | 2.914000 | 6.970000 |
| 2 | FedAvgCNN | native_control | 23.890000 | 0.000000 | 4.778000 | 6.970000 |
| 2 | ResNet18 | paired_h07 | 40.540000 | 4.972500 | 12.086000 | 4.945000 |
| 2 | ResNet18 | pair_broken_h07 | 45.880000 | 0.027500 | 9.198000 | 4.945000 |
| 2 | ResNet18 | native_control | 45.340000 | 0.002500 | 9.070000 | 4.945000 |

Groupwise causal gaps seed0/1/2 (pp): {"FedAvgCNN": [7.212500032037496, 7.279999945312738, 6.969999996945263], "ResNet18": [4.7474999583209865, 4.760000032547396, 4.944999981380533]}
Positive paired-minus-broken missing gap for both families in ALL3seeds: True. This is a claim-qualification diagnostic, not an additional or replacement gate.

Exact H13-A split/ownership/anchors preserved. Per-client initialmodelhashes and actualround1batch orders differ acrossall3seeds; withinseed all3arms pair exactmodel/base/head init and actualbatch orders. Architecture assignment remains alternatingCNN/ResNet18,all512D/head100,oneownerfromeachfamily perclass.
Initialization details: [[{"client": 0, "architecture": "FedAvgCNN", "initial_model_hash": "45f199c1b832adf0f877a365c0d96fca9521efbd03ed5f13cc09ce49bfe321e0"}, {"client": 1, "architecture": "ResNet18", "initial_model_hash": "d4118e6a382d25b388c46f6230a7daec4bb77b1fe762d4a3705583c38d6aa700"}, {"client": 2, "architecture": "FedAvgCNN", "initial_model_hash": "73071d36ce30e57fbd4842a1ce1aa84390a95534ca8f0190ac24162d20dac716"}, {"client": 3, "architecture": "ResNet18", "initial_model_hash": "dd398c654bc98c3c5c45caecb3c73b3eb0b1556ddea4854210457c8f53eb0488"}, {"client": 4, "architecture": "FedAvgCNN", "initial_model_hash": "bc7d9f72a92ccd56b448b413f92bba9afc3d074a91daaba2ad3120f8b656075b"}, {"client": 5, "architecture": "ResNet18", "initial_model_hash": "095d3cb9236b05b85c0fdb7b26ef5bf6f58e7cfdbf582063d4f23365d27392d0"}, {"client": 6, "architecture": "FedAvgCNN", "initial_model_hash": "d59c9cba583b6c50a792ab017acec02874f1cc749cddea827d864490826f698b"}, {"client": 7, "architecture": "ResNet18", "initial_model_hash": "a4e4831f25e09b1c8e73227d733f1a4f8914b6c871527c180bffecd6998eaac5"}, {"client": 8, "architecture": "FedAvgCNN", "initial_model_hash": "f1c1089f07ee734997fdc1da3c1e6b06d3f8dc0a32ee8475519c291a7d199404"}, {"client": 9, "architecture": "ResNet18", "initial_model_hash": "d2603eae8404faa507c66421dc1cd21d0205dda79906a759ba36dc7d8b67d200"}], [{"client": 0, "architecture": "FedAvgCNN", "initial_model_hash": "953f01f3b69eb39d67d6c42d5fa50b4a30d4383541bb8e10f67f4fe92c623070"}, {"client": 1, "architecture": "ResNet18", "initial_model_hash": "bf898b0e89f8d2d1cbfea169086b9b06f6841822f9d8a804bb8dcb05b35ebf98"}, {"client": 2, "architecture": "FedAvgCNN", "initial_model_hash": "81cd60be15bb9c5936675a4b6e7bf3e1d4d8d4e5a42e6d074c7f0f0d0450cd57"}, {"client": 3, "architecture": "ResNet18", "initial_model_hash": "4d92966ebc2ddbbdd6f4c0a18efef0c397044155459797800c8520949547e42d"}, {"client": 4, "architecture": "FedAvgCNN", "initial_model_hash": "c64afa49d6d1ea66452371a21f06cd7f8417f1df882755cbe53fbca028f48a5a"}, {"client": 5, "architecture": "ResNet18", "initial_model_hash": "f2c6a46427e8d89a91b6151d63aac1ec8ccba948787fb080dc2ea01f570e301d"}, {"client": 6, "architecture": "FedAvgCNN", "initial_model_hash": "71d9e3e2b5a9898fc42d24388d68a961dc9a6bb0207e2a97055f18b03049745f"}, {"client": 7, "architecture": "ResNet18", "initial_model_hash": "87b0d993dfc75c4e72d9771a136b1b9f900e7bf17bd33f0b9ac14a0beec10404"}, {"client": 8, "architecture": "FedAvgCNN", "initial_model_hash": "c8af2c666759ef3bd01253d852ff961757bd94da7923456fb086433ec6dd1ba5"}, {"client": 9, "architecture": "ResNet18", "initial_model_hash": "bd39f161b1c6294265671c4da86c0d2b6cfe2cdb5a1e64cd4028fe42d6a73eff"}], [{"client": 0, "architecture": "FedAvgCNN", "initial_model_hash": "6e60ed1a1a7ceeb4ecbf0d9335264b394d66d1407f9475c151e1053995efbec1"}, {"client": 1, "architecture": "ResNet18", "initial_model_hash": "4eee5c52376a55d62f6b4510edf274e84fd7f319e2726c4aafd2980f5c856bd3"}, {"client": 2, "architecture": "FedAvgCNN", "initial_model_hash": "cd22291fdae411479607828c19e3e278d383d0a0eaa962a92fea1cc0733a2261"}, {"client": 3, "architecture": "ResNet18", "initial_model_hash": "e4d97d9351834e5d7bb2757e42f34d4826c4bd232aeccf9098a95d7584e55c66"}, {"client": 4, "architecture": "FedAvgCNN", "initial_model_hash": "3495b24f79e80f238868600c769a4faa7379427deadb7e4da4bf2102c0b028c6"}, {"client": 5, "architecture": "ResNet18", "initial_model_hash": "2329c0751cc8eb53036534f7ec1557785940cff1a07793efccb70274cab06f5a"}, {"client": 6, "architecture": "FedAvgCNN", "initial_model_hash": "573541dfcfb08de2d6b3929bbd2c2e3f4aa0fa1f2e61037f875312ea9a4b39dc"}, {"client": 7, "architecture": "ResNet18", "initial_model_hash": "1db7eebafac08a06253de8bcfb17dfb59ad56625a06ea64f35fc34cea88a06d8"}, {"client": 8, "architecture": "FedAvgCNN", "initial_model_hash": "c73b33b7d07c8a4ce95f7f838e841a79d33f40cf5ad060807b7be01e94cb3a0f"}, {"client": 9, "architecture": "ResNet18", "initial_model_hash": "0449ce995ec56a4a81a732960efcd6ed4d5bbd6be28b99098cf641e53fc17cd3"}]]

Same-family clients have distinct deterministic initialization; this tests architecture PLUS client-specific initialization heterogeneity. Stochastic replication on onefixedsplit/assignment, not ownership-graph or architecture-assignment replication. Paired/broken/native sharefinalstates/rawmeans; no tuning or solverchoice. Seenaccuracy tradeoff, extra correspondence sideinformation and post-hoc nature remain. Rawresidual scale is not normalized; do not use its magnitude alone as a failurediagnosis. Seed0 reused fromH13A,not rerun. Fullperseed/perclient/backbone results,residuals,steps,payloads,hashes andwarnings accompanythisreport.
