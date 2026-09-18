# H07-B ordinary-local direct readout across seeds

| Seed | Arm | Seen % | Missing % | All % | Macro % | Predicted classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | localtrain_aligned_global_prototype_cosine | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| 0 | localtrain_native_global_prototype_cosine_control | 54.099999 | 0.000000 | 10.820000 | 10.820000 | 9 |
| 1 | seed1_localtrain_aligned_global_prototype_cosine | 28.350000 | 21.737500 | 23.060000 | 23.060001 | 10 |
| 1 | seed1_localtrain_native_global_prototype_cosine_control | 61.500000 | 0.000000 | 12.300000 | 12.300000 | 10 |
| 2 | seed2_localtrain_aligned_global_prototype_cosine | 24.200000 | 20.700000 | 21.400000 | 21.400000 | 10 |
| 2 | seed2_localtrain_native_global_prototype_cosine_control | 58.650000 | 0.000000 | 11.730000 | 11.730000 | 10 |

| Seed | retM | retA | Alignment gain pp | Predicted classes |
|---|---:|---:|---:|---:|
| 1 | 1.03697075 | 0.900429532 | 21.737500 | 10 |
| 2 | 1.05950096 | 0.914920905 | 20.700000 | 10 |

Frozen verdict: A: strong cross-seed final-method replication.
Seed0 is reused from H07-A; only seeds1/2 were run. Full per-client/per-class correct/count tables, local/global prototype receipts, and per-client histograms are in each final.json.
Each aligned readout uses 41,280 B semantic uplink, 5,242,880 B anchor-feature uplink, and 204,800 B global-prototype downlink total. Anchor/reference distribution remains outside this feature accounting.

Seed 1 provenance: {"oracle_sha256": "4c972e65f2c29657b36e7bd0641a3632e390090b17a1ce2899234ceffc8e5f2a", "support_sha256": "8373dc4071d7760582de0613b98aae19620374d6f1b6b1cc01b5d5c8b9e207dd", "anchor_sha256": "5ac2570f6c34481b027322b9dd1215b877e509ea696e9156d2c429e0095d23c2", "train_sha256": "72caddd11bc26699a01ec07d447ec4443d440e5bf88cd31899d09630511dbfcd"}
N256 prefix SHA256: 55fed40eff0b3ab8ac564571d3305fd009c605df16593da5b0f3b1362c0fa469
Local train SHA256: 72caddd11bc26699a01ec07d447ec4443d440e5bf88cd31899d09630511dbfcd
Per-client train SHA256: ["2725ed69dc3eb77d22946ee0af33862345f99eb45bbee5e3a9744f463dcceab2", "4fe376d7212da363bb2263b1c2fa8cbf1821084793cf4be7b2a9d89693293c1b", "0321f5950a73301eb862397d7cc35e839be723435d4ecfda8b773cc12817dd92", "32808bb0ff101b33923e68a4c65f1e878cdd1770575de252557f782319b7baa8", "563eaaf6e8010c741be06cb7a50da0347f99572b49cb1ea919bb93bc40818d3d", "8bf5984407dc3e2dd146b9e3c4c2f7184b6d6df13f4e31a8976462a918756691", "79a7ba560fe10cdae6c323fe519e035c2b5f5e7a0ed52c3538bab41c7855d813", "abd1a33fa2ffe128e5c1aaf10e51855b29262f6e040f8127e818f958b9db9112", "023b36dc21f447e935b09e533d197b423d66f70ff4fddced193eba8d0082796a", "eee3b699e820a9baada4a336c932b9cc973c1f6b5197a68afe52c8a9038e380d"]
Class sets: [[4, 8], [4, 7], [0, 7], [0, 1], [1, 2], [2, 5], [5, 9], [6, 9], [3, 6], [3, 8]]
Transform SHA256: ["39acc9c052664d9a92b6c527ff22a805aadc76188d95d9e3f76e31cdf42df3ac", "6a062e17bcb5fd3e5212576a7adb5537cf3c549e4e88706ae3bcab8be3864e3d", "a9e30b78c602f143f04434dcbaac095979c88f6737232a228e76fb4d83973fe9", "fcc92c9ee48d079089e24abd14462a5e5b7a9144ea243b4690bdd173fb0af0d7", "48de77a2c308e59800a12b955497da2b05ba66e80504433c44579c05c15434b0", "92f7a221a262b49226aa09208b1c6204d46798aa51f00a12155ca55db7e37553", "1bf113d088c8a5dc0ea9fd4eac0c377c88b53b4e791241121ceabcfd4a90d901", "2e436adeccfb8c90f1dec55f59f77492f1bfc31f55fa77fc7e287f8caa541e92", "ba4bc825f381e85f0361480f4e9a46aab4d0f9700571dc901ce1ad5357911656", "88020cb96fe2ef15c9d96439eb3e7a711f9464b2907fc4c38a16a111e7d8aadc"]
Refresh: 2000 forward examples per arm (200/client); local compute, not communication.

seed1_localtrain_aligned_global_prototype_cosine:
Aggregate SHA256: 2d787aefd7a01b93696dfdbf70ba6576695c3bb8d618912625b2d1034e716e0b
Norm range: 13.8659906 to 16.9461403
Prediction histograms (class0..9): {"overall": [1776, 741, 854, 265, 205, 1275, 2241, 806, 930, 907], "seen": [353, 155, 185, 66, 47, 243, 422, 158, 174, 197], "missing": [1423, 586, 669, 199, 158, 1032, 1819, 648, 756, 710]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total | Max/Fro hierarchical error | SHA256 |
|---|---|---|---:|---|---|
| 0 | [2, 3] | [100, 100] | 200 | 4.76837158e-07/1.31520653e-06 | 01968c4f6af33fb2e080d0e47ef483bf9fff73abf1c1317ca77c8ceec947a978 |
| 1 | [3, 4] | [100, 100] | 200 | 4.76837158e-07/1.32006778e-06 | 7bf5c8e06e54bee0173979f332891412418c6cb9879342c29240a116a44e184a |
| 2 | [4, 5] | [100, 100] | 200 | 4.76837158e-07/1.09136238e-06 | e75d0370f24167ba53835679dde07064712687409ed4b03c473bd4c027e36ef7 |
| 3 | [8, 9] | [100, 100] | 200 | 2.38418579e-07/1.10396104e-06 | dc750adccd714a3751206b8a2ee443222c016a128cf719a5f8ca0744b987d242 |
| 4 | [0, 1] | [100, 100] | 200 | 2.38418579e-07/1.06600737e-06 | e8319c09c6502656785e47dc421b1e44fcb6f77587c1b631581be2a8a83f6ba7 |
| 5 | [5, 6] | [100, 100] | 200 | 4.76837158e-07/1.25145721e-06 | 9b4c4aa9668f1433191acc74b237003294d99a0bfc81967d158202dce5c8378a |
| 6 | [7, 8] | [100, 100] | 200 | 2.38418579e-07/1.07431197e-06 | 119fa581f4a08ca2eaa00bd5c1aeb4ae2318c6401aeb19af63edeb4f387dd951 |
| 7 | [1, 2] | [100, 100] | 200 | 4.76837158e-07/1.20226127e-06 | 0c4f71ccce1659d4517ebf3923f2ad4ef2b039f2006b270e07afce26caf7b1a1 |
| 8 | [0, 9] | [100, 100] | 200 | 2.38418579e-07/1.00214299e-06 | 355945f38dc7c846c74837d4830a1f117332d2647fdf0b2bda18a244ffc9657b |
| 9 | [6, 7] | [100, 100] | 200 | 4.76837158e-07/1.31449656e-06 | aed61bf672f21a147e0eb304bf50366c1344a6162da8d1df227a9a07de0abb41 |

seed1_localtrain_native_global_prototype_cosine_control:
Aggregate SHA256: a8cf9aee29a18b7fc9639ac86c70b42cd96b723321133888381a0d68fcfefe2c
Norm range: 11.3873825 to 13.9075871
Prediction histograms (class0..9): {"overall": [1680, 387, 723, 1647, 1892, 1854, 1155, 210, 121, 331], "seen": [356, 93, 119, 289, 360, 364, 275, 32, 50, 62], "missing": [1324, 294, 604, 1358, 1532, 1490, 880, 178, 71, 269]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 0, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total | Max/Fro hierarchical error | SHA256 |
|---|---|---|---:|---|---|
| 0 | [2, 3] | [100, 100] | 200 | 2.38418579e-07/9.05121112e-07 | ed184254112323acca415bf9e171c0e8a856798598962a82353b81efe9beed23 |
| 1 | [3, 4] | [100, 100] | 200 | 2.38418579e-07/9.61049977e-07 | 12b9348f4f6d54fd34ecf9ae730f5e81972b66f7a46d97e7660d6850e30ed96c |
| 2 | [4, 5] | [100, 100] | 200 | 2.38418579e-07/8.3780634e-07 | 4d8964cbcd5fb093435c14f543a8074d699710d4bee09949502abfd573f3ff6c |
| 3 | [8, 9] | [100, 100] | 200 | 2.38418579e-07/8.84392421e-07 | 9335348735fd4933bce59aafb16d68fed45e7444f8ed67fe8ee4d84b3a2df3dd |
| 4 | [0, 1] | [100, 100] | 200 | 3.57627869e-07/9.65169193e-07 | 5674c3560a5caedbfc9ba814877bba75ec2e71dea1f62e04766132486c7cda6c |
| 5 | [5, 6] | [100, 100] | 200 | 3.57627869e-07/1.09502844e-06 | 7ab3a354c679e21b659df65ceaf942c0ac3c6b91de33e2b1bb193569f08845c5 |
| 6 | [7, 8] | [100, 100] | 200 | 2.38418579e-07/8.6891049e-07 | a4453cbe94216042593cc1acc11dc1f3a4efbc76719a88cddb95b5cc18c5a7d1 |
| 7 | [1, 2] | [100, 100] | 200 | 2.38418579e-07/9.15584963e-07 | 76b88cad2b64cbfbffa3b604488b98ef7934fc18f2cbd3b0e55b77f18d7ccb20 |
| 8 | [0, 9] | [100, 100] | 200 | 2.38418579e-07/8.76372894e-07 | 2e74f7d0771780479a4e1b3ec9ae545e70daeb8580fbb1f1379b5cb0b121cba2 |
| 9 | [6, 7] | [100, 100] | 200 | 2.38418579e-07/9.26592634e-07 | dab660fce8a0ec6538fee044f0cc14eadee620a95f439b7678ce713b574408f3 |

Seed 2 provenance: {"oracle_sha256": "d0165050117976afcb1e7ddca3afb70971bb033799ddba42d43560ad4b86dd2a", "support_sha256": "b59d0199adad3696cf32f95d272375d23f4ca0979da76e234da806d2fbefab4a", "anchor_sha256": "9582386f902eb84ac6ad3ee40d851dad04b8aa3110adb6a5d43846513e802f1c", "train_sha256": "1535c0c51f0b74f833054179c6377b4704de6cb727bfbd12e8add4dabe30ef25"}
N256 prefix SHA256: 9da6b540e718e33666fd440d53b88b350e7caa34b4010857190549546c5e467b
Local train SHA256: 1535c0c51f0b74f833054179c6377b4704de6cb727bfbd12e8add4dabe30ef25
Per-client train SHA256: ["642c8b718dfd5e5be00ca642da19d0f42e147064820c998be03f5ad27324d50a", "d3a222ca9532911e61d5499258995e359fb5cbb1f923f9d9d995b403abc0ec80", "5473860621091b39c115f2c097b71eb12e7f6d4e150951b0840a45ef3e817203", "aed9cec7e6c7ae7c579026551ad82a6daa4689959c5f844cedf1135d2a0e2352", "979093c48a8986e78c78250964afb9df840839bc7735835dd37fadb6b65762d9", "897f27357693c4221aed6352aaa385d4ffa8dc0b0bfa4d66bd159c4b2871a181", "415fa6cefd33a9b1876ddfa9f0704fef8850232b34534b1db80b7958ff44a4a0", "0a22c119ea51e315f063021119e6f4730cea3ab03e344713e1f5dc3bb7425d95", "1ec2539b51d7929fb05115a078687556ecec7f98b431c890e96d204750a49146", "76cc7fc774fbea6f71983dc342ed51b5bc9e1c648e1e7d4d5b3feae0a42757eb"]
Class sets: [[0, 2], [0, 7], [6, 7], [6, 9], [5, 9], [3, 5], [3, 4], [4, 8], [1, 8], [1, 2]]
Transform SHA256: ["734ec454e29817a259f5f660747ed1f6105fcbbb0d8c636c9afba71c8e56e00e", "f947c2630df525e75549c69d73c80b15d7433940161610fe5e120f35a867f2ce", "49e4eb1e93c4f6e24bbc47e15ff2ea9397810ad385d3fc56c253d250ab35cb1c", "814182c246956766f03d7e67a84b0109303e98e52bc9939f17057cd732304061", "65332ce077e66d4e5e13dc6455272d27fe29ff1e2b36f6cd5ae57554191da88e", "5a55dfa06ef175d795029485cf31891f8b8d9ed671fa334a7eddd943183331b9", "2b0ddf090d30df405b6a552f17aaa9a3c83a739c5ad9a798b4e9aa8915ad77a2", "0faa35d7a6ca83b34615244a5c3a3fbba7f3df02a443f8abf17622b2a9692945", "9dbd5a52e85b8fdba19a55b300abac0f748d1d5c6fd8eb81b3487b901b3ee5dc", "de3677ac73f0f076f2684b3f204daf0033d195c489f527c774ec51cf2d15103c"]
Refresh: 2000 forward examples per arm (200/client); local compute, not communication.

seed2_localtrain_aligned_global_prototype_cosine:
Aggregate SHA256: ffbc3794c951af6011c57c90e16e1da017edf46d217b04602140afe88520add3
Norm range: 13.5338898 to 16.9584179
Prediction histograms (class0..9): {"overall": [986, 613, 553, 166, 110, 1020, 2163, 851, 2374, 1164], "seen": [214, 114, 103, 35, 20, 204, 433, 168, 463, 246], "missing": [772, 499, 450, 131, 90, 816, 1730, 683, 1911, 918]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total | Max/Fro hierarchical error | SHA256 |
|---|---|---|---:|---|---|
| 0 | [0, 1] | [100, 100] | 200 | 3.57627869e-07/9.97509915e-07 | ce9e7c8bbd5b318cef6fb94f79e6f657de81af2b9fae704cba92f9e0330bfb43 |
| 1 | [8, 9] | [100, 100] | 200 | 4.76837158e-07/1.31550769e-06 | e95cfc1b103154056dba0e753a6f785497fa9372ff6013f6977b51ec05d146e7 |
| 2 | [0, 9] | [100, 100] | 200 | 2.38418579e-07/1.01119406e-06 | 913051baf1780773c791faf7862795ded592e7d7b66011b9d51117ced723fe5f |
| 3 | [5, 6] | [100, 100] | 200 | 2.38418579e-07/1.10333428e-06 | 98cf74aa43d33c04c8fd7a6d1139a99efc65785d54c9cb0e4b20e088ec1da807 |
| 4 | [6, 7] | [100, 100] | 200 | 4.76837158e-07/1.20362552e-06 | 62cfb099a92216995e866d5db85e0d654a86b747d7518cd8b5a7540ad3efb742 |
| 5 | [4, 5] | [100, 100] | 200 | 3.57627869e-07/1.14961915e-06 | a6b1fb5a1196f048db770304a9fa59d75730f6f1b66e0492f58aca01254aaf38 |
| 6 | [2, 3] | [100, 100] | 200 | 4.76837158e-07/1.36563472e-06 | 51858679c3197a6bafefa6ef13d837f4f586c9b02a60013a31a736ef2bdeeb43 |
| 7 | [1, 2] | [100, 100] | 200 | 4.76837158e-07/1.29941475e-06 | 896eb9cfe9d6dc3cd77bb4c18d85b738c981f35cda521e185020ae12211e7afa |
| 8 | [7, 8] | [100, 100] | 200 | 2.38418579e-07/1.11899283e-06 | 347e0fd13d953658fa462b40168b5aeb81a050283f0b175c3f3ee7d38cf72a90 |
| 9 | [3, 4] | [100, 100] | 200 | 2.38418579e-07/1.14263526e-06 | a5222a3a6d0fe6e9214ace7ff45ff46fcc8e07b1b11dde75e6c980066f4fc6d1 |

seed2_localtrain_native_global_prototype_cosine_control:
Aggregate SHA256: 20cc4059d3dfb40c9b535905b0eb9bb3eef70831a00dd7098e70a5f685fec46f
Norm range: 11.6060009 to 15.3357172
Prediction histograms (class0..9): {"overall": [1840, 1762, 193, 1977, 628, 516, 1831, 50, 527, 676], "seen": [386, 345, 26, 390, 113, 98, 371, 2, 128, 141], "missing": [1454, 1417, 167, 1587, 515, 418, 1460, 48, 399, 535]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 0, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total | Max/Fro hierarchical error | SHA256 |
|---|---|---|---:|---|---|
| 0 | [0, 1] | [100, 100] | 200 | 2.38418579e-07/1.01120577e-06 | 3d4236fb015c4360022331ccd356cc9343bd42cfd847ae83e4252b1c29774545 |
| 1 | [8, 9] | [100, 100] | 200 | 2.38418579e-07/9.37277377e-07 | a05a20c106d881cdbb43a6dd82730d32335b419e21e7b7e10a166a03a712ee31 |
| 2 | [0, 9] | [100, 100] | 200 | 2.38418579e-07/8.8912833e-07 | f9c159421366630fcbfe01d7fc4b5f46e40c2365822dfbe59335a0cd119c98c4 |
| 3 | [5, 6] | [100, 100] | 200 | 3.57627869e-07/1.10814278e-06 | 31fe3ea7e91d109a49780834573cd2a0cb307cacec5be7054f2aa78b44ffe8ef |
| 4 | [6, 7] | [100, 100] | 200 | 4.76837158e-07/1.03732441e-06 | 4bfc6197fc4c8ff1d4ba90d6f438188b022ea2e0b033bc1e560a220c24e0c4ba |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07/8.35186484e-07 | f26acac5f3adaf6e9da0ad7a2b1bbea0999ad8138b44690c8846ae5eb39af6b1 |
| 6 | [2, 3] | [100, 100] | 200 | 2.38418579e-07/9.67243295e-07 | 4b232b97a734c56a4bf3780aeeb8c1701e517c981a0d5f1516bb2b84938e2e9d |
| 7 | [1, 2] | [100, 100] | 200 | 2.38418579e-07/1.0061234e-06 | 9f97e5e99c9d1ecff1ce71520c45e6e309024bd668b401adb63e41073b6c0c81 |
| 8 | [7, 8] | [100, 100] | 200 | 2.38418579e-07/8.07948709e-07 | 710986230cc8feb3ace728d5295a742517c6b2afd6fa53fc3a45aa26d55e6221 |
| 9 | [3, 4] | [100, 100] | 200 | 2.38418579e-07/8.83245832e-07 | b8694b2a9640ac151731db80bf062a1ac7d854eca7db497198992dd832388454 |
