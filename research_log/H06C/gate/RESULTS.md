# H06-C direct global prototype readout

| Arm | Seen % | Missing % | All % | Macro % | Predicted classes |
|---|---:|---:|---:|---:|---:|
| aligned_prototype_learned_head_reference | 27.950000 | 23.887500 | 24.700000 | 24.700000 | not collected |
| aligned_global_prototype_cosine | 27.500000 | 23.237500 | 24.090000 | 24.090000 | 10 |
| native_global_prototype_cosine_control | 56.150000 | 0.000000 | 11.230000 | 11.230000 | 9 |

Frozen verdict: A: strong direct-global-prototype readout. ret_missing=0.972789116; ret_all=0.975303654; alignment_gain=23.237500pp.

aligned_global_prototype_cosine:
Aggregate SHA256: b34eb7e71b3422d39067d91f70422f3e896db4df482edaa58860dfc6900792f7
Norm range: 12.5222855 to 14.6042805
Prediction histograms (class0..9): {"overall": [1474, 1166, 676, 467, 1017, 1431, 1732, 525, 342, 1170], "seen": [286, 238, 157, 113, 186, 306, 328, 120, 53, 213], "missing": [1188, 928, 519, 354, 831, 1125, 1404, 405, 289, 957]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |
|---|---|---|---:|---:|---:|---|
| 0 | [6, 7] | [100, 100] | 200 | 2.38418579e-07 | 1.08294807e-06 | 467153fe909a6625fedb3909ffde34da5eb4dbf66a56f54a46514a0c5bdf8569 |
| 1 | [8, 9] | [100, 100] | 200 | 2.38418579e-07 | 1.16244235e-06 | 5b63e1d5e9620db4db703122dd62cd82e2a22af59ee521ad5c6077eb5ee1f94c |
| 2 | [1, 2] | [100, 100] | 200 | 2.38418579e-07 | 1.02201011e-06 | 0f0ac23df181f07ce59a8e2fd9b70894f859ea29dea802b2fcf5cc070ad81426 |
| 3 | [3, 4] | [100, 100] | 200 | 4.76837158e-07 | 1.16240733e-06 | 7cb2b7522fa64de9a5101beb175784ea6906f647084c3cb8abb05cbc1b3ca5c5 |
| 4 | [0, 9] | [100, 100] | 200 | 4.76837158e-07 | 9.56708391e-07 | 69fc2b4da217189a0e60a6c94e0335a10d900bf9e323c3bfba21c9f6d27602e7 |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07 | 1.01954197e-06 | 4e919450acdab32d56dbd3fdb7c445fb4149991b8acbfee7ea049eb272c1f9f9 |
| 6 | [0, 1] | [100, 100] | 200 | 2.38418579e-07 | 9.79497827e-07 | 8d934bd7e3df9291fa9f2225a1974bee3eee5e7416e651a36a992503ec2eb4d1 |
| 7 | [2, 3] | [100, 100] | 200 | 2.38418579e-07 | 1.02947274e-06 | 7f0d62eebe5cf2a507333afa7b26285b875daecf4f682ca2a54f23935ea3558c |
| 8 | [7, 8] | [100, 100] | 200 | 2.38418579e-07 | 1.01183844e-06 | 6eb741eaec527da85ea3ca66a8b796053ddc81fbaae65f394b1c60bf5b811f48 |
| 9 | [5, 6] | [100, 100] | 200 | 4.76837158e-07 | 1.31207275e-06 | 11a24cb23d4d63eef015a66d3be02de6f35400d2295bb032cc50658640157d0f |

native_global_prototype_cosine_control:
Aggregate SHA256: 3916e44c68d9628fcf03b3f22637652dc0b6679094814b4b201762c21442f28d
Norm range: 11.0866365 to 14.8064051
Prediction histograms (class0..9): {"overall": [1857, 1813, 2000, 1977, 791, 534, 381, 0, 15, 632], "seen": [387, 371, 400, 387, 134, 111, 94, 0, 1, 115], "missing": [1470, 1442, 1600, 1590, 657, 423, 287, 0, 14, 517]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 0, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |
|---|---|---|---:|---:|---:|---|
| 0 | [6, 7] | [100, 100] | 200 | 2.38418579e-07 | 8.23387211e-07 | 12046c64e327e819cc53c70baa7c3f945924a68b899f8b39228a2d577bffdc88 |
| 1 | [8, 9] | [100, 100] | 200 | 2.38418579e-07 | 1.01743717e-06 | b40d89d3c8b7dcec00a905ccd88df09b3746dd5c3dea5dd666764bdd7eb17f56 |
| 2 | [1, 2] | [100, 100] | 200 | 3.57627869e-07 | 8.09922881e-07 | 8144577c859a0abb7401c9532c2be08e9b78ae8b4d8dde86ec86e2608ca6740a |
| 3 | [3, 4] | [100, 100] | 200 | 2.38418579e-07 | 1.05255674e-06 | 7ea85b41252a14574444b22568d5f624752a69c6cbfc02a2fdca3b522b683be9 |
| 4 | [0, 9] | [100, 100] | 200 | 2.38418579e-07 | 9.36808703e-07 | c5a23c164f9ba01f498c94f54a2076edf2441cb9d1a92357c215fb2498552c32 |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07 | 8.34946206e-07 | 47dd9fd5e30f44a09a0f75f28a9a6e94eb5edfeab02a86a53f1d2dc54ec2dda9 |
| 6 | [0, 1] | [100, 100] | 200 | 3.57627869e-07 | 9.57729753e-07 | 177864886faba93b10b098e12d97ad3a6e0ba5ce6a7262912bc073c6d85000bd |
| 7 | [2, 3] | [100, 100] | 200 | 2.38418579e-07 | 8.76799788e-07 | 60211617e093ef70434a452b85610885db2e4d2df973cfa3801df168ce7993c6 |
| 8 | [7, 8] | [100, 100] | 200 | 1.1920929e-07 | 6.50442814e-07 | 458e5d5c2d7982c78b3f8b3a096a6803ae7dc7ebe7d86b84faa86e20c9baa591 |
| 9 | [5, 6] | [100, 100] | 200 | 2.38418579e-07 | 9.12341193e-07 | 21fea87c41a6701a49dc6c7005aec17e8c66335445f854ebaff226efacacb059 |
