# H07-A ordinary local semantic source

| Arm | Seen % | Missing % | All % | Macro % | Predicted classes |
|---|---:|---:|---:|---:|---:|
| heldout_aligned_global_prototype_cosine_reference | 27.500000 | 23.237500 | 24.090000 | 24.090000 | 10 |
| localtrain_aligned_global_prototype_cosine | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| localtrain_native_global_prototype_cosine_control | 54.099999 | 0.000000 | 10.820000 | 10.820000 | 9 |

Frozen verdict: A: strong ordinary-local semantic closure. ret_missing=0.957504049; ret_all=0.97094229; alignment_gain=22.250000pp.

localtrain_aligned_global_prototype_cosine:
Aggregate SHA256: c3d0f35e1b61d066185b89f1e12558aa4a51d3f86f21fc2e274930d5852d1695
Norm range: 12.4210386 to 14.7693615
Prediction histograms (class0..9): {"overall": [1109, 1429, 1036, 367, 1113, 1532, 1823, 207, 640, 744], "seen": [232, 266, 229, 84, 213, 332, 337, 48, 117, 142], "missing": [877, 1163, 807, 283, 900, 1200, 1486, 159, 523, 602]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 5242880, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |
|---|---|---|---:|---:|---:|---|
| 0 | [6, 7] | [100, 100] | 200 | 4.76837158e-07 | 1.04314461e-06 | 1782cae65657f4d1f2513b49967511831691400e85fc30d78fe7c2993ac0bbd0 |
| 1 | [8, 9] | [100, 100] | 200 | 2.38418579e-07 | 1.02717377e-06 | dfe7b52e785fda6fe96b418b705f1d5bc9f00ddfcd779b44abcc4fff5de3eb73 |
| 2 | [1, 2] | [100, 100] | 200 | 2.38418579e-07 | 9.32140892e-07 | 5d7cfa00b1a61697a2ac57b7933dcf129746a4e4f030264b1d8fb89e1d0f0668 |
| 3 | [3, 4] | [100, 100] | 200 | 3.57627869e-07 | 1.14584122e-06 | 119e8d9a63a0f7e6390f75ad05afc557fa97ebd81231fa8c272b24318425478d |
| 4 | [0, 9] | [100, 100] | 200 | 2.38418579e-07 | 8.46350019e-07 | da586ee1df90418966dd81712af4f407750331d8eeca7a409b132d22d6bc0b47 |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07 | 9.63432285e-07 | a414080a39d4cbae314ffee649b9bfd54220fd8b01b6ebb382b2b140c9d39303 |
| 6 | [0, 1] | [100, 100] | 200 | 4.76837158e-07 | 1.22051335e-06 | be5f336db0a09d8b9002c8f043141031d93758826390bd5fe6c8fc4121df5a5b |
| 7 | [2, 3] | [100, 100] | 200 | 2.38418579e-07 | 1.08478116e-06 | 39c8958d44c020298e9ade6fd20a86e0f2ba78a71384cdb4cb3cb16e43b1d82c |
| 8 | [7, 8] | [100, 100] | 200 | 4.76837158e-07 | 1.12895168e-06 | 214a44eaaf80b28a853ab9c352f73021fa276426efb6790cf6a7e65c00143485 |
| 9 | [5, 6] | [100, 100] | 200 | 2.38418579e-07 | 9.80289997e-07 | 27e3eb4f746d1718415ee472e30b2bf10c82e552d2ba51131249940b798ed042 |

localtrain_native_global_prototype_cosine_control:
Aggregate SHA256: 82159cd36fc2376abff996f3a597cc031b12e47f8463daf951b590f10a7ce3e3
Norm range: 11.6439543 to 15.0607529
Prediction histograms (class0..9): {"overall": [1842, 1929, 2000, 1982, 1037, 553, 1, 0, 33, 623], "seen": [384, 390, 400, 390, 205, 113, 1, 0, 4, 113], "missing": [1458, 1539, 1600, 1592, 832, 440, 0, 0, 29, 510]}
Communication bytes: {"semantic_uplink_bytes": 41280, "anchor_uplink_bytes": 0, "global_vectors_downlink_per_client": 20480, "global_vectors_downlink_total": 204800, "class_ids_downlink_bytes": 0, "class_order": "fixed ascending 0..9; no separate IDs transmitted", "learned_head_downlink_per_client": 20520, "learned_head_downlink_total": 205200}

| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |
|---|---|---|---:|---:|---:|---|
| 0 | [6, 7] | [100, 100] | 200 | 3.57627869e-07 | 8.85221311e-07 | 969bfe5847a5885cb8b1c1b466c61352c9b1bf82d149efb23b0aa81b8059f1a3 |
| 1 | [8, 9] | [100, 100] | 200 | 3.57627869e-07 | 1.11544318e-06 | dbfcbcd536b0083c9fcd67dfcecd6dbd50435d2b0836bb6e92feda8d59facdb7 |
| 2 | [1, 2] | [100, 100] | 200 | 2.38418579e-07 | 8.65154448e-07 | 7f0c3d71f1cfca8f8d67c175fd81b55633fb0f90eb4188073497551edbee6a74 |
| 3 | [3, 4] | [100, 100] | 200 | 2.38418579e-07 | 1.00104e-06 | 493aae11782a9fd4da6ac08e452d9ac9f52feecd2c850a77b93cb591b576a3f1 |
| 4 | [0, 9] | [100, 100] | 200 | 3.57627869e-07 | 9.26148289e-07 | 6fbe042a078b9c69d4911b92510ae6892eec76458e4fb2ad8bb25c09c4b04a86 |
| 5 | [4, 5] | [100, 100] | 200 | 2.38418579e-07 | 8.77466675e-07 | 24b08904adcf492b2d16900c182867a9b5107ac7afbaa8f3ceb76c14a5dce851 |
| 6 | [0, 1] | [100, 100] | 200 | 4.76837158e-07 | 1.01596265e-06 | 3b483596852487c38ede8fed150a38c2267646b99db45f86d04545ea17ad7449 |
| 7 | [2, 3] | [100, 100] | 200 | 4.76837158e-07 | 9.64835863e-07 | 17659f0496538431335a51a35fc912ba102c81012ecbc85f66062d51a6129353 |
| 8 | [7, 8] | [100, 100] | 200 | 3.57627869e-07 | 9.64718424e-07 | 84a279b53ff4cc1fe9ab6a67a41202b34eecb037f285e14af8eb5e24bfeb9632 |
| 9 | [5, 6] | [100, 100] | 200 | 3.57627869e-07 | 9.97603706e-07 | 2cfc54e6aa55d4f23fa4d6be38d0c575230928d27dcbe5f0d7c1629ccf581b16 |

Ordinary local train index SHA256: 5fbbd599df082e9f32630e9f65fc1e41aea4931f1ffe6957b022c2cfd56847a5
Perclient index SHA256: ["817fe99e3085d6924872ae39989466dbf9b3b63538c0e3120227460c73754b43", "ecb6f2a84c4c598e3fa2bcc7f38d8fbf982ae16d140b8e9bf2e70df8be191123", "e90b4bc94e79e7dc2aefa3a50a5da8b1fa5f1e4a1264494cf639f7fa6217990e", "699994b118d8217dbe65f598bd508de262437bcd841db81997dd66f4e75792a3", "a129cbaedc10d7e7c046855c58d44d040830b9c20ad8368690d53259531a4238", "7eb0c04dfcd9d1ac01f0db02d4df7f6d69e01351e9dd9420351185be72eec7fa", "bb2b19994bf19bd85405cf937464f0a77917eb92274ba111422996a08d41908c", "9fcb02324f1c2312bae0ef3b89d09d206b26ea947c199eb2dea3d91898c04508", "73ef4e51785dc3dfcfb269c2ec396282345741da603a390389ad76ef63e3f5b6", "8d944777dfe49ed996f17e6e7c879c0dca4974636e22a4c053e2feaf391bbb03"]
Prototype refresh forward examples: 2000 total; [200, 200, 200, 200, 200, 200, 200, 200, 200, 200] perclient. Local compute, not communication. Each diagnostic arm recomputes this same refresh.

| Class | New-vs-heldout cosine | L2 distance |
|---|---:|---:|
| 0 | 0.999792457 | 0.40283376 |
| 1 | 0.999921918 | 0.324709892 |
| 2 | 0.999927461 | 0.151780576 |
| 3 | 0.999920905 | 0.39572075 |
| 4 | 0.999927282 | 1.00032175 |
| 5 | 0.999955058 | 0.402224243 |
| 6 | 0.999985576 | 0.150540978 |
| 7 | 0.999943197 | 0.154670596 |
| 8 | 0.999676466 | 0.629140198 |
| 9 | 0.999951839 | 0.272004515 |
