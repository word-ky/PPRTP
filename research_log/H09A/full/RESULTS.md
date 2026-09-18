# H09-A dual-space classifier

| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | aligned_global_prototype_cosine | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| 0 | native_global_prototype_cosine_control | 54.099999 | 0.000000 | 10.820000 | 10.820000 | 9 |
| 0 | dual_space_owner_seen_aligned_missing | 38.050000 | 16.875000 | 21.110000 | 21.110000 | 10 |
| 1 | aligned_global_prototype_cosine | 28.350000 | 21.737500 | 23.060000 | 23.060001 | 10 |
| 1 | native_global_prototype_cosine_control | 61.500000 | 0.000000 | 12.300000 | 12.300000 | 10 |
| 1 | dual_space_owner_seen_aligned_missing | 20.450000 | 23.237500 | 22.680000 | 22.680001 | 10 |
| 2 | aligned_global_prototype_cosine | 24.200000 | 20.700000 | 21.400000 | 21.400000 | 10 |
| 2 | native_global_prototype_cosine_control | 58.650000 | 0.000000 | 11.730000 | 11.730000 | 10 |
| 2 | dual_space_owner_seen_aligned_missing | 32.500000 | 18.250000 | 21.100000 | 21.099999 | 10 |

Strong seeds: 0/3. no seed passes; inspect components and score dominance
Components are label-partitioned diagnostics only; proposed scores/predictions never use test labels. Full per-client/class counts, histograms and score distributions are in final.json.
Incremental communication0B relative to H07 aligned inference. Owner prototypes stay local. Prior anchor/global-bank/affine costs and incomplete distributed-protocol caveat remain.

Seed 0 component diagnostics: native_owner_seen_only=66.350000%; aligned_missing_only=27.500000%.
Winning group fractions: {"native_seen": 0.4721999764442444, "aligned_missing": 0.5278000235557556}
Prediction histograms: {"overall": [1052, 1661, 984, 581, 1045, 1303, 1113, 327, 891, 1043], "seen": [189, 290, 224, 122, 183, 292, 238, 84, 189, 189], "missing": [863, 1371, 760, 459, 862, 1011, 875, 243, 702, 854]}
Global bank SHA256: c3d0f35e1b61d066185b89f1e12558aa4a51d3f86f21fc2e274930d5852d1695

| True subset | Score | Mean | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|
| seen | max_seen_score | 0.995723546 | 0.992645502 | 0.996127963 | 0.998223662 |
| seen | max_missing_score | 0.994373262 | 0.989637494 | 0.995980859 | 0.998219371 |
| seen | difference | 0.00135032996 | -0.00134513364 | 0.000148445368 | 0.00414012652 |
| missing | max_seen_score | 0.9946751 | 0.990834951 | 0.995290339 | 0.997828066 |
| missing | max_missing_score | 0.993691802 | 0.988099158 | 0.995581031 | 0.998018622 |
| missing | difference | 0.000983260106 | -0.00274114008 | -0.000129699707 | 0.00432825554 |

Native/aligned winning fractions on true-seen: 0.573000014/0.426999986.
Native/aligned winning fractions on true-missing: 0.447000027/0.552999973.

Seed 1 component diagnostics: native_owner_seen_only=75.050000%; aligned_missing_only=27.675000%.
Winning group fractions: {"native_seen": 0.18559999763965607, "aligned_missing": 0.8144000023603439}
Prediction histograms: {"overall": [1607, 787, 720, 535, 483, 1190, 1925, 866, 1059, 828], "seen": [314, 155, 161, 128, 111, 220, 351, 160, 211, 189], "missing": [1293, 632, 559, 407, 372, 970, 1574, 706, 848, 639]}
Global bank SHA256: 2d787aefd7a01b93696dfdbf70ba6576695c3bb8d618912625b2d1034e716e0b

| True subset | Score | Mean | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|
| seen | max_seen_score | 0.99570024 | 0.992432415 | 0.996396661 | 0.998299718 |
| seen | max_missing_score | 0.996394336 | 0.993707299 | 0.997066617 | 0.998487532 |
| seen | difference | -0.000694101211 | -0.0027639505 | -0.000574439764 | 0.00108959642 |
| missing | max_seen_score | 0.995012641 | 0.991439939 | 0.995805144 | 0.998011231 |
| missing | max_missing_score | 0.996296048 | 0.993500888 | 0.997052193 | 0.998452485 |
| missing | difference | -0.0012834348 | -0.00373356347 | -0.000936806202 | 0.000507592922 |

Native/aligned winning fractions on true-seen: 0.256000012/0.743999988.
Native/aligned winning fractions on true-missing: 0.168000013/0.831999987.

Seed 2 component diagnostics: native_owner_seen_only=70.450000%; aligned_missing_only=25.562500%.
Winning group fractions: {"native_seen": 0.361299991607666, "aligned_missing": 0.638700008392334}
Prediction histograms: {"overall": [836, 638, 735, 546, 694, 1139, 1614, 911, 1627, 1260], "seen": [183, 121, 129, 148, 139, 222, 287, 157, 357, 257], "missing": [653, 517, 606, 398, 555, 917, 1327, 754, 1270, 1003]}
Global bank SHA256: ffbc3794c951af6011c57c90e16e1da017edf46d217b04602140afe88520add3

| True subset | Score | Mean | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|
| seen | max_seen_score | 0.993776441 | 0.988666236 | 0.994927645 | 0.997683287 |
| seen | max_missing_score | 0.99337858 | 0.98825258 | 0.995201349 | 0.997707844 |
| seen | difference | 0.000397892436 | -0.00304987421 | -0.000220924616 | 0.00339453737 |
| missing | max_seen_score | 0.992888212 | 0.986767411 | 0.99438554 | 0.997343004 |
| missing | max_missing_score | 0.992784321 | 0.986477673 | 0.995407939 | 0.997870982 |
| missing | difference | 0.000103889055 | -0.0039228797 | -0.000727981329 | 0.00325262314 |

Native/aligned winning fractions on true-seen: 0.443500012/0.556499988.
Native/aligned winning fractions on true-missing: 0.340750009/0.659249991.
