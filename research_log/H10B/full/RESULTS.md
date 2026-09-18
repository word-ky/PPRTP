# H10-B aligned group gate / native owner refinement

| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | H07 aligned | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| 0 | H09-A raw dual | 38.050000 | 16.875000 | 21.110000 | 21.110000 | 10 |
| 0 | H10-B group refine | 28.550000 | 22.250000 | 23.510000 | 23.510000 | 10 |
| 1 | H07 aligned | 28.350000 | 21.737500 | 23.060000 | 23.060001 | 10 |
| 1 | H09-A raw dual | 20.450000 | 23.237500 | 22.680000 | 22.680001 | 10 |
| 1 | H10-B group refine | 28.650000 | 21.737500 | 23.120000 | 23.120001 | 10 |
| 2 | H07 aligned | 24.200000 | 20.700000 | 21.400000 | 21.400000 | 10 |
| 2 | H09-A raw dual | 32.500000 | 18.250000 | 21.100000 | 21.099999 | 10 |
| 2 | H10-B group refine | 24.400000 | 20.700000 | 21.440000 | 21.440000 | 10 |

Strong seeds: 0/3. Frozen verdict: 0/3: stop routing/fusion line; retain H07.

No labels enter prediction/routing; no calibration or extra local refresh. Incremental communication and persistent storage: 0 B relative to existing H07 objects. Existing anchor and affine-map delivery caveats remain. Full per-client/class metrics and counts are in final.json.

Seed 0: historical owner-only seen diagnostic ceiling 66.3500%.
Routing and five transitions: {"total": 10000, "seen_count": 2000, "missing_count": 8000, "native_count": 2528, "seen_routed": 800, "missing_routed": 1728, "shared_missing_correct": 1780, "refine_missing_correct": 1780, "shared_seen_correct": 559, "refine_seen_correct": 571, "transitions": {"correct_to_correct": 519, "correct_to_wrong": 40, "wrong_owned_to_correct": 52, "wrong_owned_to_wrong": 189, "wrong_missing_unchanged": 1200}, "native_route_rate": 0.2528, "true_seen_owner_route_rate": 0.4, "true_missing_owner_route_rate": 0.216}
Prediction histograms: {"overall": [968, 1444, 1023, 361, 1116, 1533, 1819, 226, 709, 801], "seen": [186, 269, 224, 82, 215, 332, 335, 55, 144, 158], "missing": [782, 1175, 799, 279, 901, 1201, 1484, 171, 565, 643]}
Per-client exact missing (H07, H10-B): [(190, 190), (197, 197), (216, 216), (229, 229), (184, 184), (157, 157), (146, 146), (134, 134), (175, 175), (152, 152)]

Seed 1: historical owner-only seen diagnostic ceiling 75.0500%.
Routing and five transitions: {"total": 10000, "seen_count": 2000, "missing_count": 8000, "native_count": 2499, "seen_routed": 703, "missing_routed": 1796, "shared_missing_correct": 1739, "refine_missing_correct": 1739, "shared_seen_correct": 567, "refine_seen_correct": 573, "transitions": {"correct_to_correct": 552, "correct_to_wrong": 15, "wrong_owned_to_correct": 21, "wrong_owned_to_wrong": 115, "wrong_missing_unchanged": 1297}, "native_route_rate": 0.2499, "true_seen_owner_route_rate": 0.3515, "true_missing_owner_route_rate": 0.2245}
Prediction histograms: {"overall": [1775, 741, 826, 294, 228, 1300, 2199, 787, 930, 920], "seen": [353, 157, 177, 68, 53, 248, 412, 153, 174, 205], "missing": [1422, 584, 649, 226, 175, 1052, 1787, 634, 756, 715]}
Per-client exact missing (H07, H10-B): [(205, 205), (203, 203), (143, 143), (167, 167), (181, 181), (181, 181), (121, 121), (156, 156), (178, 178), (204, 204)]

Seed 2: historical owner-only seen diagnostic ceiling 70.4500%.
Routing and five transitions: {"total": 10000, "seen_count": 2000, "missing_count": 8000, "native_count": 2280, "seen_routed": 638, "missing_routed": 1642, "shared_missing_correct": 1656, "refine_missing_correct": 1656, "shared_seen_correct": 484, "refine_seen_correct": 488, "transitions": {"correct_to_correct": 476, "correct_to_wrong": 8, "wrong_owned_to_correct": 12, "wrong_owned_to_wrong": 142, "wrong_missing_unchanged": 1362}, "native_route_rate": 0.228, "true_seen_owner_route_rate": 0.319, "true_missing_owner_route_rate": 0.20525}
Prediction histograms: {"overall": [992, 611, 547, 176, 111, 1010, 2094, 909, 2374, 1176], "seen": [216, 115, 100, 36, 20, 203, 419, 179, 463, 249], "missing": [776, 496, 447, 140, 91, 807, 1675, 730, 1911, 927]}
Per-client exact missing (H07, H10-B): [(193, 193), (176, 176), (135, 135), (125, 125), (132, 132), (193, 193), (207, 207), (140, 140), (158, 158), (197, 197)]
