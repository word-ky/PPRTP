# H09-B centered residual readout

| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | aligned_global_prototype_cosine | 27.950000 | 22.250000 | 23.390000 | 23.389999 | 10 |
| 0 | dual_space_owner_seen_aligned_missing | 38.050000 | 16.875000 | 21.110000 | 21.110000 | 10 |
| 0 | centered_dual_owner_seen_aligned_missing | 25.250000 | 16.987500 | 18.640000 | 18.640000 | 10 |
| 0 | centered_aligned_global_only | 26.200000 | 17.512500 | 19.250000 | 19.250000 | 10 |
| 1 | aligned_global_prototype_cosine | 28.350000 | 21.737500 | 23.060000 | 23.060001 | 10 |
| 1 | dual_space_owner_seen_aligned_missing | 20.450000 | 23.237500 | 22.680000 | 22.680001 | 10 |
| 1 | centered_dual_owner_seen_aligned_missing | 30.500000 | 21.662500 | 23.430000 | 23.430000 | 10 |
| 1 | centered_aligned_global_only | 30.250000 | 22.187500 | 23.800000 | 23.800000 | 10 |
| 2 | aligned_global_prototype_cosine | 24.200000 | 20.700000 | 21.400000 | 21.400000 | 10 |
| 2 | dual_space_owner_seen_aligned_missing | 32.500000 | 18.250000 | 21.100000 | 21.099999 | 10 |
| 2 | centered_dual_owner_seen_aligned_missing | 26.300000 | 21.500000 | 22.460000 | 22.460000 | 10 |
| 2 | centered_aligned_global_only | 25.900000 | 21.850000 | 22.660000 | 22.660000 | 10 |

Strong seeds: 0/3. 0/3 pass; inspect component and centered-global control
Centering uses existing affine means only, no fitting or test-label routing. Component diagnostics are label-partitioned only. All per-client/class counts, norms/quantiles, histograms and hashes are in final.json.
Incremental communication0B relative to H09-A/H07; existing anchor/global-bank/affine delivery caveats remain.

Seed 0: worst owner rotation error=5.36441803e-07.
Centered component diagnostics: {"native_owner_seen_only": 0.6225, "aligned_missing_only": 0.21525}
Winning group fractions: {"native_seen": 0.26420000195503235, "aligned_missing": 0.7357999980449677}
Prediction histograms: {"overall": [404, 701, 2894, 289, 1988, 304, 1165, 1245, 376, 634], "seen": [109, 153, 509, 54, 313, 69, 273, 261, 101, 158], "missing": [295, 548, 2385, 235, 1675, 235, 892, 984, 275, 476]}

| Norm object | Minimum over clients | Mean of client means | Maximum over clients |
|---|---:|---:|---:|
| feature_raw | 3.01391459 | 13.7179429 | 31.1667862 |
| feature_centered | 0.668966055 | 3.22120068 | 15.3497753 |
| owner_raw | 12.9917393 | 14.2406823 | 17.5248699 |
| owner_centered | 0.276978314 | 0.972113764 | 1.81345713 |
| global_raw | 12.4210386 | 13.6263523 | 14.7693615 |
| global_centered | 0.449953258 | 0.920761764 | 1.61500323 |

| True subset | Score | Mean | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|
| seen | max_seen_score | 0.511220336 | -0.168279365 | 0.668552756 | 0.945795357 |
| seen | max_missing_score | 0.757056653 | 0.431949556 | 0.833817124 | 0.947531044 |
| seen | difference | -0.245836318 | -0.853108704 | -0.0510620773 | 0.0940119028 |
| missing | max_seen_score | 0.37047106 | -0.465689808 | 0.479978383 | 0.928146899 |
| missing | max_missing_score | 0.758516788 | 0.434929311 | 0.830438614 | 0.948973 |
| missing | difference | -0.388045698 | -1.16958725 | -0.166437984 | 0.0560950264 |

Native/aligned winning fractions on true-seen: 0.362000018/0.637999982.
Native/aligned winning fractions on true-missing: 0.239750013/0.760249987.

Seed 1: worst owner rotation error=4.17232513e-07.
Centered component diagnostics: {"native_owner_seen_only": 0.722, "aligned_missing_only": 0.271}
Winning group fractions: {"native_seen": 0.2685999870300293, "aligned_missing": 0.7314000129699707}
Prediction histograms: {"overall": [540, 1312, 2287, 267, 787, 382, 873, 1201, 1698, 653], "seen": [129, 306, 459, 54, 124, 71, 186, 245, 278, 148], "missing": [411, 1006, 1828, 213, 663, 311, 687, 956, 1420, 505]}

| Norm object | Minimum over clients | Mean of client means | Maximum over clients |
|---|---:|---:|---:|
| feature_raw | 4.61305571 | 13.8362884 | 24.8253193 |
| feature_centered | 0.576897681 | 2.88201411 | 10.8448172 |
| owner_raw | 12.6236715 | 14.1766624 | 15.9401598 |
| owner_centered | 0.289986283 | 1.13190895 | 2.10251713 |
| global_raw | 13.8659906 | 15.5080175 | 16.9461403 |
| global_centered | 0.343241155 | 1.09651613 | 1.8531996 |

| True subset | Score | Mean | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|
| seen | max_seen_score | 0.493506432 | -0.247481525 | 0.653248668 | 0.940529525 |
| seen | max_missing_score | 0.754810631 | 0.477662027 | 0.810191274 | 0.938678443 |
| seen | difference | -0.2613042 | -1.02239764 | -0.0727912188 | 0.133992687 |
| missing | max_seen_score | 0.321517736 | -0.511352003 | 0.435112298 | 0.910776973 |
| missing | max_missing_score | 0.774387956 | 0.49447304 | 0.836530089 | 0.953770816 |
| missing | difference | -0.45287019 | -1.37802744 | -0.232781708 | 0.0823365524 |

Native/aligned winning fractions on true-seen: 0.384000003/0.615999997.
Native/aligned winning fractions on true-missing: 0.239750013/0.760249987.

Seed 2: worst owner rotation error=4.76837158e-07.
Centered component diagnostics: {"native_owner_seen_only": 0.6695, "aligned_missing_only": 0.263}
Winning group fractions: {"native_seen": 0.2457999885082245, "aligned_missing": 0.7542000114917755}
Prediction histograms: {"overall": [355, 751, 1600, 662, 853, 366, 1046, 1305, 2216, 846], "seen": [86, 156, 297, 141, 164, 72, 204, 277, 418, 185], "missing": [269, 595, 1303, 521, 689, 294, 842, 1028, 1798, 661]}

| Norm object | Minimum over clients | Mean of client means | Maximum over clients |
|---|---:|---:|---:|
| feature_raw | 3.23000264 | 14.429681 | 35.3865318 |
| feature_centered | 0.693916142 | 3.93937435 | 20.8810806 |
| owner_raw | 12.9331636 | 14.8224305 | 17.0199509 |
| owner_centered | 0.449405551 | 1.46590011 | 2.80694485 |
| global_raw | 13.5338898 | 15.3796177 | 16.9584179 |
| global_centered | 0.625900447 | 1.43741059 | 2.41054893 |

| True subset | Score | Mean | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|
| seen | max_seen_score | 0.511802793 | -0.332719028 | 0.723103762 | 0.950333416 |
| seen | max_missing_score | 0.78769356 | 0.520832479 | 0.846217871 | 0.953200638 |
| seen | difference | -0.275890827 | -1.16615212 | -0.0540702939 | 0.0974851996 |
| missing | max_seen_score | 0.379345179 | -0.53245151 | 0.585464537 | 0.928587914 |
| missing | max_missing_score | 0.79520905 | 0.541525185 | 0.854604363 | 0.952761829 |
| missing | difference | -0.415863842 | -1.41105914 | -0.147528142 | 0.0557446182 |

Native/aligned winning fractions on true-seen: 0.341500014/0.658499986.
Native/aligned winning fractions on true-missing: 0.221875012/0.778124988.
