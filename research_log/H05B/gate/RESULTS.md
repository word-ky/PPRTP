# H05-B invertible conditioning

| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE before/after | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| rel256_paired_zscore | 56.149999 | 12.012500 | 20.840000 | 20.840000 | 68.650001 | 2.30258393/1.07599366 | 0.00222687563 | 0.0183848478 | 774.708557/0.930271804 | 2000/2133 |
| rel256_broken_zscore | 59.850000 | 4.925000 | 15.910000 | 15.910000 | 76.450002 | 2.30258393/0.855554402 | 0.00147384172 | 0.0186458286 | 810.236938/0.686846137 | 2000/2097 |

Rz=12.012500%,Sz=4.925000%,q_rel_z=0.546333148,delta_rel_z=7.087500pp. Frozen branch: optimization remains unresolved after fixed conditioning.

| Arm | Matrix | Std min/median/max | Singular max/min-nonzero | Rank | Tolerance | Condition nonzero | Abs max | RMS |
|---|---|---|---|---:|---:|---:|---:|---:|
| rel256_paired_zscore | before | 1.57717165/9.1773968/35.8514184 | 8651.9041/7.56336555e-05 | 256 | 3.84221726e-09 | 114392251 | 176.526276 | 12.6599894 |
| rel256_paired_zscore | after | 0.999999883/0.999999973/1.00000008 | 618.458245/5.648753e-06 | 256 | 2.74650634e-10 | 109485801 | 7.08438349 | 0.999999971 |
| rel256_broken_zscore | before | 5.65783097/12.1932904/23.4394305 | 4421.63347/7.60112406e-05 | 256 | 1.96359971e-09 | 58170784.1 | 176.526276 | 12.6599894 |
| rel256_broken_zscore | after | 0.999999905/0.999999975/1.00000007 | 325.865825/5.41460697e-06 | 256 | 1.44713497e-10 | 60182729.2 | 9.34680557 | 0.999999977 |

rel256_paired_zscore conditioning statistics SHA256: 9844754bca15db5922bb7ae2d96c0f7e978f890d7894e033e80f26dcf6385651; raw support SHA256: d05243ebd179d5e2acf8b319b71557d643348495b7ca4ba49bee7998428d0273.


rel256_broken_zscore conditioning statistics SHA256: 12cc7f9c35da131a7d6297b6761cc2fcc9dd81ffbb6bc68a24e4f97445445611; raw support SHA256: cdee85fe25a34a1f353494907908c4ecb19712ab4fdde527b95ccd9ec342508a.

SVD diagnostic tolerance: max(matrix_shape)*eps64*smax. SVD uses the actual uncentered matrix before/after zscore, float64 diagnostics only. Std diagnostics are recomputed in double (median=0.5 quantile); actual mean/std and inputs remain float32. Full mean/std vectors in raw JSON; no labels or test samples used for statistics.
