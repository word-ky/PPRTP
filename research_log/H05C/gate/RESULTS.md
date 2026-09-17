# H05-C structural null audit

| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rel255_paired_helmert_zscore | 56.199999 | 11.787500 | 20.670000 | 20.670000 | 68.700004 | 1.07979739 | 0.00125634123 | 0.0112194698 | 762.251526/0.915916264 | 2000/2114 |
| rel255_broken_helmert_zscore | 60.249999 | 4.887500 | 15.960000 | 15.960000 | 76.000005 | 0.853311598 | 0.00191304844 | 0.0230582841 | 819.577148/0.627656102 | 2000/2082 |

R255=11.787500%,S255=4.887500%,q_rel_255=0.536100057,delta_rel_255=6.900000pp. Frozen branch: still optimization-unresolved.

| Arm | Data | Max absolute row sum | RMS row sum | Ones-direction energy / total energy |
|---|---|---:|---:|---:|
| rel255_paired_helmert_zscore | support | 0.000535860658 | 9.92043353e-05 | 9.36946282e-16 |
| rel255_paired_helmert_zscore | test | 0.0019326508 | 0.000241749009 | 5.31122291e-15 |
| rel255_broken_helmert_zscore | support | 0.000535860658 | 9.92043353e-05 | 9.36946282e-16 |
| rel255_broken_helmert_zscore | test | 0.0019326508 | 0.000241749009 | 5.31122291e-15 |

| Arm | Stage | Epsilon rule | Rank | smax | Smallest above tolerance | Tolerance | Nonzero condition |
|---|---|---|---:|---:|---:|---:|---:|
| rel255_paired_helmert_zscore | raw256 | float64 | 256 | 8651.9041 | 7.56336555e-05 | 3.84221726e-09 | 114392251 |
| rel255_paired_helmert_zscore | raw256 | float32 | 72 | 8651.9041 | 2.08086298 | 2.06277468 | 4157.84422 |
| rel255_paired_helmert_zscore | Helmert255 | float64 | 255 | 8651.90412 | 0.221877624 | 3.84221726e-09 | 38994.0362 |
| rel255_paired_helmert_zscore | Helmert255 | float32 | 72 | 8651.90412 | 2.08086325 | 2.06277469 | 4157.84368 |
| rel255_paired_helmert_zscore | Helmert255+zscore | float64 | 255 | 617.183665 | 0.0107741077 | 2.74084606e-10 | 57283.9705 |
| rel255_paired_helmert_zscore | Helmert255+zscore | float32 | 109 | 617.183665 | 0.147470375 | 0.147148052 | 4185.13661 |
| rel255_broken_helmert_zscore | raw256 | float64 | 256 | 4421.63347 | 7.60112406e-05 | 1.96359971e-09 | 58170784.1 |
| rel255_broken_helmert_zscore | raw256 | float32 | 220 | 4421.63347 | 1.0663845 | 1.05419957 | 4146.37823 |
| rel255_broken_helmert_zscore | Helmert255 | float64 | 255 | 4421.63347 | 0.648275172 | 1.96359972e-09 | 6820.61209 |
| rel255_broken_helmert_zscore | Helmert255 | float32 | 220 | 4421.63347 | 1.066386 | 1.05419957 | 4146.3724 |
| rel255_broken_helmert_zscore | Helmert255+zscore | float64 | 255 | 325.768851 | 0.0475532008 | 1.44670432e-10 | 6850.61878 |
| rel255_broken_helmert_zscore | Helmert255+zscore | float32 | 226 | 325.768851 | 0.0778047342 | 0.0776693465 | 4187.00551 |

Tolerance=max(matrix_shape)*eps(dtype)*smax;SVD in double on actual float32 input. No data-dependent projection/truncation. Fixed Helmert basis built by closed form in double and cast to float32;training remains float32.

rel255_paired_helmert_zscore: basis SHA256=b0cf8c0a162fd645fd1f44dd476ac578ab783488a83433efc79cd85132caeefd;support-only conditioning SHA256=23cf357c139b4b334e0183756967808b1c8a832209e3fe3069437003a4f3bd2a.

rel255_broken_helmert_zscore: basis SHA256=b0cf8c0a162fd645fd1f44dd476ac578ab783488a83433efc79cd85132caeefd;support-only conditioning SHA256=5be5a326738c13976d48066cb11fdd728722eaf9f267d7d17d7e02ea6bdf85b9.
