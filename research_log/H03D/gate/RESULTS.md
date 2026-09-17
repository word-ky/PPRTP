# H03-D convexity audit

| Arm | Seen % | Missing % | All % | Macro % | CE | Fit % | Iter/eval |
|---|---:|---:|---:|---:|---:|---:|---|
| no_align_500 reference | 70.800000 | 0.000000 | 14.160000 | 14.160000 | 1.50203121e-08 | 100.000000 | 320/360 |
| paired_500 reproduced | 34.600000 | 25.750000 | 27.520000 | 27.520000 | 0.628464222 | 79.100001 | 500/518 |
| paired_2000 | 31.100000 | 23.550000 | 25.060000 | 25.060001 | 9.20881931e-08 | 100.000000 | 1651/1743 |

grad_inf=9.56242658e-08;grad_l2=6.13980319e-07;weight_norm=952430.812;bias_norm=19294.3613.
q2000=0.719633305;delta2000=23.550000pp;O10=32.725%. Branch: formal persistence gate closes positively.

| Client | Support correct/count | Accuracy % |
|---|---|---:|
| 0 | 200/200 | 100.000000 |
| 1 | 200/200 | 100.000000 |
| 2 | 200/200 | 100.000000 |
| 3 | 200/200 | 100.000000 |
| 4 | 200/200 | 100.000000 |
| 5 | 200/200 | 100.000000 |
| 6 | 200/200 | 100.000000 |
| 7 | 200/200 | 100.000000 |
| 8 | 200/200 | 100.000000 |
| 9 | 200/200 | 100.000000 |

Alignment residuals, orthogonality and every transform hash exactly reproduce H03-C (full arrays in raw final.json). All online records, saved indices, state/RNG/modes match.
