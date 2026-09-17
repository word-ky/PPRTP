# H02-B server-head adequacy probe

Seed0 only. All online hashes and metrics exactly reproduce H02-A; probe changes no client or online server parameters.
LBFGS full batch, lr1 (default), max_iter100, strong_wolfe, tolerance_grad1e-9, tolerance_change1e-12; no regularizer.

| Round | Readout | Seen % | Missing % | All % | Macro % |
|---|---|---:|---:|---:|---:|
| 2 | global_head_post_server | 52.949999 | 0.000000 | 10.590000 | 10.590000 |
| 2 | probe_head_postfit | 64.350001 | 0.012500 | 12.880000 | 12.880001 |
| 10 | global_head_post_server | 50.000000 | 0.000000 | 10.000000 | 10.000000 |
| 10 | probe_head_postfit | 65.199999 | 0.000000 | 13.040000 | 13.040000 |

| Round | Probe CE before / after | Prototype acc before / after | LBFGS iterations / evaluations | Owner cosine mean / min / max |
|---|---|---|---|---|
| 1 | 2.27462101 / 0 | 0.200000 / 1.000000 | 35 / 56 | 0.989057 / 0.987981 / 0.991554 |
| 2 | 2.17908669 / 5.96046412e-09 | 0.600000 / 1.000000 | 33 / 56 | 0.907416 / 0.885563 / 0.924862 |
| 3 | 2.00717211 / 1.78813924e-08 | 0.500000 / 1.000000 | 34 / 54 | 0.796133 / 0.744153 / 0.832910 |
| 4 | 1.66261733 / 5.48361186e-07 | 0.550000 / 1.000000 | 31 / 49 | 0.705825 / 0.641574 / 0.754197 |
| 5 | 1.18170762 / 0 | 0.500000 / 1.000000 | 33 / 37 | 0.652488 / 0.578910 / 0.726869 |
| 6 | 0.889025688 / 1.66295411e-06 | 0.500000 / 1.000000 | 35 / 76 | 0.627835 / 0.557559 / 0.670165 |
| 7 | 0.820750058 / 0 | 0.500000 / 1.000000 | 33 / 50 | 0.618506 / 0.558795 / 0.667923 |
| 8 | 0.85314244 / 5.42400812e-07 | 0.500000 / 1.000000 | 34 / 77 | 0.613131 / 0.555550 / 0.669911 |
| 9 | 0.884667754 / 1.01327871e-07 | 0.500000 / 1.000000 | 32 / 50 | 0.609865 / 0.529731 / 0.695624 |
| 10 | 0.916397572 / 7.7486014e-08 | 0.500000 / 1.000000 | 34 / 77 | 0.606158 / 0.497890 / 0.723193 |

Per-client metrics, norms, hashes and side-effect receipts are preserved in raw JSON. All accuracy numbers use the same official test subset as H02-A.
No explicit optimizer termination reason is exposed by PyTorch; counts are logged. High prototype fit does not establish test generalization.
