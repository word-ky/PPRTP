# H02-D owner-sample diagnostic

Exact frozen local indices hash: `5fbbd599df082e9f32630e9f65fc1e41aea4931f1ffe6957b022c2cfd56847a5`; 200/client,2000total,200/class; zero oracle overlap.
All10round H02-A online records exact; probe preserves client/server/prototype state and CPU/CUDA RNG.

| Round | Seen % | Missing % | All % | Macro % | Fit CE before / after | Fit accuracy before / after | LBFGS iterations / evaluations |
|---|---:|---:|---:|---:|---|---|---|
| 2 | 72.549999 | 0.212500 | 14.680000 | 14.680000 | 2.30258393 / 9.41750677e-09 | 0.100000 / 1.000000 | 86 / 104 |
| 10 | 71.899999 | 0.000000 | 14.380000 | 14.380000 | 2.30258393 / 0.216422439 | 0.100000 / 0.907500 | 100 / 106 |

Round10 recovered oracle gap q=0. Fixed references: mean missing round2=.0125%,round10=0%; shared oracle round2=31.45%,round10=32.725%.
Raw JSON retains all per-client metrics, optimizer diagnostics, parameter hashes/norms and before/after state receipts. This is an analysis-only communication upper bound.
