# H02-E held-out owner-support diagnostic

Disjoint held-out indices hash: `2cd3cb1195bf1d68895cf0743e76d074036f479d579fab635771b4c853d59073`; 200/client,2000total,200/class; zero train/oracle/cross-client overlap.
All10round H02-A online records exact; probe preserves client/server/prototype state and CPU/CUDA RNG.

| Round | Seen % | Missing % | All % | Macro % | Fit CE before / after | Fit accuracy before / after | LBFGS iterations / evaluations |
|---|---:|---:|---:|---:|---|---|---|
| 2 | 71.750000 | 0.087500 | 14.420000 | 14.420000 | 2.30258393 / 5.30481259e-09 | 0.100000 / 1.000000 | 94 / 115 |
| 10 | 72.600001 | 0.000000 | 14.520000 | 14.520000 | 2.30258393 / 0.225471973 | 0.100000 / 0.899500 | 100 / 108 |

Round10 recovered oracle gap q_hold=0. Fixed references: reused-owner missing round2=.2125%,round10=0%; shared oracle round2=31.45%,round10=32.725%.
Raw JSON retains all per-client metrics, optimizer diagnostics, parameter hashes/norms and before/after state receipts. This is an analysis-only communication upper bound.
