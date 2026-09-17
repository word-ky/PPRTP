# H02-C oracle diagnostic

Calibration hash `283003b4219d2e4278982b622a225d59c62ef7376c182d3ed4de3574a0a071da`. Official training split only,100/class; no client-training overlap. Seed314159.
Seed0 only; all10 online rounds exactly reproduce H02-A. Oracle state and CPU/CUDA RNG unchanged.
Oracle labels are analysis-only and unavailable to online FL. Settings fixed before observing results.

| Round | Readout | Seen % | Missing % | All % | Macro % |
|---|---|---:|---:|---:|---:|
| 1 | oracle_individual | 31.40000 | 32.20000 | 32.04000 | 32.04000 |
| 1 | oracle_shared | 31.60000 | 32.10000 | 32.00000 | 32.00000 |
| 2 | oracle_individual | 31.75000 | 31.95000 | 31.91000 | 31.91000 |
| 2 | oracle_shared | 31.10000 | 31.45000 | 31.38000 | 31.38000 |
| 10 | oracle_individual | 33.60000 | 34.26250 | 34.13000 | 34.13000 |
| 10 | oracle_shared | 31.95000 | 32.72500 | 32.57000 | 32.57000 |

| Round | Individual fit accuracy mean/min/max | Individual fit CE mean/min/max | Shared fit accuracy / CE | Owner cosine mean/min/max |
|---|---|---|---|---|
| 1 | 1 / 1 / 1 | 0.0001135619 / 6.67572e-09 / 0.0006134772 | 0.9525 / 0.140831 | 0.9890567 / 0.9879805 / 0.9915545 |
| 2 | 0.9955 / 0.9850001 / 1 | 0.02533511 / 1.816852e-06 / 0.08138768 | 0.9284 / 0.1951831 | 0.9074156 / 0.885563 / 0.924862 |
| 10 | 0.6541 / 0.627 / 0.693 | 1.001857 / 0.8478163 / 1.086116 | 0.5331 / 1.329544 | 0.6061584 / 0.4978905 / 0.7231926 |

H02-B20-mean probe missing reference: round2=.0125%, round10=0%; reused, not rerun.
Raw rounds.jsonl contains all per-client metrics, before/after fits, head norms/hashes, LBFGS counts and state receipts.
