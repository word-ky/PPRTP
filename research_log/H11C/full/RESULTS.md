# H11-C full-data pair-breaking control

| Seed | Readout | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | paired_h07 | 41.860000 | 19.166250 | 23.705000 | 23.705000 | 10 |
| 0 | pair_broken_h07 | 54.594999 | 6.517500 | 16.133000 | 16.133000 | 10 |
| 0 | native_control | 80.690000 | 0.000000 | 16.138000 | 16.138000 | 10 |
| 1 | paired_h07 | 40.090000 | 19.273750 | 23.437000 | 23.437000 | 10 |
| 1 | pair_broken_h07 | 55.330000 | 5.718750 | 15.641000 | 15.641000 | 10 |
| 1 | native_control | 87.210000 | 0.000000 | 17.442000 | 17.442000 | 10 |
| 2 | paired_h07 | 38.620000 | 18.205000 | 22.288000 | 22.288000 | 10 |
| 2 | pair_broken_h07 | 52.160000 | 5.928750 | 15.175000 | 15.175000 | 10 |
| 2 | native_control | 85.745000 | 0.000000 | 17.149000 | 17.149000 | 10 |

Frozen verdict: 3/3 PASS: correct correspondence remains causally important at full-data scale.
Missing gaps (pp): [12.648750003427267, 13.554999884217978, 12.276250086724758].

All three paired/native readouts reproduce historical H11-A/B exactly excluding timing; all ten online rounds, split, initial state and final hashes match. Raw local means/counts and model/server/prototype state match across readouts; RNG, existing gradients and modes unchanged. Anchors and test labels are excluded from transform fitting.

Legacy permutation seeds314160..314168 unchanged; exact fixed-point counts [1,0,1,2,1,0,2,3,1]. Client8 keeps3/256 (1.171875%) rows, per pre-performance lead amendment f9c70a2. All permutations are bijective and preserve anchor-feature multisets bitwise. Full permutations, SHA receipts and feature hashes are in final.json and verification.json. No tuning or alternate permutations.

Only FedGH was rerun because archived checkpoints held client0/server head, not all10clients. These are final-state diagnostic readouts, not online-training or communication-efficiency evidence. Aggregate class coverage does not imply every client predicts all10classes.

## Seed 0
Missing gap 12.648750 pp; all gap 7.572000 pp; gate >=8 pp: True.
Per-client class coverage: {"paired_h07": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "pair_broken_h07": [10, 9, 9, 9, 9, 10, 9, 10, 10, 8], "native_control": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 31.294674 | 99.444523 |
| 2 | 40.677062 | 94.153925 |
| 3 | 36.559596 | 95.980241 |
| 4 | 38.788906 | 82.386444 |
| 5 | 70.757395 | 106.249339 |
| 6 | 45.298712 | 93.579471 |
| 7 | 45.226393 | 85.050226 |
| 8 | 55.010641 | 89.591946 |
| 9 | 60.533867 | 104.805921 |

## Seed 1
Missing gap 13.555000 pp; all gap 7.796000 pp; gate >=8 pp: True.
Per-client class coverage: {"paired_h07": [10, 7, 10, 9, 10, 8, 10, 10, 7, 10], "pair_broken_h07": [9, 7, 7, 8, 10, 8, 9, 9, 10, 9], "native_control": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 71.272909 | 103.302928 |
| 2 | 53.757984 | 117.026972 |
| 3 | 66.683190 | 101.482861 |
| 4 | 64.081636 | 122.918535 |
| 5 | 72.202225 | 102.628360 |
| 6 | 57.863251 | 115.801440 |
| 7 | 59.519404 | 118.931810 |
| 8 | 74.909130 | 109.041178 |
| 9 | 42.066235 | 125.579670 |

## Seed 2
Missing gap 12.276250 pp; all gap 7.113000 pp; gate >=8 pp: True.
Per-client class coverage: {"paired_h07": [10, 10, 10, 10, 10, 6, 8, 10, 10, 10], "pair_broken_h07": [8, 8, 7, 8, 8, 9, 7, 8, 8, 10], "native_control": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}.

| Client | Paired centered residual | Broken centered residual |
|---|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 1 | 53.375529 | 104.605196 |
| 2 | 65.302869 | 106.637081 |
| 3 | 45.532348 | 99.190690 |
| 4 | 50.743736 | 104.890860 |
| 5 | 43.307759 | 77.049861 |
| 6 | 58.600684 | 94.507522 |
| 7 | 45.215808 | 101.824424 |
| 8 | 55.299126 | 90.594420 |
| 9 | 45.661083 | 102.374491 |
