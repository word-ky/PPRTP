# H11-B full-data cross-seed replication

| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |
|---|---|---:|---:|---:|---:|---:|
| 0 | local | 81.889999 | 0.000000 | 16.378000 | 16.378000 | 10 |
| 0 | fedproto | 82.595000 | 0.003750 | 16.522000 | 16.522000 | 10 |
| 0 | fedgh | 77.420000 | 0.000000 | 15.484000 | 15.484000 | 10 |
| 0 | pprtp | 41.860000 | 19.166250 | 23.705000 | 23.705000 | 10 |
| 0 | native | 80.690000 | 0.000000 | 16.138000 | 16.138000 | 10 |
| 1 | local | 87.605000 | 0.000000 | 17.521000 | 17.521000 | 10 |
| 1 | fedproto | 87.824999 | 0.000000 | 17.565000 | 17.565000 | 10 |
| 1 | fedgh | 85.890000 | 0.000000 | 17.178000 | 17.178000 | 10 |
| 1 | pprtp | 40.090000 | 19.273750 | 23.437000 | 23.437000 | 10 |
| 1 | native | 87.210000 | 0.000000 | 17.442000 | 17.442000 | 10 |
| 2 | local | 85.369999 | 0.000000 | 17.074000 | 17.074000 | 10 |
| 2 | fedproto | 85.860000 | 0.000000 | 17.172000 | 17.172000 | 10 |
| 2 | fedgh | 82.235000 | 0.000000 | 16.447000 | 16.447000 | 10 |
| 2 | pprtp | 38.620000 | 18.205000 | 22.288000 | 22.288000 | 10 |
| 2 | native | 85.745000 | 0.000000 | 17.149000 | 17.149000 | 10 |

Mean +/- sample standard deviation (n=3, ddof=1), percentage points:

| Arm | Seen | Missing | All |
|---|---:|---:|---:|
| local | 84.954999 +/- 2.880013 | 0.000000 +/- 0.000000 | 16.991000 +/- 0.576003 |
| fedproto | 85.426666 +/- 2.641791 | 0.001250 +/- 0.002165 | 17.086333 +/- 0.526751 |
| fedgh | 81.848333 +/- 4.248219 | 0.000000 +/- 0.000000 | 16.369667 +/- 0.849644 |
| pprtp | 40.190000 +/- 1.622313 | 18.881667 +/- 0.588470 | 23.143333 +/- 0.752763 |
| native | 84.548333 +/- 3.420761 | 0.000000 +/- 0.000000 | 16.909667 +/- 0.684152 |

Frozen verdict: 3/3 STRONG: full-data replication accepted; await lead.
Per-seed strong: [True, True, True]
Per-seed correspondence missing gain (pp): [19.166250005364418, 19.273749887943268, 18.205000087618828]

All anchors exactly identical to historical H11-A seed0; ownership inherited from each historical seed. Detailed per-seed gates, provenance, classwise counts, residuals, communication, forward costs, runtime and state isolation are in seed1/seed2 RESULTS.md and raw final.json. Seed0 is unchanged committed H11-A, not rerun. Same deployed readouts; no selection or tuning.
