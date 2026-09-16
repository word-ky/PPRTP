# H01-B results

Seeds: [0, 1, 2]. Accuracy percentages; mean ± sample standard deviation across seeds.
CIFAR-10 subset: 2000 train examples, 1000 official test examples; 10 clients, K=2.
Frozen configuration: 10 rounds, 1 epoch/round, SGD .01, lambda 1, scale 10.
Source: `52a6c8f3d167b7b7827386221238e1576e1eaea8`.

All seeds passed exact round-1 client/prototype equality and matched initial weights/splits.
All three round-2 prototype banks differ: identical-looking accuracies do not imply an inactive loss.

| Round | Training | Readout | Seen | Missing | All | Macro |
|---|---|---|---:|---:|---:|---:|
| 2 | local | head | 52.42 ± 1.06 | 0.00 ± 0.00 | 10.48 ± 0.21 | 10.48 ± 0.21 |
| 2 | local | cosine | 66.93 ± 4.81 | 0.00 ± 0.00 | 13.39 ± 0.96 | 13.39 ± 0.96 |
| 2 | local | l2 | 63.25 ± 3.31 | 0.02 ± 0.02 | 12.67 ± 0.65 | 12.67 ± 0.65 |
| 2 | fedproto | head | 52.43 ± 1.07 | 0.00 ± 0.00 | 10.49 ± 0.21 | 10.49 ± 0.21 |
| 2 | fedproto | cosine | 67.00 ± 4.80 | 0.00 ± 0.00 | 13.40 ± 0.96 | 13.40 ± 0.96 |
| 2 | fedproto | l2 | 63.50 ± 3.23 | 0.02 ± 0.02 | 12.72 ± 0.63 | 12.72 ± 0.63 |
| 2 | gpc | head | 53.50 ± 0.58 | 0.00 ± 0.00 | 10.70 ± 0.12 | 10.70 ± 0.12 |
| 2 | gpc | cosine | 65.97 ± 4.75 | 0.00 ± 0.00 | 13.19 ± 0.95 | 13.19 ± 0.95 |
| 2 | gpc | l2 | 64.33 ± 4.25 | 0.00 ± 0.00 | 12.87 ± 0.85 | 12.87 ± 0.85 |
| 10 | local | head | 63.83 ± 2.32 | 0.00 ± 0.00 | 12.77 ± 0.46 | 12.77 ± 0.46 |
| 10 | local | cosine | 66.55 ± 6.51 | 0.00 ± 0.00 | 13.31 ± 1.30 | 13.31 ± 1.30 |
| 10 | local | l2 | 68.62 ± 4.88 | 0.02 ± 0.04 | 13.74 ± 0.98 | 13.74 ± 0.98 |
| 10 | fedproto | head | 65.18 ± 3.79 | 0.00 ± 0.00 | 13.04 ± 0.76 | 13.04 ± 0.76 |
| 10 | fedproto | cosine | 66.85 ± 5.57 | 0.00 ± 0.00 | 13.37 ± 1.11 | 13.37 ± 1.11 |
| 10 | fedproto | l2 | 69.83 ± 4.77 | 0.00 ± 0.00 | 13.97 ± 0.95 | 13.97 ± 0.95 |
| 10 | gpc | head | 66.98 ± 4.43 | 0.00 ± 0.00 | 13.40 ± 0.89 | 13.40 ± 0.89 |
| 10 | gpc | cosine | 69.80 ± 4.93 | 0.00 ± 0.00 | 13.96 ± 0.99 | 13.96 ± 0.99 |
| 10 | gpc | l2 | 70.77 ± 4.89 | 0.00 ± 0.00 | 14.15 ± 0.98 | 14.15 ± 0.98 |

## First active knowledge round: client 0, first batch

| Seed | Training | Local grad norm | Lambda-scaled knowledge norm | Ratio | Missing probability |
|---|---|---:|---:|---:|---:|
| 0 | fedproto | 1.21295 | 0.00947862 | 0.00781455 | 0.71697 |
| 0 | gpc | 1.21295 | 5.75848 | 4.74752 | 0.71697 |
| 1 | fedproto | 1.1279 | 0.00800925 | 0.00710101 | 0.760207 |
| 1 | gpc | 1.1279 | 3.72995 | 3.30698 | 0.760207 |
| 2 | fedproto | 1.24102 | 0.0109992 | 0.00886303 | 0.750245 |
| 2 | gpc | 1.24102 | 3.79103 | 3.05477 | 0.750245 |

## Runtime and prototype norms

| Seed | Training | Training/evaluation seconds | Prototype norm min/max over rounds | Final off-diagonal cosine min/max |
|---|---|---:|---:|---:|
| 0 | local | 7.96 | 0.880059 / 11.3651 | 0.450613 / 0.846013 |
| 0 | fedproto | 6.32 | 0.880059 / 9.89679 | 0.453301 / 0.856308 |
| 0 | gpc | 7.61 | 0.880059 / 11.5657 | 0.214068 / 0.816565 |
| 1 | local | 6.83 | 0.874861 / 11.5211 | 0.445597 / 0.841194 |
| 1 | fedproto | 6.39 | 0.874861 / 10.0226 | 0.448483 / 0.852022 |
| 1 | gpc | 7.30 | 0.874861 / 10.9554 | 0.256183 / 0.82847 |
| 2 | local | 6.20 | 0.941744 / 13.0381 | 0.450015 / 0.8359 |
| 2 | fedproto | 7.24 | 0.941744 / 10.5157 | 0.463756 / 0.856244 |
| 2 | gpc | 7.05 | 0.941744 / 11.829 | 0.263785 / 0.801037 |

Local head is the local-only deployment result. Local cosine/L2 are posthoc prototype probes requiring communication.
Primary injection contrast compares GPC and FedProto under the SAME cosine readout.
Macro equals all-class accuracy here because the test set is class-balanced; client metrics are averaged equally.
Common payload per round: 40,960 upload vector bytes + 160 count bytes; 204,800 download vector bytes.
No serialization or class-ID overhead included. All ten prototypes valid after round 1; round 1 training has no knowledge loss.
Per-client counts, metrics, hashes, loss traces and cosine matrices are retained in receipts/.
No claim of convergence or benchmark-scale performance follows from this short subset run.
