# H01-B results

Seeds: [0]. Accuracy percentages; mean ± sample standard deviation across seeds.
CIFAR-10 subset: 2000 train examples, 1000 official test examples; 10 clients, K=2.
Frozen configuration: 10 rounds, 1 epoch/round, SGD .01, lambda 1, scale 10.
Source: `52a6c8f3d167b7b7827386221238e1576e1eaea8`.

All seeds passed exact round-1 client/prototype equality and matched initial weights/splits.
All three round-2 prototype banks differ: identical-looking accuracies do not imply an inactive loss.

| Round | Training | Readout | Seen | Missing | All | Macro |
|---|---|---|---:|---:|---:|---:|
| 2 | local | head | 53.15 ± 0.00 | 0.00 ± 0.00 | 10.63 ± 0.00 | 10.63 ± 0.00 |
| 2 | local | cosine | 63.60 ± 0.00 | 0.00 ± 0.00 | 12.72 ± 0.00 | 12.72 ± 0.00 |
| 2 | local | l2 | 60.45 ± 0.00 | 0.02 ± 0.00 | 12.11 ± 0.00 | 12.11 ± 0.00 |
| 2 | fedproto | head | 53.15 ± 0.00 | 0.00 ± 0.00 | 10.63 ± 0.00 | 10.63 ± 0.00 |
| 2 | fedproto | cosine | 63.65 ± 0.00 | 0.00 ± 0.00 | 12.73 ± 0.00 | 12.73 ± 0.00 |
| 2 | fedproto | l2 | 60.85 ± 0.00 | 0.02 ± 0.00 | 12.19 ± 0.00 | 12.19 ± 0.00 |
| 2 | gpc | head | 52.90 ± 0.00 | 0.00 ± 0.00 | 10.58 ± 0.00 | 10.58 ± 0.00 |
| 2 | gpc | cosine | 61.15 ± 0.00 | 0.00 ± 0.00 | 12.23 ± 0.00 | 12.23 ± 0.00 |
| 2 | gpc | l2 | 60.00 ± 0.00 | 0.00 ± 0.00 | 12.00 ± 0.00 | 12.00 ± 0.00 |
| 10 | local | head | 61.15 ± 0.00 | 0.00 ± 0.00 | 12.23 ± 0.00 | 12.23 ± 0.00 |
| 10 | local | cosine | 59.20 ± 0.00 | 0.00 ± 0.00 | 11.84 ± 0.00 | 11.84 ± 0.00 |
| 10 | local | l2 | 63.70 ± 0.00 | 0.00 ± 0.00 | 12.74 ± 0.00 | 12.74 ± 0.00 |
| 10 | fedproto | head | 60.90 ± 0.00 | 0.00 ± 0.00 | 12.18 ± 0.00 | 12.18 ± 0.00 |
| 10 | fedproto | cosine | 60.60 ± 0.00 | 0.00 ± 0.00 | 12.12 ± 0.00 | 12.12 ± 0.00 |
| 10 | fedproto | l2 | 65.35 ± 0.00 | 0.00 ± 0.00 | 13.07 ± 0.00 | 13.07 ± 0.00 |
| 10 | gpc | head | 62.10 ± 0.00 | 0.00 ± 0.00 | 12.42 ± 0.00 | 12.42 ± 0.00 |
| 10 | gpc | cosine | 64.40 ± 0.00 | 0.00 ± 0.00 | 12.88 ± 0.00 | 12.88 ± 0.00 |
| 10 | gpc | l2 | 65.65 ± 0.00 | 0.00 ± 0.00 | 13.13 ± 0.00 | 13.13 ± 0.00 |

## First active knowledge round: client 0, first batch

| Seed | Training | Local grad norm | Lambda-scaled knowledge norm | Ratio | Missing probability |
|---|---|---:|---:|---:|---:|
| 0 | fedproto | 1.21295 | 0.00947862 | 0.00781455 | 0.71697 |
| 0 | gpc | 1.21295 | 5.75848 | 4.74752 | 0.71697 |

## Runtime and prototype norms

| Seed | Training | Training/evaluation seconds | Prototype norm min/max over rounds |
|---|---|---:|---:|
| 0 | local | 7.96 | 0.880059 / 11.3651 |
| 0 | fedproto | 6.32 | 0.880059 / 9.89679 |
| 0 | gpc | 7.61 | 0.880059 / 11.5657 |

Local head is the local-only deployment result. Local cosine/L2 are posthoc prototype probes requiring communication.
Primary injection contrast compares GPC and FedProto under the SAME cosine readout.
Macro equals all-class accuracy here because the test set is class-balanced; client metrics are averaged equally.
Common payload per round: 40,960 upload vector bytes + 160 count bytes; 204,800 download vector bytes.
No serialization or class-ID overhead included. All ten prototypes valid after round 1; round 1 training has no knowledge loss.
Per-client counts, metrics, hashes, loss traces and cosine matrices are retained in receipts/.
No claim of convergence or benchmark-scale performance follows from this short subset run.
