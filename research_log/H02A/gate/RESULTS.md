# H02-A FedGH control

All historical pairing, server update, base preservation, broadcast, and persistence checks passed.
Server SGD lr=.01, no momentum/decay; batch size 1; exactly one pass per round in client ID then class ID order.
Full model checkpoints remain in the remote run directory. These receipts retain JSON and logs.
Accuracies: percent, mean ± sample SD.

| Round | Readout | Seen | Missing | All | Macro |
|---|---|---:|---:|---:|---:|
| 2 | local_head_pre_server | 52.8500 ± 0.0000 | 0.0000 ± 0.0000 | 10.5700 ± 0.0000 | 10.5700 ± 0.0000 |
| 2 | global_head_post_server | 52.9500 ± 0.0000 | 0.0000 ± 0.0000 | 10.5900 ± 0.0000 | 10.5900 ± 0.0000 |
| 2 | cosine | 63.5000 ± 0.0000 | 0.0125 ± 0.0000 | 12.7100 ± 0.0000 | 12.7100 ± 0.0000 |
| 2 | l2 | 62.2500 ± 0.0000 | 0.0500 ± 0.0000 | 12.4900 ± 0.0000 | 12.4900 ± 0.0000 |

| Seed | Round | Server CE before / after | Server accuracy before / after | Owner cosine mean / min / max |
|---|---|---|---|---|
| 0 | 1 | 2.275302 / 2.274621 | 0.2000 / 0.2000 | 0.989057 / 0.987981 / 0.991554 |
| 0 | 2 | 2.182737 / 2.179087 | 0.5500 / 0.6000 | 0.907416 / 0.885563 / 0.924862 |

All parameter hashes, deterministic sample orders, per-client readouts, payload sizes and model metadata are in artifacts/experiment.
Comparison to H01-D is a mechanism/method control: FedGH learns a head and retains separate owner means; it is not a same-information causal ablation.
