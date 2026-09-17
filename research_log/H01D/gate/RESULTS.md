# H01-D gate

Source `ac57d8666b231e5bcc2b362012b7806b81afe9ca`. Seeds [0]. Seed0 round2 seen/all ratio 1.00001237; gate True.
Exact first-round pairing and FedProto historical model/prototype hash parity passed.

Accuracies in percent, mean ± sample SD. Frozen lambdas: FedProto 1, all .002, seen .03498.

| Round | Mode | Readout | Seen | Missing | All | Macro |
|---|---|---|---:|---:|---:|---:|
| 2 | fedproto | cosine | 63.6500 ± 0.0000 | 0.0000 ± 0.0000 | 12.7300 ± 0.0000 | 12.7300 ± 0.0000 |
| 2 | fedproto | head | 53.1500 ± 0.0000 | 0.0000 ± 0.0000 | 10.6300 ± 0.0000 | 10.6300 ± 0.0000 |
| 2 | fedproto | l2 | 60.8500 ± 0.0000 | 0.0250 ± 0.0000 | 12.1900 ± 0.0000 | 12.1900 ± 0.0000 |
| 2 | gpc_all_match | cosine | 63.6500 ± 0.0000 | 0.0000 ± 0.0000 | 12.7300 ± 0.0000 | 12.7300 ± 0.0000 |
| 2 | gpc_all_match | head | 53.1500 ± 0.0000 | 0.0000 ± 0.0000 | 10.6300 ± 0.0000 | 10.6300 ± 0.0000 |
| 2 | gpc_all_match | l2 | 60.5500 ± 0.0000 | 0.0250 ± 0.0000 | 12.1300 ± 0.0000 | 12.1300 ± 0.0000 |
| 2 | gpc_seen_match | cosine | 63.7000 ± 0.0000 | 0.0000 ± 0.0000 | 12.7400 ± 0.0000 | 12.7400 ± 0.0000 |
| 2 | gpc_seen_match | head | 53.0500 ± 0.0000 | 0.0000 ± 0.0000 | 10.6100 ± 0.0000 | 10.6100 ± 0.0000 |
| 2 | gpc_seen_match | l2 | 60.7500 ± 0.0000 | 0.0250 ± 0.0000 | 12.1700 ± 0.0000 | 12.1700 ± 0.0000 |

## Same feature / same bank gradient direction at seed0 round2

| Arm | Feature-gradient cosine | Unscaled all/seen feature-gradient norm ratio | Base local norm | Base scaled knowledge norm |
|---|---:|---:|---:|---:|
| fedproto | 0.382520884 | 2.33901405 | 1.21294522 | 0.00947861932 |
| gpc_all_match | 0.382520884 | 2.33901405 | 1.21294522 | 0.0115169547 |
| gpc_seen_match | 0.382520884 | 2.33901405 | 1.21294522 | 0.0115170972 |

## Strength ratio seen/all across respective training arms

| Seed | Round | Scaled base-gradient ratio |
|---|---|---:|
| 0 | 2 | 1.00001237 |

## Owner prototype compatibility (before aggregation)

| Seed | Round | Arm | Mean | Min | Max |
|---|---|---|---:|---:|---:|
| 0 | 1 | fedproto | 0.989056653 | 0.987980545 | 0.991554499 |
| 0 | 1 | gpc_all_match | 0.989056653 | 0.987980545 | 0.991554499 |
| 0 | 1 | gpc_seen_match | 0.989056653 | 0.987980545 | 0.991554499 |
| 0 | 2 | fedproto | 0.908854854 | 0.887295842 | 0.926129818 |
| 0 | 2 | gpc_all_match | 0.908422846 | 0.886698127 | 0.925801039 |
| 0 | 2 | gpc_seen_match | 0.908658224 | 0.887051284 | 0.926072001 |

No retuning after the seed0 gate. Later strength ratios compare diverged arm states, not identical tensors.
Same-tensor direction uses dL/dz; strength calibration uses gradients into base parameters. These norms differ.
Finite prototype norms do not establish convergence. This is a small CIFAR-10 subset experiment.
