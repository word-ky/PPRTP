# H01-D full result

Source `ac57d8666b231e5bcc2b362012b7806b81afe9ca`. Seeds [0, 1, 2]. Seed0 round2 seen/all ratio 1.00001237; gate True.
Exact first-round pairing and FedProto historical model/prototype hash parity passed.

Accuracies in percent, mean ± sample SD. Frozen lambdas: FedProto 1, all .002, seen .03498.

| Round | Mode | Readout | Seen | Missing | All | Macro |
|---|---|---|---:|---:|---:|---:|
| 2 | fedproto | cosine | 67.0000 ± 4.8008 | 0.0000 ± 0.0000 | 13.4000 ± 0.9602 | 13.4000 ± 0.9602 |
| 2 | fedproto | head | 52.4333 ± 1.0728 | 0.0000 ± 0.0000 | 10.4867 ± 0.2146 | 10.4867 ± 0.2146 |
| 2 | fedproto | l2 | 63.5000 ± 3.2315 | 0.0208 ± 0.0191 | 12.7167 ± 0.6337 | 12.7167 ± 0.6337 |
| 2 | gpc_all_match | cosine | 66.9667 ± 4.7435 | 0.0000 ± 0.0000 | 13.3933 ± 0.9487 | 13.3933 ± 0.9487 |
| 2 | gpc_all_match | head | 52.4167 ± 1.0611 | 0.0000 ± 0.0000 | 10.4833 ± 0.2122 | 10.4833 ± 0.2122 |
| 2 | gpc_all_match | l2 | 63.3167 ± 3.2868 | 0.0208 ± 0.0191 | 12.6800 ± 0.6451 | 12.6800 ± 0.6451 |
| 2 | gpc_seen_match | cosine | 67.1500 ± 4.7752 | 0.0000 ± 0.0000 | 13.4300 ± 0.9550 | 13.4300 ± 0.9550 |
| 2 | gpc_seen_match | head | 52.4333 ± 0.9412 | 0.0000 ± 0.0000 | 10.4867 ± 0.1882 | 10.4867 ± 0.1882 |
| 2 | gpc_seen_match | l2 | 63.5833 ± 3.2297 | 0.0208 ± 0.0191 | 12.7333 ± 0.6341 | 12.7333 ± 0.6341 |
| 10 | fedproto | cosine | 66.8500 ± 5.5725 | 0.0000 ± 0.0000 | 13.3700 ± 1.1145 | 13.3700 ± 1.1145 |
| 10 | fedproto | head | 65.1833 ± 3.7896 | 0.0000 ± 0.0000 | 13.0367 ± 0.7579 | 13.0367 ± 0.7579 |
| 10 | fedproto | l2 | 69.8333 ± 4.7724 | 0.0000 ± 0.0000 | 13.9667 ± 0.9545 | 13.9667 ± 0.9545 |
| 10 | gpc_all_match | cosine | 66.6667 ± 6.4161 | 0.0000 ± 0.0000 | 13.3333 ± 1.2832 | 13.3333 ± 1.2832 |
| 10 | gpc_all_match | head | 63.8500 ± 2.2517 | 0.0000 ± 0.0000 | 12.7700 ± 0.4503 | 12.7700 ± 0.4503 |
| 10 | gpc_all_match | l2 | 68.6333 ± 4.9760 | 0.0208 ± 0.0361 | 13.7433 ± 0.9962 | 13.7433 ± 0.9962 |
| 10 | gpc_seen_match | cosine | 67.0500 ± 5.6340 | 0.0000 ± 0.0000 | 13.4100 ± 1.1268 | 13.4100 ± 1.1268 |
| 10 | gpc_seen_match | head | 64.1000 ± 2.6058 | 0.0000 ± 0.0000 | 12.8200 ± 0.5212 | 12.8200 ± 0.5212 |
| 10 | gpc_seen_match | l2 | 68.8167 ± 4.9250 | 0.0250 ± 0.0433 | 13.7833 ± 0.9855 | 13.7833 ± 0.9855 |

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
| 0 | 5 | 11.1710108 |
| 0 | 10 | 10.9947028 |
| 1 | 2 | 5.64879324 |
| 1 | 5 | 18.7095172 |
| 1 | 10 | 18.0870266 |
| 2 | 2 | 3.13287649 |
| 2 | 5 | 14.8852817 |
| 2 | 10 | 16.9231845 |

## Owner prototype compatibility (before aggregation)

| Seed | Round | Arm | Mean | Min | Max |
|---|---|---|---:|---:|---:|
| 0 | 1 | fedproto | 0.989056653 | 0.987980545 | 0.991554499 |
| 0 | 1 | gpc_all_match | 0.989056653 | 0.987980545 | 0.991554499 |
| 0 | 1 | gpc_seen_match | 0.989056653 | 0.987980545 | 0.991554499 |
| 0 | 2 | fedproto | 0.908854854 | 0.887295842 | 0.926129818 |
| 0 | 2 | gpc_all_match | 0.908422846 | 0.886698127 | 0.925801039 |
| 0 | 2 | gpc_seen_match | 0.908658224 | 0.887051284 | 0.926072001 |
| 0 | 5 | fedproto | 0.707848758 | 0.640716612 | 0.77817452 |
| 0 | 5 | gpc_all_match | 0.703667367 | 0.637091517 | 0.773131728 |
| 0 | 5 | gpc_seen_match | 0.704649442 | 0.643210649 | 0.773630738 |
| 0 | 10 | fedproto | 0.731823397 | 0.680425346 | 0.79589057 |
| 0 | 10 | gpc_all_match | 0.691547805 | 0.63982439 | 0.759135246 |
| 0 | 10 | gpc_seen_match | 0.693239516 | 0.652355373 | 0.757708073 |
| 1 | 1 | fedproto | 0.988952053 | 0.986974359 | 0.991235614 |
| 1 | 1 | gpc_all_match | 0.988952053 | 0.986974359 | 0.991235614 |
| 1 | 1 | gpc_seen_match | 0.988952053 | 0.986974359 | 0.991235614 |
| 1 | 2 | fedproto | 0.907712317 | 0.874017656 | 0.922521472 |
| 1 | 2 | gpc_all_match | 0.907228327 | 0.873348236 | 0.922069669 |
| 1 | 2 | gpc_seen_match | 0.907673728 | 0.873594761 | 0.922635555 |
| 1 | 5 | fedproto | 0.705436963 | 0.670221627 | 0.724835932 |
| 1 | 5 | gpc_all_match | 0.700486028 | 0.667871833 | 0.718334794 |
| 1 | 5 | gpc_seen_match | 0.702285361 | 0.670864701 | 0.722470045 |
| 1 | 10 | fedproto | 0.744591564 | 0.705258489 | 0.759464681 |
| 1 | 10 | gpc_all_match | 0.701139694 | 0.669846058 | 0.715718627 |
| 1 | 10 | gpc_seen_match | 0.704273444 | 0.672845721 | 0.715022504 |
| 2 | 1 | fedproto | 0.987619489 | 0.98543185 | 0.990453839 |
| 2 | 1 | gpc_all_match | 0.987619489 | 0.98543185 | 0.990453839 |
| 2 | 1 | gpc_seen_match | 0.987619489 | 0.98543185 | 0.990453839 |
| 2 | 2 | fedproto | 0.897950214 | 0.872335017 | 0.915448844 |
| 2 | 2 | gpc_all_match | 0.897408247 | 0.871658981 | 0.91506654 |
| 2 | 2 | gpc_seen_match | 0.897774613 | 0.872778356 | 0.915565312 |
| 2 | 5 | fedproto | 0.708520919 | 0.666886091 | 0.761092901 |
| 2 | 5 | gpc_all_match | 0.702815551 | 0.663109422 | 0.752051592 |
| 2 | 5 | gpc_seen_match | 0.704205346 | 0.666562319 | 0.749786496 |
| 2 | 10 | fedproto | 0.749566257 | 0.714256108 | 0.812705398 |
| 2 | 10 | gpc_all_match | 0.702331823 | 0.673236847 | 0.756780624 |
| 2 | 10 | gpc_seen_match | 0.705011433 | 0.680892825 | 0.753448963 |

No retuning after the seed0 gate. Later strength ratios compare diverged arm states, not identical tensors.
Same-tensor direction uses dL/dz; strength calibration uses gradients into base parameters. These norms differ.
Finite prototype norms do not establish convergence. This is a small CIFAR-10 subset experiment.
