# H01-C seed-0 strength gate

Gate: STOP; no tuning; later rounds/seeds NOT RUN. H01-B FedProto first-two-round hash parity: True.

| Mode | Local grad | Scaled knowledge grad | Knowledge/local | Knowledge/FedProto |
|---|---:|---:|---:|---:|
| fedproto | 1.21294522 | 0.00947861932 | 0.00781454891 | 1 |
| gpc_all_match | 1.21294522 | 0.0115169547 | 0.00949503295 | 1.21504561 |
| gpc_seen_match | 1.21294522 | 0.000658496167 | 0.00054289028 | 0.0694717389 |

Round 2 accuracy percentages:

| Mode | Readout | Seen | Missing | All | Macro |
|---|---|---:|---:|---:|---:|
| fedproto | head | 53.1500 | 0.0000 | 10.6300 | 10.6300 |
| fedproto | cosine | 63.6500 | 0.0000 | 12.7300 | 12.7300 |
| fedproto | l2 | 60.8500 | 0.0250 | 12.1900 | 12.1900 |
| gpc_all_match | head | 53.1500 | 0.0000 | 10.6300 | 10.6300 |
| gpc_all_match | cosine | 63.6500 | 0.0000 | 12.7300 | 12.7300 |
| gpc_all_match | l2 | 60.5500 | 0.0250 | 12.1300 | 12.1300 |
| gpc_seen_match | head | 53.1500 | 0.0000 | 10.6300 | 10.6300 |
| gpc_seen_match | cosine | 63.6500 | 0.0000 | 12.7300 | 12.7300 |
| gpc_seen_match | l2 | 60.5500 | 0.0250 | 12.1300 | 12.1300 |

Cross-client same-class prototype cosine, computed before aggregation:

| Round | Mode | Mean | Min | Max |
|---|---|---:|---:|---:|
| 1 | fedproto | 0.989056653 | 0.987980545 | 0.991554499 |
| 1 | gpc_all_match | 0.989056653 | 0.987980545 | 0.991554499 |
| 1 | gpc_seen_match | 0.989056653 | 0.987980545 | 0.991554499 |
| 2 | fedproto | 0.908854854 | 0.887295842 | 0.926129818 |
| 2 | gpc_all_match | 0.908422846 | 0.886698127 | 0.925801039 |
| 2 | gpc_seen_match | 0.908556002 | 0.886922359 | 0.925877571 |

Round-1 pairing checked across all three arms; exact same initial states, data and batch order.
This short gate is not a round-10 conclusion. Round-2 metrics are descriptive only.
