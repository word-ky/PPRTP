# H15-B FedTGP post-update / 100-cycle stress test

Primary endpoint cycle100; checkpoints10/25/50/100 fixed before run, no best-checkpoint selection.

| Cycle | Readout | Seen % | Missing % | All % | Macro % | Aggregate classes |
|---|---|---:|---:|---:|---:|---:|
| 10 | official nearest | 34.399999 | 0.000000 | 6.880000 | 6.880000 | 100 |
| 10 | local head | 33.365000 | 0.000000 | 6.673000 | 6.673000 | 100 |
| 25 | official nearest | 46.990000 | 0.000000 | 9.398000 | 9.398000 | 100 |
| 25 | local head | 46.215000 | 0.000000 | 9.243000 | 9.243000 | 100 |
| 50 | official nearest | 56.219999 | 0.000000 | 11.244000 | 11.244000 | 100 |
| 50 | local head | 56.115001 | 0.000000 | 11.223000 | 11.223000 | 100 |
| 100 | official nearest | 57.970000 | 0.000000 | 11.594000 | 11.594000 | 100 |
| 100 | local head | 58.290000 | 0.000000 | 11.658000 | 11.658000 | 100 |

| Frozen comparator | Seen % | Missing % | All % |
|---|---:|---:|---:|
| PPRTP seed0 (10 cycles) | 15.895000 | 9.345000 | 10.655000 |
| H15-A round-start FedTGP (10 cycles) | 31.950000 | 0.000000 | 6.390000 |

Frozen endpoint verdict: COMPETITIVE / NOVELTY WARNING.
PPRTP minus cycle100 FedTGP (pp): {"seen": -42.07499988377094, "missing": 9.345000013709068, "all": -0.9389998763799673}
Cycle10 post-update minus H15-A round-start (pp): {"seen": 2.4499991536140464, "missing": 0.0, "all": 0.49000002443790464}

Fixed checkpoint receipts (loss is mean over the final server inner epoch):
- Cycle10: {"metrics": {"l2": {"seen": 0.34399999380111695, "missing": 0.0, "all": 0.06879999935626983, "macro": 0.06880000010132789}, "head": {"seen": 0.3336500018835068, "missing": 0.0, "all": 0.06673000082373619, "macro": 0.06672999933362007}}, "per_client_coverage": {"l2": [20, 20, 20, 20, 20, 19, 20, 20, 20, 20], "head": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]}, "aggregate_coverage": {"l2": 100, "head": 100}, "last_epoch_server_loss": 5.83629825592041, "first_epoch_server_loss": 5.930757350921631, "client_steps": 15600, "server_steps": 7000, "semantic_vector_uplink_bytes": 4096000, "semantic_label_uplink_bytes": 16000, "global_proto_downlink_bytes_all_clients": 20480000, "wall_seconds": 185.18465447425842, "server_seconds": 28.85195779800415, "checkpoint_file": "checkpoint_cycle10.pt"}
- Cycle25: {"metrics": {"l2": {"seen": 0.46990000307559965, "missing": 0.0, "all": 0.09398000165820122, "macro": 0.09397999867796898}, "head": {"seen": 0.46215000152587893, "missing": 0.0, "all": 0.09243000000715255, "macro": 0.09242999777197838}}, "per_client_coverage": {"l2": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "head": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]}, "aggregate_coverage": {"l2": 100, "head": 100}, "last_epoch_server_loss": 6.1141758728027344, "first_epoch_server_loss": 6.295003128051758, "client_steps": 39000, "server_steps": 17500, "semantic_vector_uplink_bytes": 10240000, "semantic_label_uplink_bytes": 40000, "global_proto_downlink_bytes_all_clients": 51200000, "wall_seconds": 462.9856507778168, "server_seconds": 71.59773135185242, "checkpoint_file": "checkpoint_cycle25.pt"}
- Cycle50: {"metrics": {"l2": {"seen": 0.5621999949216843, "missing": 0.0, "all": 0.11244000047445298, "macro": 0.11244000047445298}, "head": {"seen": 0.5611500084400177, "missing": 0.0, "all": 0.11222999915480614, "macro": 0.11222999840974808}}, "per_client_coverage": {"l2": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "head": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]}, "aggregate_coverage": {"l2": 100, "head": 100}, "last_epoch_server_loss": 5.9044092178344725, "first_epoch_server_loss": 6.037796802520752, "client_steps": 78000, "server_steps": 35000, "semantic_vector_uplink_bytes": 20480000, "semantic_label_uplink_bytes": 80000, "global_proto_downlink_bytes_all_clients": 102400000, "wall_seconds": 925.0949380397797, "server_seconds": 142.62897276878357, "checkpoint_file": "checkpoint_cycle50.pt"}
- Cycle100: {"metrics": {"l2": {"seen": 0.5796999990940094, "missing": 0.0, "all": 0.11593999937176705, "macro": 0.11593999639153481}, "head": {"seen": 0.5828999996185302, "missing": 0.0, "all": 0.11657999977469444, "macro": 0.11657999902963638}}, "per_client_coverage": {"l2": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "head": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]}, "aggregate_coverage": {"l2": 100, "head": 100}, "last_epoch_server_loss": 4.892917251586914, "first_epoch_server_loss": 4.9578520202636716, "client_steps": 156000, "server_steps": 70000, "semantic_vector_uplink_bytes": 40960000, "semantic_label_uplink_bytes": 160000, "global_proto_downlink_bytes_all_clients": 204800000, "wall_seconds": 1854.57115483284, "server_seconds": 273.42015314102173, "checkpoint_file": "checkpoint_cycle100.pt"}

Cycles whose final server-epoch loss is <0.001: []
Fresh trajectory with historical seed0client/server initialhashes, exactsplit/anchors-excluded/firstbatch and round1clientmodelhashes. Only prototypecollection timing changes at cycle10; cycle100 additionally has10xlocal/server/communication budget. All100 globalprototype distances finite. No anchors/testlabels/PPRTP optimization path.
PPRTP extraanchor uplink5242880bytes remains; no equal-information or communication-efficiency claim. FedTGP server576512parameters not transmitted. Final serverhash: 802f736f7777e7d9cd80e777db722c3cea788e57ef4dcf233feb802cb29695fc
This is one seed and bounded100cycles; do not claim globally converged FedTGP unless its criterion is met, and do not extend/tune to improve results. H15-A artifacts unchanged. Fixed binarycheckpoints stayremote; JSONreceipts and per-round/per-client metrics are committed.
