# H15-A matched-protocol FedTGP seed0

| Arm | Seen % | Missing % | All % | Macro % |
|---|---:|---:|---:|---:|
| FedTGP official nearest | 31.950000 | 0.000000 | 6.390000 | 6.390000 |
| FedTGP local head | 32.750000 | 0.000000 | 6.550000 | 6.550000 |
| frozen PPRTP | 15.895000 | 9.345000 | 10.655000 | 10.655000 |
| frozen local | 32.440000 | 0.000000 | 6.488000 | 6.488000 |
| frozen fedproto | 33.900000 | 0.000000 | 6.780000 | 6.780000 |
| frozen fedgh | 16.180000 | 0.000000 | 3.236000 | 3.236000 |

Frozen verdict: CLEAR PPRTP EDGE.
PPRTP minus FedTGP (pp): {"seen": -16.055000200867653, "missing": 9.345000013709068, "all": 4.265000149607659}
Per-client coverage: {"l2": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], "head": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]}
Aggregate predicted classes: {"head": 100, "cosine": 96, "l2": 100}
Communication: {"client_vectors_uplink_bytes_per_cycle": 409600, "client_labels_uplink_bytes_per_cycle": 1600, "global_prototype_downlink_bytes_per_client": 204800, "global_prototype_downlink_bytes_all_clients_per_cycle": 2048000, "server_model_parameters": 576512, "server_model_transmitted": false, "pprtp_anchor_uplink_bytes": 5242880}
Client optimizer steps15600; server SGD steps7000 (100epochs*7batches*10cycles). Total wall seconds176.751; server-update seconds26.96817898750305
Server first/last epoch losses by cycle: [[4.934429168701172, 4.876330623626709], [12.899390678405762, 11.252895889282227], [6.342536849975586, 5.5849441719055175], [6.5212524795532225, 6.130675392150879], [6.393396167755127, 6.155823097229004], [6.698735980987549, 6.511864566802979], [6.477854175567627, 6.307934722900391], [6.367000637054443, 6.2381095123291015], [5.896900634765625, 5.766415596008301], [6.164946880340576, 6.026274299621582]]
Initial/final server hashes: bb59b3677d36c58abcae6c7df05ef198b8f279df2043747a5e12b09da5c3e141 / b8d8729236f5fdb140afdebc81c0ecc3793bd4064dfb7869cd60fa054aecd239

ExactH12split/initialization and round1trainedmodelhashes; actualround1batchhashes equalH13A on exactH12split/seed/loader. Allclasses finite distance predictions. No anchor/testdata in client/server optimization; no PPRTPtransport.
Pinned official disk ordering uploads round-start checkpoint features; retained and tested. Matched batch32/drop_lastFalse/shuffled loaders/10cycles/finalpostserver evaluation and deterministicprivate server RNG are deliberate adaptations. Server margin unweighted-classmeans; individualprototype updates, no samplecount weighting.
Short-budget matched baseline only, not best/converged FedTGP. OfficialREADME discusses >1000 communicationiterations; no extension permitted here. PPRTP uses extra unlabeled same-image correspondence and has a seen-class tradeoff; no communication-efficiency claim. Only one seed. Provenance and adaptations in PROVENANCE.md.
