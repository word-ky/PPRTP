# H01-D frozen execution

Research instruction: origin/main 0d72091, REVIEW 13. Implementation ac57d8666b231e5bcc2b362012b7806b81afe9ca.
H01-C stop honored; new authorized lambda for seen-only .03498, all .002, FedProto 1.
All other H01-B settings unchanged. Same seeds and deterministic splits/batch order.

First run: 20260917-113423-h01d-gate. Seed0 two rounds, 11 tests then real CUDA.
Seen/all scaled base-gradient ratio 1.0000123724 passed [0.8,1.25].
Exact first-round pairing and historical FedProto hashes passed.
Both dL/dz gradients computed on identical z/P before optimizer update; cosine .382520884,
unscaled all/seen feature-gradient norm ratio 2.33901405. No tuning.

Authorized full run: 20260917-113545-h01d-full, seeds0/1/2, 10 rounds, all three arms.
Commands use scripts/run_h01.sh and PPRTP_SOURCE_SHA=ac57d8666b231e5bcc2b362012b7806b81afe9ca:

```bash
bash scripts/run_h01.sh --seeds 0 --rounds 2 --seen-lamda 0.03498 --modes fedproto gpc_all_match gpc_seen_match
bash scripts/run_h01.sh --seeds 0 1 2 --seen-lamda 0.03498 --modes fedproto gpc_all_match gpc_seen_match
```

Reports: scripts/report_h01d.py RUN_PATH [--gate], standard-library summary only.
Do not pool gate receipts with the full run as additional seeds.
Round2/5/10 seen/all ratios after divergence use each arm's own client0 first-batch
base gradient; they are not shared-state calibrations. No dynamic matching applied.
Prior full BRIDGE history was summarized by research lead; exact original reports
remain in Git before 0d72091 and in project-local receipts.
