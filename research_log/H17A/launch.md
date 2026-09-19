# H17-A launch — 2026-09-19 18:54 +08

STATUS: PARTIAL / RUNNING. Source c59015e7e8bc9fc457012c5ec141cda5cc3e4cba; release 20260919-184407-h17a; run 20260919-185409-h17a-tiny-one-owner. Remote root /home/wenchang/asdasdsad/wjq/PPRTP. Local root C:/work/PPRTP.

Local full 79 tests PASS (516.365s); remote same 79 PASS (172.595s), saved remote_tests.txt. Raw official archive 248100043 bytes, ZIP CRC all 120609 members passed, SHA256 6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821 matches remote. Official HTTPS byte-range download on Windows replaced the slow remote transfer; no data substitution. Archive receipt and download logs retained. Extracted under shared/tiny-h17/tiny-imagenet-200.

Preflight verify_raw.py passed exact raw train100000/val10000 and500/50 perclass, canonical mapping/annotation/file hashes, oneowner20classes/client,99744supervised+256anchors. Compact receipt raw_verification.json. Initial relative-path preflight command failed because current resolves inside releases; retried unchanged script using absolute path successfully. No training retry.

Launch command via scripts/autodl-run.ps1 -Name h17a-tiny-one-owner:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PPRTP_SOURCE_SHA=c59015e7e8bc9fc457012c5ec141cda5cc3e4cba
cp /home/wenchang/asdasdsad/wjq/PPRTP/shared/h17a-tests.txt "$AUTODL_ARTIFACTS_DIR/tests.txt"
/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python -u -m pprtp.run --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/tiny-h17/tiny-imagenet-200 --output "$AUTODL_ARTIFACTS_DIR/experiment" --modes local fedproto fedgh fedavg --seeds 0 --rounds 10 --full-data --dataset TinyImageNet --num-classes 200 --k 20 --owners-per-class 1 --ownership-seed 120200
```

Tmux started18:54:16+08, PythonPID749949. Dataset loading observed; no accuracy available yet. Existing other GPU jobs untouched. Monitor this run, do not duplicate. Collect compact experiment artifacts to research_log/H17A/full and run scripts/report_h17a.py after completion; retain LocalSeen<10% readiness override and all frozen thresholds. Stop after seed0 regardless of outcome.

GitHub push failed403: You must verify your email address, including retry18:53. Fetch works. User notified to verify at https://github.com/settings/emails. Local commits and remote project mirrors remain available; do not claim GitHub synchronized until push succeeds. Credit recharge does not resolve this separate GitHub restriction. Every20minute heartbeat pprtp-chatgpt remains ACTIVE and points at C:/work/PPRTP.
[2026-09-19T18:57+08:00] Remote mirror verified after eae6645. H17-A healthy: Local rounds1/2 completed (31.24s/59.58s cumulative); seen17.25%/22.95%,missing0. These are interim only, final readiness/result gate waits10cycles. split.json,metadata.json,rounds.jsonl present. No OOM/traceback observed. GitHub push remains pending emailverification.
