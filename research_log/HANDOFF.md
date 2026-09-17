# Latest status — H01-D COMPLETE (2026-09-17)

Gate113423 and full113545 completed, source ac57d8666b231e5bcc2b362012b7806b81afe9ca. See research_log/H01D/full/RESULTS.md and CODEX REPORT H01-D. No next stage authorized; do not repeat H01-D. All vs seen common-cosine all accuracy13.33 vs13.41%, both missing0; initial strength match passed but later drifted. D: nearly full: full checkpoints remain authoritative on A6000 under PPRTP/runs/20260917-113545-h01d-full, compact metrics local. One redundant local gate checkpoint evicted after remote SHA match; see progress.

# Latest status — H01-C gate stopped as instructed (2026-09-17)

Source c744b1b; run 20260917-102726-h01c-gate exit 0. Ten tests pass locally/remotely. All-class ratio 1.215 PASS; seen-only .06947 FAIL. Seed 0 rounds 1–2 only. No further seeds or rounds authorized after failed gate; no tuning. See research_log/H01C and appended CODEX REPORT H01-C. Await a NEW or substantively revised research instruction; do not repeat the old ACTIVE task. Heartbeat pprtp-chatgpt checks every 20 minutes.

# PPRTP handoff — 2026-09-17

H01/H01-B implementation and three-seed real CIFAR-10 subset run COMPLETE.
Await research lead review / next ACTIVE task; do not add modules or tune parameters.

Local root: D:/work/fightccfa-agin/CVPR2027/personalized prototype.
Remote root: /home/wenchang/asdasdsad/wjq/PPRTP.
Source revision: 52a6c8f3d167b7b7827386221238e1576e1eaea8.
Pinned upstream PFLlib: 0169ba7e412c9856a08bb3faefab1e35f538a3c1, unchanged submodule.
Remote runtime: /home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python (used read-only).
GPU: RTX A6000 device 0, Torch2.4.0+cu121 / torchvision0.19.0+cu121.

Completed runs (all exit0):
- 20260916-221227-h01-baseline: original upstream two-round CPU health test.
- 20260917-000608-h01b-smoke: 8 tests + three-arm two-round CIFAR CUDA smoke.
- 20260917-000651-h01b-seed0: 8 tests + seed0, 10-round three-arm experiment.
- 20260917-000820-h01b-seeds12: 8 tests + seeds1/2, identical frozen configuration.
All raw outputs and representative checkpoints fetched under research_log/remote_runs.
Git-visible scientific receipts: research_log/H01B/receipts.
Seed0-only detailed table: research_log/H01B_seed0/RESULTS.md.
Final three-seed table: research_log/H01B/RESULTS.md.

Key result: round10 common cosine all-class accuracy FedProto13.37±1.11%,
GPC13.96±0.99%; missing accuracy0% in both, every seed. Paired first-round model
and prototype hashes exactly equal. Same split/initialization across arms.
First active knowledge/local gradient ratio: FedProto .00710–.00886,
GPC3.05–4.75. No nonfinite losses, zero-norm or identical-direction prototype collapse.
Do not claim strong missing-class transfer or establish the bottleneck hypothesis.

Data recovered from MindSpore-documented mirror:
https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-python.tar.gz
Archive MD5 c58f30108f718f92721af3b95e74349a, matching torchvision official archive.
Extracted under remote shared/cifar10. Partial official/UCSD downloads preserved.
Failed run 20260916-221759-h01-smoke stopped before training at Python SSL CA failure;
system curl retained TLS verification. No server driver/environment change performed.
No automation created in this task; user's research-side hourly loop is external.


