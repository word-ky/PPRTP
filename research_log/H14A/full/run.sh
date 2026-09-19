#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/PPRTP/current'
export AUTODL_RUN_ID='20260919-122706-h14a-ownership1'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-122706-h14a-ownership1'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-122706-h14a-ownership1/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
PPRTP_SOURCE_SHA=c9b91c4dd55065cc3297afca363aadd40169ee4a bash scripts/run_h01.sh --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/cifar100 --modes local fedproto fedgh --seeds 0 --rounds 10 --full-data --dataset CIFAR100 --num-classes 100 --k 20 --ownership-seed 1
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
