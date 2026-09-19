#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/PPRTP/current'
export AUTODL_RUN_ID='20260919-144131-h15b-postupdate100'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-144131-h15b-postupdate100'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-144131-h15b-postupdate100/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
PPRTP_SOURCE_SHA=e0e171f6bfb0247eba599a263ca26241e248d9f7 bash scripts/run_h01.sh --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/cifar100 --modes fedtgp --seeds 0 --rounds 100 --full-data --dataset CIFAR100 --num-classes 100 --k 20 --ownership-seed 120100 --fedtgp-prototype-timing post_update
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
