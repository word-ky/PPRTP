#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/PPRTP/current'
export AUTODL_RUN_ID='20260919-042844-h12b-cifar100-seeds12'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-042844-h12b-cifar100-seeds12'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-042844-h12b-cifar100-seeds12/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
PPRTP_SOURCE_SHA=070b4436d68348548fbf85bce0a680bf5fb84bbe bash scripts/run_h01.sh --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/cifar100 --modes local fedproto fedgh --seeds 1 2 --rounds 10 --full-data --dataset CIFAR100 --num-classes 100 --k 20
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
