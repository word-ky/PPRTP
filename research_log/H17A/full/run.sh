#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/PPRTP/current'
export AUTODL_RUN_ID='20260919-185409-h17a-tiny-one-owner'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-185409-h17a-tiny-one-owner'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-185409-h17a-tiny-one-owner/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PPRTP_SOURCE_SHA=c59015e7e8bc9fc457012c5ec141cda5cc3e4cba; cp /home/wenchang/asdasdsad/wjq/PPRTP/shared/h17a-tests.txt "$AUTODL_ARTIFACTS_DIR/tests.txt"; /home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python -u -m pprtp.run --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/tiny-h17/tiny-imagenet-200 --output "$AUTODL_ARTIFACTS_DIR/experiment" --modes local fedproto fedgh fedavg --seeds 0 --rounds 10 --full-data --dataset TinyImageNet --num-classes 200 --k 20 --owners-per-class 1 --ownership-seed 120200
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
