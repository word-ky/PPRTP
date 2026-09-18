#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/PPRTP/current'
export AUTODL_RUN_ID='20260919-002740-h11b-full-seeds12'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-002740-h11b-full-seeds12'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260919-002740-h11b-full-seeds12/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
PPRTP_SOURCE_SHA=9e6b25b6a36352457dadc4d7787381d022c7b77e bash scripts/run_h01.sh --modes local fedproto fedgh --seeds 1 2 --rounds 10 --full-data
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
