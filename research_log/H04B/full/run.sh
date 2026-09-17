#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/PPRTP/current'
export AUTODL_RUN_ID='20260918-002118-h04b-crossseed'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-002118-h04b-crossseed'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-002118-h04b-crossseed/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
PPRTP_SOURCE_SHA=c27f05b8996018c65ae7fa032487a40b9154edbf bash scripts/run_h01.sh --modes fedgh --seeds 1 2 --rounds 10 --cross-seed-probe
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
