#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/PPRTP/current'
export AUTODL_RUN_ID='20260918-103900-h07b-crossseed-direct'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-103900-h07b-crossseed-direct'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/PPRTP/runs/20260918-103900-h07b-crossseed-direct/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
PPRTP_SOURCE_SHA=07d32b6d4e09b7792739409fb209de8246f362c8 bash scripts/run_h01.sh --modes fedgh --seeds 1 2 --rounds 10 --direct-cross-seed-probe
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
