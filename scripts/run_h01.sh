#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
PY=/home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python
"$PY" -m unittest discover -s tests -v 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/tests.txt"
"$PY" -m pprtp.run --data /home/wenchang/asdasdsad/wjq/PPRTP/shared/cifar10 --output "$AUTODL_ARTIFACTS_DIR/experiment" "$@"
