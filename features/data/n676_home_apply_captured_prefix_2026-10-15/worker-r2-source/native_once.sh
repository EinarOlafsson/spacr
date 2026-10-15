#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014
PY=/home/carruthers/anaconda3/envs/spacr/bin/python
exec "$ROOT/tools/run_capped.sh" 12G "$PY" "/media/carruthers/mnt3/claude/n676_stages/work/lesson05-selective-20261015/worker-apply-capture-r2/supervise_once.py"
