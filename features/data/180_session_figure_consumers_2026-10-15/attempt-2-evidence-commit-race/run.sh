#!/bin/bash
set -eu
consumer_scratch=/mnt/wd4tb/scratch/home-b-n180-figure-consumers-20261015
consumer_root=/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-graph-api-20261015
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XDG_CONFIG_HOME="$consumer_scratch/config" XDG_CACHE_HOME="$consumer_scratch/cache" XDG_DATA_HOME="$consumer_scratch/data" XDG_STATE_HOME="$consumer_scratch/state" SPACR_HOME="$consumer_scratch/spacr"
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:"$consumer_root"
cd "$consumer_root"
python "$consumer_scratch/check-output-pins.py"
/usr/bin/time -v -o "$consumer_scratch/time.txt" tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python "$consumer_scratch/check-consumers.py" --root "$consumer_root" --expected-commit ab983f6e0e4a597081b8f7242d7a9dfde352711c --expected-help-sha256 16b4cc60e976eefe15fc309b35f653f342808682f297cda530f885a089d221ee --before-help /mnt/wd4tb/scratch/home-b-n180-figure-producer-20261015/run-1/before-help_api_index.py > "$consumer_scratch/result.json"
python "$consumer_scratch/check-output-pins.py"
