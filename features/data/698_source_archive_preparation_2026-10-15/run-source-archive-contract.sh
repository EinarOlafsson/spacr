#!/usr/bin/env bash
set -uo pipefail
scratch=/mnt/wd4tb/scratch/n698-managed-provisioning-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$scratch/tmp" PYTHONPYCACHEPREFIX="$scratch/pycache" MPLCONFIGDIR="$scratch/mpl" XDG_CACHE_HOME="$scratch/cache"
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p no:randomly -o addopts='' tests/test_web_space_build_requirements_n698.py::test_actual_offline_source_archive_hash_and_extra_resolution --junitxml="$scratch/source-archive-contract.xml" --basetemp="$scratch/source-archive-contract-tmp" >"$scratch/source-archive-contract.log" 2>&1
status=$?
printf '%s\n' "$status" >"$scratch/source-archive-contract.exit"
exit "$status"
