#!/usr/bin/env bash
set -uo pipefail
scratch=/mnt/wd4tb/scratch/n698-managed-provisioning-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$scratch/tmp" PYTHONPYCACHEPREFIX="$scratch/pycache" MPLCONFIGDIR="$scratch/mpl" XDG_CACHE_HOME="$scratch/cache" NUMBA_CACHE_DIR="$scratch/numba"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m tools.web_mask_generator.provision --prepare-manifest /mnt/wd4tb/scratch/n698-zerogpu-deployment-preparation-20261015/public-weight-metadata.json --output tools/web_mask_generator/space/models.json >"$scratch/manifest.log" 2>&1
status=$?
printf '%s\n' "$status" >"$scratch/manifest.exit"
if test "$status" != 0; then printf '%s\n' "$status" >"$scratch/owning.exit"; exit "$status"; fi
tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p no:randomly -o addopts='' tests/test_web_managed_provision_n698.py tests/test_web_space_entry_import_order_n698.py --junitxml="$scratch/owning.xml" --basetemp="$scratch/owning-tmp" >"$scratch/owning.log" 2>&1
status=$?
printf '%s\n' "$status" >"$scratch/owning.exit"
exit "$status"
