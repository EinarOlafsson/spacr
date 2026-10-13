#!/usr/bin/env bash
set -euo pipefail
installation_repo=/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005
installation_stage=/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-standard-install-neutral-current-r1
installation_cache=/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-standard-install-current-r1/installation_pip_cache
mkdir -p "$installation_stage" "$installation_cache"
exec "$installation_repo/tools/run_capped.sh" 8G \
    bwrap --die-with-parent --bind / / --dev-bind /dev /dev \
    --ro-bind "$installation_repo" /tmp/spacr-code \
    --bind "$installation_stage" /tmp/spacr-tutorials \
    --bind "$installation_cache" /tmp/spacr-tutorials/installation_pip_cache \
    --ro-bind /home/carruthers/anaconda3 /tmp/conda \
    --chdir /tmp/spacr-code -- \
    env CUDA_VISIBLE_DEVICES= QT_QPA_PLATFORM=offscreen \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PATH=/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/xvfb-tools/extracted/usr/bin:/usr/local/bin:/usr/bin:/bin \
    /tmp/conda/envs/spacr12/bin/python "$@"
