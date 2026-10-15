#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
out=/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$out/tmp" PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL PIP_NO_INDEX PIP_FIND_LINKS PYTHONPATH
trap 'rc=$?; printf "%s\n" "$rc" > "$out/checker-overall.exit"' EXIT
sha256sum "$out"/artifacts/* > "$out/checker-artifact-before.sha256"
printf 'checker-venv\n' > "$out/checker-phase"
tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr_new/bin/python -m venv "$out/checker-venv" > "$out/checker-venv.log" 2>&1
printf 'checker-install\n' > "$out/checker-phase"
tools/run_capped.sh 2G "$out/checker-venv/bin/python" -m pip install --index-url https://pypi.org/simple --only-binary=:all: twine==6.2.0 > "$out/checker-install.log" 2>&1
printf '0\n' > "$out/checker-install.exit"
printf 'waiting-both-locks\n' > "$out/checker-phase"
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
printf 'same-artifacts-check\n' > "$out/checker-phase"
"$out/checker-venv/bin/python" -m twine --version > "$out/checker-version.txt"
tools/run_capped.sh 2G "$out/checker-venv/bin/python" -m twine check "$out"/artifacts/* > "$out/checker-twine.log" 2>&1
printf '0\n' > "$out/checker-twine.exit"
sha256sum "$out"/artifacts/* > "$out/checker-artifact-after.sha256"
cmp "$out/checker-artifact-before.sha256" "$out/checker-artifact-after.sha256"
printf 'terminal-pass\n' > "$out/checker-phase"
