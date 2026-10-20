#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/tmp
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL PIP_NO_INDEX PIP_FIND_LINKS
out=/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015
python_bin=/home/olafsson/anaconda3/envs/spacr_new/bin/python
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
trap 'rc=$?; printf "%s\n" "$rc" > "$out/overall.exit"' EXIT
git rev-parse HEAD > "$out/source-sha.txt"
test "$(cat "$out/source-sha.txt")" = b51fb194bb183990ab25c981d70e3030063bf9ac
test -z "$(git status --porcelain)"
printf 'download-wheelhouse\n' > "$out/phase"
tools/run_capped.sh 4G "$python_bin" -m pip download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --dest "$out/wheelhouse" setuptools==83.0.0 wheel==0.47.0 packaging==26.3 > "$out/download.log" 2>&1
printf '0\n' > "$out/download.exit"
sha256sum "$out"/wheelhouse/*.whl > "$out/wheelhouse-sha256.txt"
export PIP_NO_INDEX=1 PIP_FIND_LINKS="$out/wheelhouse"
printf 'isolated-build\n' > "$out/phase"
tools/run_capped.sh 4G "$python_bin" -m build --outdir "$out/artifacts" > "$out/build.log" 2>&1
printf '0\n' > "$out/build.exit"
printf 'twine-check\n' > "$out/phase"
tools/run_capped.sh 4G /usr/bin/twine check "$out"/artifacts/* > "$out/twine.log" 2>&1
printf '0\n' > "$out/twine.exit"
sha256sum "$out"/artifacts/* > "$out/artifact-sha256.txt"
git diff --exit-code > "$out/source-diff.log"
test "$(git rev-parse HEAD)" = b51fb194bb183990ab25c981d70e3030063bf9ac
printf 'terminal-pass\n' > "$out/phase"
