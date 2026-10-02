#!/bin/bash
set -eu
stage=$1
script=$2
receipt=$3
mkdir -p "$stage"/{config,state,cache,logs,mpl,tmp,app-state,bin}
cat > "$stage/bin/spacr-run" <<'EOF'
#!/bin/sh
exec /home/carruthers/anaconda3/envs/spacr/bin/python -m spacr.cli "$@"
EOF
chmod +x "$stage/bin/spacr-run"
exec systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=100% -- env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/spacr-implementation-20261001/suggest-capture QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 /home/carruthers/anaconda3/envs/spacr/bin/python /tmp/spacr-implementation-20261001/guide-refresh/run_check.py "$receipt" /usr/bin/bwrap --ro-bind / / --dev-bind /dev /dev --proc /proc --bind "$stage" "$stage" --bind "$stage/app-state" /home/carruthers/.spacr --unshare-net --setenv XDG_CONFIG_HOME "$stage/config" --setenv XDG_CACHE_HOME "$stage/cache" --setenv XDG_STATE_HOME "$stage/state" --setenv SPACR_LOG_DIR "$stage/logs" --setenv MPLCONFIGDIR "$stage/mpl" --setenv TMPDIR "$stage/tmp" /home/carruthers/anaconda3/envs/spacr/bin/python "$script" "$stage"
