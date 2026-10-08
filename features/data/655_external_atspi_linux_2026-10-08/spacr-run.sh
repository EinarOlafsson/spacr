set -e
export CUDA_VISIBLE_DEVICES=''
export SPACR_DEVICE=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QT_QPA_PLATFORM=xcb
export QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1
export QT_ACCESSIBILITY=1
export SPACR_N655_OUT=/mnt/wd4tb/scratch/n663-spaceout-prefs-current-20261008/n655-native
/mnt/wd4tb/spacr-worktrees/codex-readme-opening-20261008/tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n663-spaceout-prefs-current-20261008/n655-native-app.py > "$SPACR_N655_OUT.app.log" 2>&1 &
app_shell_pid=$!
for n in $(seq 1 30); do
  if grep -q '^READY ' "$SPACR_N655_OUT.app.log"; then break; fi
  sleep .2
done
app_pid=$(awk '/^READY / {print $2; exit}' "$SPACR_N655_OUT.app.log")
if [ -z "$app_pid" ]; then
  cat "$SPACR_N655_OUT.app.log"
  kill "$app_shell_pid" || true
  exit 1
fi
/usr/bin/python3 /mnt/wd4tb/scratch/n663-spaceout-prefs-current-20261008/n655-atspi-reader.py "$app_pid" > "$SPACR_N655_OUT.reader.json"
wait "$app_shell_pid"
