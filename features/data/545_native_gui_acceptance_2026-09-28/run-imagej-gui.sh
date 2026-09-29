#!/usr/bin/env bash
set -euo pipefail
TASK_OUT=/mnt/wd4tb/scratch/spacr-completion/545-native-interop-preparation
TASK_TREE=/mnt/wd4tb/scratch/spacr-completion/worktree
TASK_JAVA=/home/olafsson/Fiji.app/java/linux-amd64/zulu8.86.0.25-ca-fx-jre8.0.452-linux_x64
test ! -e "$TASK_OUT/imagej-exit.txt"
cd "$TASK_TREE"
date --iso-8601=seconds > "$TASK_OUT/imagej-started.txt"
set +e
tools/run_capped.sh 4G timeout 90 env \
 HOME="$TASK_OUT/home" XDG_CONFIG_HOME="$TASK_OUT/home/config" \
 XDG_CACHE_HOME="$TASK_OUT/home/cache" XDG_DATA_HOME="$TASK_OUT/home/data" \
 TMPDIR="$TASK_OUT/tmp" \
 CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 TASK_OUT="$TASK_OUT" TASK_JAVA="$TASK_JAVA" \
 xvfb-run -a -s '-screen 0 1440x900x24 -nolisten tcp' bash -c '
   set -euo pipefail
   task_cgroup=$(awk -F: '\''$1 == "0" {print $3}'\'' /proc/$$/cgroup)
   task_cap=$(cat "/sys/fs/cgroup$task_cgroup/memory.max")
   task_swap=$(cat "/sys/fs/cgroup$task_cgroup/memory.swap.max")
   printf "cgroup=%s\nmemory.max=%s\nmemory.swap.max=%s\ndisplay=%s\n" "$task_cgroup" "$task_cap" "$task_swap" "$DISPLAY" > "$TASK_OUT/imagej-cgroup.txt"
   test "$task_cap" = 4294967296
   test "$task_swap" = 0
   task_original_display="$DISPLAY"
   /home/olafsson/Fiji.app/ImageJ-linux64 \
     --java-home "$TASK_JAVA" --mem=512M \
     -Duser.home="$TASK_OUT/home" \
     -Djava.util.prefs.userRoot="$TASK_OUT/home/java-prefs" \
     -Dsun.java2d.opengl=false -Dsun.java2d.xrender=false -Dprism.order=sw \
     -XX:ActiveProcessorCount=2 \
     -- --ij1 --allow-multiple --no-splash -macro "$TASK_OUT/imagej-interop.ijm" \
     > "$TASK_OUT/imagej-native.log" 2>&1 &
   task_imagej_pid=$!
   printf "%s\n" "$task_imagej_pid" > "$TASK_OUT/imagej.pid"
   trap '\''kill "$task_imagej_pid" 2>/dev/null || true'\'' EXIT
   for task_turn in $(seq 1 300); do
     test ! -e "$TASK_OUT/imagej-native-status.txt" || break
     kill -0 "$task_imagej_pid" 2>/dev/null || break
     sleep 0.1
   done
   test "$DISPLAY" = "$task_original_display"
   xwininfo -root -tree > "$TASK_OUT/imagej-xwindows.txt"
   import -window root "$TASK_OUT/imagej-gui.png"
   test -e "$TASK_OUT/imagej-native-status.txt"
   printf "%s\n" "Captured private native GUI" > "$TASK_OUT/screenshot-complete.txt"
   wait "$task_imagej_pid"
   trap - EXIT
 ' > "$TASK_OUT/imagej-runner.log" 2>&1
TASK_STATUS=$?
set -e
printf '%s\n' "$TASK_STATUS" > "$TASK_OUT/imagej-exit.txt"
date --iso-8601=seconds > "$TASK_OUT/imagej-finished.txt"
exit "$TASK_STATUS"
