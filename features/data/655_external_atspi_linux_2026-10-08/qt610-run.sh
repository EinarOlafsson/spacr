set -e
root=/mnt/wd4tb/scratch/n663-spaceout-prefs-current-20261008
export QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1
export QT_ACCESSIBILITY=1
export QT_QPA_PLATFORM=xcb
/home/olafsson/anaconda3/bin/python "$root/n655-qt-control.py" > "$root/n655-qt610-control.app.log" 2>&1 &
app_shell_pid=$!
for n in $(seq 1 20); do
  if grep -q '^READY ' "$root/n655-qt610-control.app.log"; then break; fi
  sleep .2
done
app_pid=$(awk '/^READY / {print $2; exit}' "$root/n655-qt610-control.app.log")
/usr/bin/python3 "$root/n655-atspi-reader.py" "$app_pid" > "$root/n655-qt610-control.reader.json"
wait "$app_shell_pid"
