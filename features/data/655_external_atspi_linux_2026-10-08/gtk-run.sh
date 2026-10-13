set -e
root=/mnt/wd4tb/scratch/n663-spaceout-prefs-current-20261008
/usr/bin/python3 "$root/n655-gtk-control.py" > "$root/n655-gtk-control.app.log" 2>&1 &
app_shell_pid=$!
for n in $(seq 1 20); do
  if grep -q '^READY ' "$root/n655-gtk-control.app.log"; then break; fi
  sleep .2
done
app_pid=$(awk '/^READY / {print $2; exit}' "$root/n655-gtk-control.app.log")
/usr/bin/python3 "$root/n655-atspi-reader.py" "$app_pid" > "$root/n655-gtk-control.reader.json"
wait "$app_shell_pid"
