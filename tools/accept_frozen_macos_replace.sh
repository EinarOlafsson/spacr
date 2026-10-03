#!/bin/bash
# Unattended macOS user-writable frozen-bundle acceptance on a clean runner.
# Usage: accept_frozen_macos_replace.sh OLD_ROOT NEW_ROOT OUTPUT
# OLD_ROOT/NEW_ROOT are legacy-build-macos artifacts (dist/*.dmg,
# acceptance/SHA256SUMS, acceptance/source-commit.txt) for 1.5.1.0 and 1.5.1.1.
# Phases: replace (real exchange + new-version restart), crash after the
# atomic exchange and crash before it (process killed with os._exit, then a
# fresh process runs recovery), each followed by a frozen Measure launch.
# The protected /Applications flows are out of scope (maintainer only).
set -euo pipefail
old_root=$(cd "$1" && pwd); new_root=$(cd "$2" && pwd); out="$3"
mkdir -p "$out"; out=$(cd "$out" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)
apps="$HOME/Applications"; target="$apps/spaCR.app"
mkdir -p "$apps"

verify() { (cd "$1/dist" && shasum -a 256 -c ../acceptance/SHA256SUMS) >/dev/null; }
verify "$old_root"; verify "$new_root"
old_dmg=$(ls "$old_root"/dist/*1.5.1.0*.dmg); new_dmg=$(ls "$new_root"/dist/*.dmg)
old_commit=$(tr -d '\r\n' < "$old_root/acceptance/source-commit.txt")
new_commit=$(tr -d '\r\n' < "$new_root/acceptance/source-commit.txt")
new_mount="$RUNNER_TEMP/new-mounted"; old_mount="$RUNNER_TEMP/old-mounted"
mkdir -p "$new_mount" "$old_mount"
hdiutil attach -readonly -nobrowse -mountpoint "$old_mount" "$old_dmg" >/dev/null
hdiutil attach -readonly -nobrowse -mountpoint "$new_mount" "$new_dmg" >/dev/null

version_of() { plutil -extract SPACRPackageVersion raw "$1/Contents/Info.plist"; }
new_version=$(python3 -c 'import plistlib,sys;print(plistlib.load(open(sys.argv[1],"rb"))["SPACRPackageVersion"])' "$new_mount/spaCR.app/Contents/Info.plist")

install_old() {
  rm -rf "$target" "$apps"/.spacr-update-*
  ditto "$old_mount/spaCR.app" "$target"
}

smoke() {  # label commit -> exit status recorded, smoke.json checked
  local label="$1" commit="$2" receipt="$out/$1"
  mkdir -p "$receipt/profile" "$receipt/cwd"
  set +e
  (cd "$receipt/cwd" && env -i HOME="$receipt/profile" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
     SPACR_DEVICE=cpu SPACR_NO_SETUP=1 SPACR_DISTRIBUTION_SMOKE=1 SPACR_DISTRIBUTION_KIND=frozen \
     OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
     SPACR_ACCEPTANCE_SOURCE_COMMIT="$commit" SPACR_BENCHMARK_JSON="$receipt/smoke.json" \
     "$target/Contents/MacOS/spacr" --no-setup > "$receipt/application.log" 2>&1) &
  local pid=$!
  (sleep 660; kill "$pid" 2>/dev/null) & local dog=$!
  wait "$pid"; local rc=$?
  kill "$dog" 2>/dev/null; wait "$dog" 2>/dev/null
  set -e
  echo "$rc" > "$receipt/exit.txt"
  test "$rc" = 0
  test "$(plutil -extract status raw "$receipt/smoke.json")" = passed
  test "$(plutil -extract frozen raw "$receipt/smoke.json")" = true
}

stage_new() {  # -> prints transaction dir with staged bundle
  local tx; tx=$(mktemp -d "$apps/.spacr-update-XXXXXX"); chmod 700 "$tx"
  ditto "$new_mount/spaCR.app" "$tx/spaCR.app"
  echo "$tx"
}

helper() {  # mode tx -> runs install_cleanup in a fresh python process
  python3 - "$repo/spacr/install_cleanup.py" "$1" "$target" "$2" "$new_mount/spaCR.app" "$new_version" <<'PY'
import importlib.util, json, os, sys
path, mode, target, tx, mounted = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("install_cleanup", path)
ic = importlib.util.module_from_spec(spec); sys.modules["install_cleanup"] = ic; spec.loader.exec_module(ic)
if mode == "recover":
    print(json.dumps(ic._macos_recover_user_bundle(tx), default=str)); sys.exit(0)
def swap(a, b):
    if mode == "crash-before":
        os._exit(137)
    ic._macos_swap(a, b)
    if mode == "crash-after":
        os._exit(137)
expected = ic._macos_tree(mounted)
staged = os.path.join(tx, "spaCR.app")
print(json.dumps(ic._macos_replace_user_bundle(target, staged, tx, sys.argv[6], expected, swap=swap)))
PY
}

journal_state() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["state"])' "$1/journal.json"; }

# Phase 1: replace and restart
install_old
smoke before-replace "$old_commit"
tx=$(stage_new)
helper replace "$tx" > "$out/replace.json"
replace_state=$(journal_state "$tx")
replace_version=$(version_of "$target"); backup_version=$(version_of "$tx/spaCR.app")
smoke after-replace "$new_commit"

# Phases 2 and 3: process killed after / before the atomic exchange
for when in after before; do
  install_old
  tx=$(stage_new)
  set +e; helper "crash-$when" "$tx" > "$out/crash-$when.json" 2>&1; crash_rc=$?; set -e
  echo "$crash_rc" > "$out/crash-$when-exit.txt"
  echo "$(journal_state "$tx")" > "$out/crash-$when-journal-before-recovery.txt"
  helper recover "$tx" > "$out/recover-$when.json"
  echo "$(journal_state "$tx")" > "$out/crash-$when-journal-after-recovery.txt"
  version_of "$target" > "$out/crash-$when-version.txt"
  test "$(cat "$out/crash-$when-version.txt")" = 1.5.1.0
  if [ "$when" = after ]; then smoke after-crash-recovery "$old_commit"
  else smoke after-precrash-recovery "$old_commit"; fi
done

hdiutil detach "$old_mount" >/dev/null; hdiutil detach "$new_mount" >/dev/null
python3 - "$out" "$replace_state" "$replace_version" "$backup_version" "$old_commit" "$new_commit" <<'PY'
import json, platform, sys
out, state, new_v, backup_v, oc, nc = sys.argv[1:7]
r = lambda n: open(f"{out}/{n}").read().strip()
json.dump({"schema": 1, "macos": platform.mac_ver()[0], "machine": platform.machine(),
           "old_commit": oc, "new_commit": nc,
           "replace_journal": state, "installed_version_after_replace": new_v,
           "retained_backup_version": backup_v, "restart_smoke_after_replace": "passed",
           "crash_after_exchange": {"exit": r("crash-after-exit.txt"),
               "journal": r("crash-after-journal-before-recovery.txt"),
               "recovered": r("crash-after-journal-after-recovery.txt"),
               "version": r("crash-after-version.txt"), "smoke": "passed"},
           "crash_before_exchange": {"exit": r("crash-before-exit.txt"),
               "journal": r("crash-before-journal-before-recovery.txt"),
               "recovered": r("crash-before-journal-after-recovery.txt"),
               "version": r("crash-before-version.txt"), "smoke": "passed"}},
          open(f"{out}/macos-replace.json", "w"), indent=2)
PY
cat "$out/macos-replace.json"
