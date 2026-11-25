#!/usr/bin/env bash
# Capture an existing owned HTML gallery; run only inside the root's scheduler job.
# Chrome/graphics processes are never GPU compute or GPU queue blockers.
set -euo pipefail
set +m
umask 077

if [[ $# != 3 ]]; then
    printf 'Usage: %s HTML_PATH FRESH_SCREENSHOT.png FRESH_PROFILE_DIRECTORY\n' "$0" >&2
    exit 2
fi
root=$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")/..")
html=$(realpath -e -- "$1")
output=$(realpath -m -- "$2")
profile=$(realpath -m -- "$3")
for requested in "$1" "$2" "$3"; do
    [[ $(realpath -ms -- "$requested") == "$(realpath -m -- "$requested")" ]] || {
        printf 'Refusing a symlinked input/output/profile path: %s\n' "$requested" >&2
        exit 2
    }
done
[[ $html == "$root"/reports/graphstyle/g2/evidence/*.html && -f $html ]]
[[ $output == "$root"/reports/graphstyle/g2/evidence/*.png ]]
[[ $profile == "$root"/.codex-audit/* ]]
[[ $profile != "$root"/.codex-audit && -d $(dirname -- "$profile") ]]
[[ -d $(dirname -- "$output") && ! -e $output && ! -L $output ]]
[[ ! -e $output.chrome.log && ! -L $output.chrome.log ]]
[[ ! -e $profile && ! -L $profile ]]
# These generated galleries need only embedded data images and inline CSS.
# Refuse active/remote content rather than enabling a network or changing the page.
if LC_ALL=C grep -Eiq '<(script|iframe|object|embed|link|base)([[:space:]>])|https?://|ftp://|@import|<meta[^>]+http-equiv' "$html"; then
    printf 'Refusing active or external gallery content: %s\n' "$html" >&2
    exit 2
fi
[[ $html != *'%'* && $html != *'#'* && $html != *'?'* ]]
[[ -x /usr/bin/google-chrome && -x /usr/bin/setsid && -x /usr/bin/timeout ]]
html_sha=$(sha256sum -- "$html" | cut -d ' ' -f 1)
# Chrome's SingletonSocket is below TMPDIR; keep its UNIX-socket path under
# 108 bytes without a symlink, shared /tmp, or changing the separate profile.
temp_template="$root/.codex-audit/ct.XXXXXX"
socket_example="$temp_template/com.google.Chrome.XXXXXX/SingletonSocket"
[[ ${#temp_template} -lt 70 && ${#socket_example} -lt 108 ]]
[[ $(realpath -ms -- "$root/.codex-audit") == "$(realpath -e -- "$root/.codex-audit")" ]]
chrome_temp=$(mktemp -d -- "$temp_template")
[[ -d $chrome_temp && ! -L $chrome_temp && $chrome_temp == "$root"/.codex-audit/ct.* ]]
mkdir -m 700 -- "$profile"
mkdir -m 700 -- "$profile/xdg-config" "$profile/xdg-cache" \
    "$profile/xdg-data" "$profile/runtime"
chrome_job=''
cleanup() {
    # Signal only the new session created below, never any existing Chrome process.
    if [[ -n $chrome_job ]] && kill -0 -- "-$chrome_job" 2>/dev/null; then
        kill -TERM -- "-$chrome_job" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$chrome_job" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
printf 'HTML=%s\nHTML_SHA256=%s\nVIEWPORT=1440x1000\nPROFILE=%s\nTEMP_DIRECTORY=%s\nSOCKET_PATH_BUDGET=%s\n' \
    "$html" "$html_sha" "$profile" "$chrome_temp" "${#socket_example}"
XDG_CONFIG_HOME="$profile/xdg-config" XDG_CACHE_HOME="$profile/xdg-cache" \
XDG_DATA_HOME="$profile/xdg-data" XDG_RUNTIME_DIR="$profile/runtime" \
TMPDIR="$chrome_temp" TMP="$chrome_temp" TEMP="$chrome_temp" \
    /usr/bin/setsid /usr/bin/timeout --signal=TERM --kill-after=5s 45s \
    /usr/bin/google-chrome --headless=new --disable-gpu \
    --user-data-dir="$profile" --disk-cache-dir="$profile/xdg-cache/chrome" \
    --no-first-run --no-default-browser-check --disable-background-networking \
    --disable-component-update --disable-default-apps --disable-sync \
    --disable-domain-reliability --disable-breakpad --no-pings --metrics-recording-only \
    --disable-features=MediaRouter,OptimizationHints,Translate,AutofillServerCommunication \
    --no-proxy-server --host-resolver-rules='MAP * ~NOTFOUND' \
    --force-device-scale-factor=1 --window-size=1440,1000 \
    --run-all-compositor-stages-before-draw --virtual-time-budget=1500 \
    --screenshot="$output" "file://$html" >"$output.chrome.log" 2>&1 &
chrome_job=$!
status=0
wait "$chrome_job" || status=$?
cleanup
chrome_job=''
[[ $(sha256sum -- "$html" | cut -d ' ' -f 1) == "$html_sha" ]] || {
    printf 'Input HTML changed during capture.\n' >&2
    exit 1
}
if [[ $status != 0 ]]; then
    printf 'Chrome capture failed with status %s; log/profile/owned temp retained.\n' "$status" >&2
    exit "$status"
fi
[[ -s $output && ! -L $output ]]
[[ $(od -An -tx1 -N8 -- "$output" | tr -d ' \n') == 89504e470d0a1a0a ]]
sha256sum -- "$output" "$output.chrome.log"
printf 'Captured original HTML at fixed viewport; no theme injection or full-page/overflow certification.\n'
