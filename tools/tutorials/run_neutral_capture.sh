#!/usr/bin/env bash
# Give the real application a neutral account inside the recording namespace.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <private-stage> <python-executable> [capture_refresh options]" >&2
    exit 2
fi
capture_stage=$1
capture_python=$2
shift 2
for capture_argument in "$@"; do
    case "$capture_argument" in
        --stage|--stage=*)
            echo "Pass the private stage as the first argument, not --stage." >&2
            exit 2 ;;
    esac
done
case "$capture_stage" in
    /nas_mnt|/nas_mnt/*)
        echo "A recording stage must be on a healthy local filesystem." >&2
        exit 2 ;;
esac
capture_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p -- "$capture_stage"
capture_stage=$(realpath -- "$capture_stage")
capture_python=$(realpath --no-symlinks -- "$capture_python")
capture_mount=/tmp/spacr-tutorials
capture_network=()
# A fixed display (SPACR_TUTORIAL_XVFB_SERVER=93) keeps parallel recorders apart.
capture_display=(-a)
# Extra read-only inputs a recorder reads beside its stage, as "SRC:DEST" pairs
# (e.g. a retained catalog or synthetic database at DEST=/tmp/catalog).
capture_binds=()
for capture_pair in ${SPACR_TUTORIAL_EXTRA_RO_BINDS:-}; do
    capture_binds+=(--ro-bind "${capture_pair%%:*}" "${capture_pair#*:}")
done
if [[ -n ${SPACR_TUTORIAL_XVFB_SERVER:-} ]]; then
    capture_display=(-n "$SPACR_TUTORIAL_XVFB_SERVER")
fi
if [[ ${SPACR_TUTORIAL_OFFLINE:-0} == 1 ]]; then
    capture_network=(--unshare-net)
fi
mkdir -p -- "$capture_stage/profile/.cache/spacr/example_data" \
    "$capture_stage/profile/.spacr/runs"

# No host account file is changed. With HOME absent, normal getpwuid/Qt calls
# find this namespace's account; the application needs no path monkeypatch.
awk -F: -v home_path="$capture_mount/profile" '
    BEGIN { OFS=":" }
    $3 == 65534 || $1 == "tutorial" { next }
    { print }
    END { print "tutorial", "x", 65534, 65534, "Tutorial recording", home_path, "/bin/bash" }
' /etc/passwd > "$capture_stage/passwd"

# Alpha features get no tutorials. Every private profile in this stage starts
# with Preferences -> Show alpha features off; capture_refresh.py forces it off
# again in the recording's own store and refuses a frame while it is on (only
# its --preferences-alpha-toggle-scene opt-in may record the toggle itself).
capture_profiles=("$capture_stage/profile/.config")
for capture_config in "$capture_stage"/config/*/; do
    [[ -d $capture_config ]] && capture_profiles+=("${capture_config%/}")
done
"$capture_repo/tools/run_capped.sh" 2G env QT_QPA_PLATFORM=offscreen \
    "$capture_python" "$capture_repo/tools/tutorials/capture_policy.py" \
    --force-alpha-off "${capture_profiles[@]}"

exec "$capture_repo/tools/run_capped.sh" "${SPACR_TUTORIAL_MEMORY_CAP:-6G}" \
    bwrap --unshare-user --uid 65534 --gid 65534 "${capture_network[@]}" \
    --bind / / --dev-bind /dev /dev --proc /proc --tmpfs /nas_mnt \
    --ro-bind "$capture_repo" /tmp/spacr-code --chdir /tmp/spacr-code \
    --bind "$capture_stage" "$capture_mount" "${capture_binds[@]}" \
    --ro-bind "$capture_stage/passwd" /etc/passwd \
    --unsetenv HOME --unsetenv USER --unsetenv LOGNAME \
    --unsetenv USERNAME --unsetenv SUDO_UID --unsetenv PKEXEC_UID -- \
    env CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    xvfb-run "${capture_display[@]}" -s '-screen 0 3840x2160x24' "$capture_python" \
    tools/tutorials/capture_refresh.py --stage "$capture_mount" --platform xcb "$@"
