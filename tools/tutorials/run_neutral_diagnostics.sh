#!/usr/bin/env bash
# run_neutral_capture.sh's namespace, entering capture_diagnostics.py, which adds the
# private dbus-run-session and XDG dirs the external file manager and image viewer need.
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: $0 <private-stage> <python> <project-name-under-stage/regression_runs> [capture options]" >&2; exit 2; }
capture_stage=$(realpath -- "$1"); capture_python=$(realpath --no-symlinks -- "$2"); project_name=$3
shift 3
capture_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
capture_mount=/tmp/spacr-tutorials
mkdir -p -- "$capture_stage/profile/.cache/spacr/example_data" "$capture_stage/profile/.spacr/runs"
awk -F: -v home_path="$capture_mount/profile" '
    BEGIN { OFS=":" }
    $3 == 65534 || $1 == "tutorial" { next }
    { print }
    END { print "tutorial", "x", 65534, 65534, "Tutorial recording", home_path, "/bin/bash" }
' /etc/passwd > "$capture_stage/passwd"
# Cinnamon checks thumbnail ownership against the session identity. Inherited
# USERNAME/SUDO_UID/PKEXEC_UID would select the host account inside this namespace.
exec "$capture_repo/tools/run_capped.sh" 4G \
    bwrap --unshare-user --uid 65534 --gid 65534 --unshare-net \
    --bind / / --dev-bind /dev /dev --proc /proc --tmpfs /nas_mnt \
    --ro-bind "$capture_repo" /tmp/spacr-code --chdir /tmp/spacr-code \
    --bind "$capture_stage" "$capture_mount" \
    --ro-bind "$capture_stage/passwd" /etc/passwd \
    --unsetenv HOME --unsetenv USER --unsetenv LOGNAME --unsetenv USERNAME \
    --unsetenv SUDO_UID --unsetenv PKEXEC_UID -- \
    env CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$capture_python" tools/tutorials/capture_diagnostics.py --stage "$capture_mount" \
    --project "$capture_mount/regression_runs/$project_name" "$@"
