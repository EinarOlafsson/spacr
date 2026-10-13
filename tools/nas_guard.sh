#!/usr/bin/env bash
# Touch a network mount without ever hanging the session that touches it.
#
# WHY THIS EXISTS. On 2026-09-20 a session stopped responding and had to be
# restarted by hand; it was waiting on the NAS. `/nas_mnt` is NFSv3 mounted
# `hard` (timeo=600, retrans=2, x-systemd.automount). A hard mount does not
# fail when the server stops answering -- it blocks, forever, in
# uninterruptible sleep, where SIGKILL does nothing at all.
#
# `timeout` DOES NOT SAVE YOU, and this is the part that surprises people:
# after its deadline it signals the child and then WAITS for it to die. A
# process in D state never dies, so timeout waits with it and the caller
# hangs anyway. The same goes for `timeout -k`: SIGKILL is delivered when
# the process next leaves the kernel, which is never.
#
# The only thing that works is to never wait on the process that touches the
# mount. Every probe here runs DETACHED, in its own session, and reports
# through a file. This script polls the file against a deadline and returns
# whether or not the probe ever finished. Giving up costs one sleeping
# process, which the kernel reaps if the server comes back; it does not cost
# the session.
#
#   tools/nas_guard.sh check [PATH] [SECONDS]  0 answers, 2 did not, 3 not a mount
#   tools/nas_guard.sh run SECONDS CMD...      CMD under a deadline, detached
#   tools/nas_guard.sh paths                   network mounts to treat this way
#
# `paths` reads /proc/self/mountinfo through findmnt, which asks the kernel
# what is mounted and never asks the server anything, so it is safe to run
# when the server is gone.
set -uo pipefail

NAS_GUARD_DEFAULT_PATH="${NAS_GUARD_DEFAULT_PATH:-/nas_mnt}"
NAS_GUARD_DEFAULT_SECONDS="${NAS_GUARD_DEFAULT_SECONDS:-5}"

_poll_for() {
    # Wait for a file to appear, up to a deadline, without ever calling
    # `wait` on the process that is meant to write it.
    local marker="$1" deadline="$2" waited=0
    local step=0.1
    while [[ ! -s "$marker" ]]; do
        # bash arithmetic is integer, so count in tenths.
        if (( waited >= ${deadline%.*} * 10 )); then
            return 1
        fi
        sleep "$step"
        waited=$(( waited + 1 ))
    done
    return 0
}

_detach() {
    # Start a command in its own session, with every descriptor pointed away
    # from this shell, so nothing it does can block us and nothing it writes
    # can arrive after we have given up.
    setsid "$@" </dev/null >/dev/null 2>&1 &
    disown 2>/dev/null || true
}

cmd_check() {
    local path="${1:-$NAS_GUARD_DEFAULT_PATH}"
    local deadline="${2:-$NAS_GUARD_DEFAULT_SECONDS}"
    local dir
    dir="$(mktemp -d)"
    local marker="$dir/answer"

    _detach bash -c 'if stat -c %d -- "$1" >/dev/null 2>&1; then echo ok > "$2"; else echo fail > "$2"; fi' _ "$path" "$marker"

    if ! _poll_for "$marker" "$deadline"; then
        echo "nas_guard: $path did not answer within ${deadline}s -- treating it as unavailable" >&2
        rm -rf "$dir" 2>/dev/null
        return 2
    fi
    local answer
    answer="$(cat "$marker" 2>/dev/null)"
    rm -rf "$dir" 2>/dev/null
    if [[ "$answer" == "ok" ]]; then
        return 0
    fi
    echo "nas_guard: $path answered, and it is not there" >&2
    return 3
}

cmd_run() {
    local deadline="${1:?usage: nas_guard.sh run SECONDS CMD...}"
    shift
    if [[ $# -eq 0 ]]; then
        echo "usage: nas_guard.sh run SECONDS CMD..." >&2
        return 2
    fi
    local dir
    dir="$(mktemp -d)"
    local marker="$dir/status" out="$dir/out" err="$dir/err"

    _detach bash -c 'set +e; "${@:4}" >"$1" 2>"$2"; echo "$?" > "$3".tmp; mv "$3".tmp "$3"' _ "$out" "$err" "$marker" "$@"

    if ! _poll_for "$marker" "$deadline"; then
        echo "nas_guard: gave up after ${deadline}s on: $*" >&2
        echo "nas_guard: the command was abandoned, not killed -- a hard NFS wait ignores signals" >&2
        cat "$out" 2>/dev/null
        cat "$err" >&2 2>/dev/null
        return 124
    fi
    local status
    status="$(cat "$marker" 2>/dev/null)"
    cat "$out" 2>/dev/null
    cat "$err" >&2 2>/dev/null
    rm -rf "$dir" 2>/dev/null
    return "${status:-1}"
}

cmd_paths() {
    findmnt --noheadings --output TARGET \
            --types nfs,nfs4,cifs,smb3,fuse.sshfs,afs,ceph 2>/dev/null
}

case "${1:-}" in
    check) shift; cmd_check "$@" ;;
    run)   shift; cmd_run "$@" ;;
    paths) shift; cmd_paths "$@" ;;
    *)
        echo "usage: $0 {check [PATH] [SECONDS] | run SECONDS CMD... | paths}" >&2
        exit 2
        ;;
esac
