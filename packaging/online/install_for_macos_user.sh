#!/bin/sh
set -eu
SUPPORT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. "$SUPPORT_DIR/installer_messages.sh"
RUNTIME_ROOT="${SPACR_USER_INSTALL_ROOT:-$HOME/Library/Application Support/spaCR}"
INSTALLER="$SUPPORT_DIR/install-online.sh"
LOCK_DIR="$RUNTIME_ROOT/.installing"
# The same owner test as the backend's spacr_lease_owner_dead: only this
# user's lease, written on this host in this pid namespace, holding nothing
# but its owner record, whose process the system says is gone (or carries
# another start stamp of the same kind), is left for the backend to take over.
lease_field() { sed -n "s/^$2=//p" "$1/owner" 2>/dev/null | head -n 1; }
lease_namespace() {
    if [ "$(uname -s)" = Linux ]; then readlink "/proc/$$/ns/pid" 2>/dev/null || echo unknown; fi
    return 0
}
lease_started() {
    if [ "$(uname -s)" = Linux ]; then
        text="$(sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20)" || text=""
        case "$text" in ''|*[!0-9]*) ;; *) printf 'ticks:%s\n' "$text" ;; esac
        return 0
    fi
    text="$(LC_ALL=C TZ=UTC0 ps -o lstart= -p "$1" 2>/dev/null | tr -s ' \t' ' ' | sed 's/^ //; s/ $//')" || text=""
    [ -z "$text" ] || printf 'lstart:%s\n' "$text"
    return 0
}
lease_gone() {
    kill -0 "$1" 2>/dev/null && return 1
    case "$( (LC_ALL=C; export LC_ALL; kill -0 "$1") 2>&1 )" in *"No such process"*) return 0 ;; esac
    return 1
}
lease_owner_dead() {
    [ -d "$1" ] && [ ! -L "$1" ] && [ -f "$1/owner" ] && [ ! -L "$1/owner" ] || return 1
    [ -O "$1" ] && [ -O "$1/owner" ] || return 1
    [ "$(ls -A "$1")" = "owner" ] || return 1
    ! grep -qvE '^(pid|started|host|pidns)=' "$1/owner" || return 1
    for key in pid started host pidns; do
        [ "$(grep -c "^$key=" "$1/owner")" = 1 ] || return 1
    done
    pid="$(lease_field "$1" pid)"; started="$(lease_field "$1" started)"
    case "$pid" in ''|0*|*[!0-9]*) return 1 ;; esac
    [ -n "$started" ] && [ "$(lease_field "$1" host)" = "$(uname -n)" ] || return 1
    namespace="$(lease_namespace)"
    [ "$namespace" != unknown ] && [ "$(lease_field "$1" pidns)" = "$namespace" ] || return 1
    lease_gone "$pid" && return 0
    now="$(lease_started "$pid")"
    [ -n "$now" ] && [ "${now%%:*}" = "${started%%:*}" ] && [ "$now" != "$started" ]
}
if { [ -e "$LOCK_DIR" ] || [ -L "$LOCK_DIR" ]; } && ! lease_owner_dead "$LOCK_DIR"; then
    spacr_say already_running
    owner="$(lease_field "$LOCK_DIR" pid)"
    if [ -n "$owner" ]; then spacr_say lease_held "$LOCK_DIR" "$owner"; else spacr_say lease_held_unknown "$LOCK_DIR"; fi
    exit 75
fi
trap 'exit 130' INT
trap 'exit 143' TERM
"$INSTALLER" --platform macos --install-root "$RUNTIME_ROOT" \
    --no-command-launcher --no-launch --consent-collected "$@"
spacr_say ready_opening
if [ "${SPACR_NO_RELAUNCH:-0}" != "1" ]; then
    open -a "/Applications/spaCR.app"
fi
