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
lease_group_live_member() {
    listing="$(ps -A -o pid=,ppid=,pgid=,uid= 2>/dev/null)" || { echo unknown; return 0; }
    [ -n "$listing" ] || { echo unknown; return 0; }
    printf '%s\n' "$listing" | awk -v group="$1" -v me="$$" -v uid="$(id -u)" '
        { pid[NR] = $1; parent[$1] = $2; pg[$1] = $3; owner[$1] = $4 }
        END {
            for (p = me; p != "" && p != 0 && !(p in mine); p = parent[p]) mine[p] = 1
            for (i = 1; i <= NR; i++) {
                p = pid[i]
                if (pg[p] != group || owner[p] != uid || (p in mine)) continue
                steps = 0
                for (q = parent[p]; q != "" && q != 0 && q != 1 && steps++ < 4096; q = parent[q]) if (q == me) break
                if (q == me) continue
                print p; exit
            }
        }'
}
lease_owner_dead() {
    [ -d "$1" ] && [ ! -L "$1" ] && [ -f "$1/owner" ] && [ ! -L "$1/owner" ] || return 1
    [ -O "$1" ] && [ -O "$1/owner" ] || return 1
    [ "$(ls -A "$1")" = "owner" ] || return 1
    ! grep -qvE '^(pid|started|host|pidns|pgid)=' "$1/owner" || return 1
    for key in pid started host pidns; do
        [ "$(grep -c "^$key=" "$1/owner")" = 1 ] || return 1
    done
    [ "$(grep -c '^pgid=' "$1/owner")" -le 1 ] || return 1
    pid="$(lease_field "$1" pid)"; started="$(lease_field "$1" started)"
    case "$pid" in ''|0*|*[!0-9]*) return 1 ;; esac
    [ -n "$started" ] && [ "$(lease_field "$1" host)" = "$(uname -n)" ] || return 1
    namespace="$(lease_namespace)"
    [ "$namespace" != unknown ] && [ "$(lease_field "$1" pidns)" = "$namespace" ] || return 1
    if ! lease_gone "$pid"; then
        now="$(lease_started "$pid")"
        [ -n "$now" ] && [ "${now%%:*}" = "${started%%:*}" ] && [ "$now" != "$started" ] || return 1
    fi
    group="$(lease_field "$1" pgid)"
    [ -z "$group" ] && return 0
    case "$group" in 0*|*[!0-9]*) return 1 ;; esac
    [ -z "$(lease_group_live_member "$group")" ]
}
if { [ -e "$LOCK_DIR" ] || [ -L "$LOCK_DIR" ]; } && ! lease_owner_dead "$LOCK_DIR"; then
    spacr_say already_running
    owner="$(lease_field "$LOCK_DIR" pid)"
    if [ -n "$owner" ] && lease_gone "$owner"; then
        group="$(lease_field "$LOCK_DIR" pgid)"
        member=""
        [ -z "$group" ] || member="$(lease_group_live_member "$group")"
        case "$member" in
            ''|*[!0-9]*)
                if grep -qvE '^(pid|started|host|pidns|pgid)=' "$LOCK_DIR/owner" 2>/dev/null; then owner=""; fi ;;
            *) owner="$member" ;;
        esac
    fi
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
