#!/bin/sh
set -eu
SUPPORT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. "$SUPPORT_DIR/installer_messages.sh"
RUNTIME_ROOT="${SPACR_USER_INSTALL_ROOT:-$HOME/Library/Application Support/spaCR}"
INSTALLER="$SUPPORT_DIR/install-online.sh"
LOCK_DIR="$RUNTIME_ROOT/.installing"
# The same owner test as the backend's spacr_lease_owner_dead: only a lease
# whose recorded owner is provably gone, and that holds nothing else, is
# left for the backend to take over.
lease_field() { sed -n "s/^$2=//p" "$1/owner" 2>/dev/null | head -n 1; }
lease_owner_dead() {
    [ -d "$1" ] && [ ! -L "$1" ] && [ -f "$1/owner" ] && [ ! -L "$1/owner" ] || return 1
    [ "$(ls -A "$1")" = "owner" ] || return 1
    pid="$(lease_field "$1" pid)"; started="$(lease_field "$1" started)"
    case "$pid" in ''|*[!0-9]*) return 1 ;; esac
    [ -n "$started" ] && [ "$(lease_field "$1" host)" = "$(uname -n)" ] || return 1
    now="$(ps -o lstart= -p "$pid" 2>/dev/null | tr -s ' \t' ' ' | sed 's/^ //; s/ $//')"
    [ -z "$now" ] || [ "$now" != "$started" ]
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
