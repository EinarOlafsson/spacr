#!/bin/sh
set -eu
SUPPORT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. "$SUPPORT_DIR/installer_messages.sh"
RUNTIME_ROOT="${SPACR_USER_INSTALL_ROOT:-$HOME/Library/Application Support/spaCR}"
INSTALLER="$SUPPORT_DIR/install-online.sh"
LOCK_DIR="$RUNTIME_ROOT/.installing"
if [ -e "$LOCK_DIR" ] || [ -L "$LOCK_DIR" ]; then
    spacr_say already_running
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
