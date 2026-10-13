#!/bin/bash
# Run only on a disposable native runner. Arguments: artifact smoke.json,
# output directory, expected source commit, and scope (ubuntu-container/macos-hosted).
# Uses the account's real home and login shell; never edits a profile or adds PATH.
set -euo pipefail
[ "$#" -eq 4 ] || { echo "Usage: $0 RECEIPT OUTPUT COMMIT SCOPE" >&2; exit 2; }
receipt=$1
output=$2
commit=$3
scope=$4
case "$receipt:$output" in /*:/*) ;; *) echo 'Receipt and output must be absolute paths' >&2; exit 2 ;; esac
mkdir -p "$output"
output=$(cd "$output" && pwd -P)
jq_bin=$(command -v jq)
status=failed
stage=preconditions
install_exit=null
fresh_exit=null
profile=''
login_shell=''
hint=''
receipt_hash=''
native_hash=''
node_scope=''

hash_file() {
    # $1 is a file whose bytes should be SHA256 recorded, following a symlink.
    if [ "$(uname -s)" = Darwin ]; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -d ' ' -f 1
}

finish() {
    # $1 is the original process exit code, retained after writing its receipt.
    result=$1
    trap - EXIT
    "$jq_bin" -n --arg status "$status" --arg stage "$stage" --arg command "$hint" \
        --arg source_commit "$commit" --arg receipt_hash "$receipt_hash" \
        --arg profile "$profile" --arg shell "$login_shell" --arg scope "$scope" \
        --arg node_scope "$node_scope" --arg native_hash "$native_hash" \
        --argjson install_exit "$install_exit" --argjson fresh_exit "$fresh_exit" \
        --argjson exit_code "$result" \
        '{status:$status,stage:$stage,command:$command,source_commit:$source_commit,
          native_receipt_sha256:$receipt_hash,profile:$profile,login_shell:$shell,
          scope:$scope,node_scope:$node_scope,exit_code:$exit_code,
          install_exit_code:$install_exit,fresh_terminal_exit_code:$fresh_exit,
          installed_native_sha256:$native_hash,authenticated:false,
          path_registration_scope:"No verifier PATH additions or profile edits; normal startup files run, including vendor-made edits. On hosted macOS only, directories exposing Node tools are then excluded and logged.",
          terminal_scope:"New interactive login shell with real account profiles; no terminal-emulator window.",
          vendor_documentation:"https://code.claude.com/docs/en/setup",
          path_documentation:"https://code.claude.com/docs/en/troubleshoot-install#verify-your-path"}' \
        > "$output/acceptance.json" || result=1
    exit "$result"
}
trap 'finish "$?"' EXIT
[ "$(id -u)" -ne 0 ] || { echo 'Installer must run as a normal user' >&2; exit 1; }
case "$(uname -s):$scope" in
    Linux:ubuntu-container)
        expected_qt=xcb
        record=$(getent passwd "$(id -u)")
        profile=$(printf '%s\n' "$record" | cut -d : -f 6)
        login_shell=$(printf '%s\n' "$record" | cut -d : -f 7)
        node_scope='Clean Ubuntu container; nodejs/npm packages absent and commands unavailable in both actual login shells. Host Actions tooling is outside the container.'
        dpkg-query -W -f='${binary:Package}\n' > "$output/packages.log"
        if grep -Eq '^(nodejs|npm)(:.*)?$' "$output/packages.log"; then
            echo 'Node or npm package present in clean container' >&2; exit 1
        fi
        ;;
    Darwin:macos-hosted)
        expected_qt=cocoa
        record=$(/usr/bin/dscl . -read "/Users/$(id -un)" NFSHomeDirectory UserShell)
        profile=$(printf '%s\n' "$record" | sed -n 's/^NFSHomeDirectory: //p')
        login_shell=$(printf '%s\n' "$record" | sed -n 's/^UserShell: //p')
        node_scope='Hosted macOS retains Node tooling. After unchanged login profiles run, only directories exposing node/nodejs/npm/npx are excluded from each witness PATH. No path is added; physical Node absence is not claimed.'
        ;;
    *) echo 'Unsupported platform/scope' >&2; exit 1 ;;
esac
[ -n "$profile" ] && [ "$HOME" = "$profile" ] && [ -d "$profile" ] || {
    echo 'HOME must equal the actual OS account profile' >&2; exit 1;
}
case "$login_shell" in /bin/bash|/bin/zsh) ;; *) echo "Unreviewed account login shell: $login_shell" >&2; exit 1 ;; esac
[ -n "$commit" ] || { echo 'Expected source commit is required' >&2; exit 2; }
"$jq_bin" -e --arg commit "$commit" --arg qt "$expected_qt" \
    '.status == "passed" and .frozen == true and .source_commit == $commit and .qt_platform == $qt' \
    "$receipt" > /dev/null
hint=$("$jq_bin" -er '.claude_install_hint | select(type == "string")' "$receipt")
"$jq_bin" -e --arg expected 'curl -fsSL https://claude.ai/install.sh | bash' \
    '.claude_install_hint == $expected' "$receipt" > /dev/null || {
    echo 'Artifact hint changed; review the displayed command and vendor docs first' >&2; exit 1;
}
receipt_hash=$(hash_file "$receipt")
cp "$receipt" "$output/native-smoke.json"
printf '%s\n' "$hint" > "$output/pasted-command.sh"
for existing in .local/bin/claude .local/share/claude .claude .claude.json; do
    [ ! -e "$profile/$existing" ] && [ ! -L "$profile/$existing" ] || {
        echo "Fresh-profile precondition failed: $existing exists" >&2; exit 1;
    }
done

profile_inventory() {
    # $1 names the evidence file receiving hashes, never profile contents/secrets.
    for name in .profile .bash_profile .bash_login .bashrc .zprofile .zshrc .zshenv .zlogin; do
        if [ -f "$profile/$name" ]; then
            printf '%s %s\n' "$(hash_file "$profile/$name")" "$name"
        else
            printf 'absent %s\n' "$name"
        fi
    done > "$1"
}
profile_inventory "$output/profile-before.log"
mkdir "$output/empty-cwd"
cat > "$output/login-witness.sh" <<'WITNESS'
# Sourced inside a fresh real login shell after its normal startup files run.
printf 'uid=%s\nhome=%s\nshell=%s\npath=%s\n' "$(id -u)" "$HOME" "$SHELL" "$PATH"
[ "$HOME" = "$ACCEPTANCE_PROFILE" ] && [ "$(id -u)" = "$ACCEPTANCE_UID" ] || exit 20
if [ "$ACCEPTANCE_SCOPE" = macos-hosted ]; then
    remaining_path=$PATH
    filtered_path=''
    have_entry=false
    removed_paths="$ACCEPTANCE_OUTPUT/node-path-exclusions-$ACCEPTANCE_PHASE.log"
    : > "$removed_paths"
    while :; do
        case "$remaining_path" in
            *:*) entry=${remaining_path%%:*}; remaining_path=${remaining_path#*:}; more=true ;;
            *) entry=$remaining_path; more=false ;;
        esac
        exposes_node=false
        for tool in node nodejs npm npx; do
            if [ -x "${entry:-.}/$tool" ]; then exposes_node=true; break; fi
        done
        if [ "$exposes_node" = true ]; then
            printf '%s\n' "$entry" >> "$removed_paths"
        elif [ "$have_entry" = true ]; then
            filtered_path="$filtered_path:$entry"
        else
            filtered_path=$entry
            have_entry=true
        fi
        [ "$more" = true ] || break
    done
    [ "$have_entry" = true ] || exit 21
    PATH=$filtered_path
    export PATH
    hash -r
    printf 'node_excluded_path=%s\n' "$PATH"
fi
for tool in node nodejs npm npx; do
    if command -v "$tool" >/dev/null 2>&1; then
        command -v "$tool"
        echo "No-Node precondition failed after login startup: $tool" >&2; exit 21
    fi
done
printf 'node_unavailable=true\nnpm_unavailable=true\n'
[ -z "${ANTHROPIC_API_KEY:-}${CLAUDE_CODE_OAUTH_TOKEN:-}${ANTHROPIC_AUTH_TOKEN:-}" ] || exit 22
cd "$ACCEPTANCE_OUTPUT/empty-cwd" || exit 23
if [ "$ACCEPTANCE_PHASE" = install ]; then
    if command -v claude >/dev/null 2>&1; then echo 'Claude already resolves before install' >&2; exit 24; fi
    command -v curl >/dev/null && command -v bash >/dev/null || exit 25
    . "$ACCEPTANCE_OUTPUT/pasted-command.sh"
    exit "$?"
fi
resolved=$(command -v claude) || {
    echo 'Fresh login cannot resolve claude; the product command did not establish terminal PATH' >&2; exit 26;
}
printf '%s\n' "$resolved" > "$ACCEPTANCE_OUTPUT/resolved.log"
[ "$resolved" = "$HOME/.local/bin/claude" ] || { echo 'Fresh login resolved a different Claude' >&2; exit 27; }
claude --version > "$ACCEPTANCE_OUTPUT/version.log" 2>&1
exit "$?"
WITNESS

login_witness() {
    # $1 selects install/fresh. Only real OS identity, a base system PATH, and
    # evidence paths enter the new shell; login startup may change PATH itself.
    /usr/bin/env -i HOME="$profile" USER="$(id -un)" LOGNAME="$(id -un)" \
        SHELL="$login_shell" PATH=/usr/bin:/bin:/usr/sbin:/sbin TERM=dumb LANG=C \
        ACCEPTANCE_PROFILE="$profile" ACCEPTANCE_UID="$(id -u)" \
        ACCEPTANCE_OUTPUT="$output" ACCEPTANCE_PHASE="$1" ACCEPTANCE_SCOPE="$scope" \
        "$login_shell" -lic '. "$ACCEPTANCE_OUTPUT/login-witness.sh"'
}
stage=exact-command
install_exit=0
login_witness install > "$output/install.log" 2>&1 || install_exit=$?
profile_inventory "$output/profile-after.log"
[ "$install_exit" -eq 0 ] || { cat "$output/install.log" >&2; exit 1; }
stage=native-payload
launcher="$profile/.local/bin/claude"
[ -L "$launcher" ] && [ -x "$launcher" ] || { echo 'Native launcher symlink missing' >&2; exit 1; }
readlink "$launcher" > "$output/launcher-target.log"
file -L "$launcher" > "$output/native-file.log"
case "$(uname -s)" in
    Linux) grep -q 'ELF .*executable' "$output/native-file.log" ;;
    Darwin) grep -q 'Mach-O' "$output/native-file.log" ;;
esac
native_hash=$(hash_file "$launcher")
stage=fresh-login
fresh_exit=0
login_witness fresh > "$output/fresh-login.log" 2>&1 || fresh_exit=$?
[ "$fresh_exit" -eq 0 ] || { cat "$output/fresh-login.log" >&2; exit 1; }
[ "$(hash_file "$launcher")" = "$native_hash" ] || { echo 'Native payload changed across fresh-shell check' >&2; exit 1; }
grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+.*Claude Code' "$output/version.log"
status=passed
stage=complete
