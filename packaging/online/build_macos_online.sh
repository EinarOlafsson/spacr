#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "The macOS online installer must be built on macOS." >&2
    exit 2
fi

VERSION="$(python3 -c 'import ast,pathlib; t=ast.parse(pathlib.Path("setup.py").read_text()); print(next(ast.literal_eval(n.value) for n in t.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "VERSION" for x in n.targets)))')"
OUT_DIR="dist/online"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/spacr-online-pkg.XXXXXX")"
ROOT="$WORK/root"
SCRIPTS="$WORK/scripts"
APP="$ROOT/Applications/SpaCR.app"
SUPPORT="$ROOT/Library/Application Support/SpaCR"
OUT="$OUT_DIR/SpaCR-$VERSION-macOS-Universal-Online.pkg"
ICON_SOURCE="spacr/resources/icons/app_icon.png"
ICONSET="$WORK/SpaCR.iconset"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$SUPPORT" "$SCRIPTS" "$OUT_DIR"

if [[ ! -f "$ICON_SOURCE" ]]; then
    echo "Application icon not found: $ICON_SOURCE" >&2
    exit 3
fi
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$ICON_SOURCE" \
        --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    retina=$((size * 2))
    sips -z "$retina" "$retina" "$ICON_SOURCE" \
        --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/SpaCR.icns"

python3 packaging/i18n/render.py \
    --embed-unix packaging/online/install_spacr_unix.sh \
    --output "$SUPPORT/install-online.sh" \
    --version "$VERSION"
chmod 755 "$SUPPORT/install-online.sh"
cp packaging/online/generated/installer_messages.sh \
    "$SUPPORT/installer_messages.sh"

cat > "$APP/Contents/MacOS/SpaCR" <<'EOF'
#!/bin/sh
RUNTIME_ROOT="$HOME/Library/Application Support/SpaCR"
PYTHON="$RUNTIME_ROOT/venv/bin/python"
FIRST_RUN="/Library/Application Support/SpaCR/install-for-user.sh"

if [ ! -x "$PYTHON" ]; then
    osascript - "$FIRST_RUN" <<'APPLESCRIPT'
on run argv
    set helperPath to item 1 of argv
    tell application "Terminal"
        activate
        do script quoted form of helperPath
    end tell
end run
APPLESCRIPT
    exit 0
fi

exec "$PYTHON" -m spacr.qt "$@"
EOF
chmod 755 "$APP/Contents/MacOS/SpaCR"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key><string>spaCR</string>
  <key>CFBundleExecutable</key><string>SpaCR</string>
  <key>CFBundleIconFile</key><string>SpaCR.icns</string>
  <key>CFBundleIdentifier</key><string>com.einarolafsson.spacr</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>spaCR</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

cat > "$SUPPORT/uninstall-spacr.sh" <<'EOF'
#!/bin/sh
set -eu
. "/Library/Application Support/SpaCR/installer_messages.sh"
rm -f /usr/local/bin/spacr
rm -rf "/Applications/SpaCR.app"
rm -rf "/Library/Application Support/SpaCR"
spacr_say removed
EOF
chmod 755 "$SUPPORT/uninstall-spacr.sh"

cat > "$SUPPORT/install-for-user.sh" <<'EOF'
#!/bin/sh
set -eu

. "/Library/Application Support/SpaCR/installer_messages.sh"

RUNTIME_ROOT="${SPACR_USER_INSTALL_ROOT:-$HOME/Library/Application Support/SpaCR}"
INSTALLER="/Library/Application Support/SpaCR/install-online.sh"
LOCK_DIR="$RUNTIME_ROOT/.installing"

mkdir -p "$RUNTIME_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    spacr_say already_running
    exit 0
fi
cleanup() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$INSTALLER" \
  --platform macos \
  --install-root "$RUNTIME_ROOT" \
  --no-command-launcher \
  --no-launch

echo
spacr_say ready_opening
if [ "${SPACR_NO_RELAUNCH:-0}" != "1" ]; then
    open -a "/Applications/SpaCR.app"
fi
EOF
chmod 755 "$SUPPORT/install-for-user.sh"

cat > "$SCRIPTS/postinstall" <<EOF
#!/bin/sh
set -eu
mkdir -p /usr/local/bin
ln -sfn "/Applications/SpaCR.app/Contents/MacOS/SpaCR" /usr/local/bin/spacr
exit 0
EOF
chmod 755 "$SCRIPTS/postinstall"

codesign --force --deep --sign - "$APP"
pkgbuild \
    --root "$ROOT" \
    --scripts "$SCRIPTS" \
    --identifier "com.einarolafsson.spacr.online" \
    --version "$VERSION" \
    --install-location / \
    "$OUT"

if [[ -n "${PRODUCTSIGN_IDENTITY:-}" ]]; then
    SIGNED="$OUT_DIR/SpaCR-$VERSION-macOS-Universal-Online-signed.pkg"
    productsign --sign "$PRODUCTSIGN_IDENTITY" "$OUT" "$SIGNED"
    mv "$SIGNED" "$OUT"
fi

echo "Built $OUT"
