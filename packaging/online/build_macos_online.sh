#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "The macOS online installer must be built on macOS." >&2
    exit 2
fi

SIGNING_HELPER="packaging/online/macos_signing.py"
python3 "$SIGNING_HELPER" validate

VERSION="$(python3 -c 'import ast,pathlib; t=ast.parse(pathlib.Path("setup.py").read_text()); print(next(ast.literal_eval(n.value) for n in t.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "VERSION" for x in n.targets)))')"
read -r APP_VERSION APP_BUILD < <(python3 "$SIGNING_HELPER" bundle-versions "$VERSION")
OUT_DIR="dist/online"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/spacr-online-pkg.XXXXXX")"
ROOT="$WORK/root"
SCRIPTS="$WORK/scripts"
APP="$ROOT/Applications/spaCR.app"
SUPPORT="$ROOT/Library/Application Support/spaCR"
OUT="$OUT_DIR/spaCR-$VERSION-macOS-Universal-Online.pkg"
ICON_SOURCE="spacr/resources/icons/app_icon.png"
ICONSET="$WORK/spaCR.iconset"
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
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/spaCR.icns"

python3 packaging/i18n/render.py \
    --embed-unix packaging/online/install_spacr_unix.sh \
    --output "$SUPPORT/install-online.sh" \
    --version "$VERSION"
chmod 755 "$SUPPORT/install-online.sh"
cp packaging/online/generated/installer_messages.sh \
    "$SUPPORT/installer_messages.sh"

xcrun clang -Wall -Wextra -Werror -O2 -arch arm64 -arch x86_64 \
    -mmacosx-version-min=11.0 "-DSPACR_PACKAGE_VERSION=\"$VERSION\"" packaging/online/macos_launcher.c \
    -o "$APP/Contents/MacOS/spaCR"
lipo "$APP/Contents/MacOS/spaCR" -verify_arch arm64 x86_64
chmod 755 "$APP/Contents/MacOS/spaCR"
xcrun clang -Wall -Wextra -Werror -O2 -fobjc-arc -arch arm64 -arch x86_64 \
    -mmacosx-version-min=11.0 -framework Cocoa packaging/online/macos_installer.m \
    packaging/online/macos_gui_process.c -o "$APP/Contents/MacOS/install-spacr-gui"
lipo "$APP/Contents/MacOS/install-spacr-gui" -verify_arch arm64 x86_64
mkdir -p "$APP/Contents/Resources/installer-i18n"
cp packaging/i18n/*.json "$APP/Contents/Resources/installer-i18n/"
printf '%s\n' '{"native_uninstall_schema":1}' > "$APP/Contents/Resources/uninstall-capabilities.json"


cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key><string>spaCR</string>
  <key>CFBundleExecutable</key><string>spaCR</string>
  <key>CFBundleIconFile</key><string>spaCR.icns</string>
  <key>CFBundleIdentifier</key><string>com.einarolafsson.spacr</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>spaCR</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$APP_VERSION</string>
  <key>CFBundleVersion</key><string>$APP_BUILD</string>
  <key>SPACRPackageVersion</key><string>$VERSION</string>
  <key>CFBundleGetInfoString</key><string>spaCR $VERSION</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

cat > "$SUPPORT/uninstall-spacr.sh" <<'EOF'
#!/bin/sh
set -eu
. "/Library/Application Support/spaCR/installer_messages.sh"
# --purge also deletes spaCR's caches, backends and user data, after
# listing them and asking (spacr/install_cleanup.py purge).
for arg in "$@"; do
    if [ "$arg" = "--purge" ]; then
        yes=""
        for other in "$@"; do
            [ "$other" = "--yes" ] && yes="--yes"
        done
        py="$HOME/Library/Application Support/spaCR/venv/bin/python"
        if [ -x "$py" ]; then
            "$py" -m spacr.install_cleanup purge $yes || true
        fi
    fi
done
rm -f /usr/local/bin/spacr
rm -rf "/Applications/spaCR.app"
rm -rf "/Library/Application Support/spaCR"
spacr_say removed
EOF
chmod 755 "$SUPPORT/uninstall-spacr.sh"

cp spacr/install_cleanup.py "$SUPPORT/verify_runtime.py"
cp packaging/online/install_for_macos_user.sh "$SUPPORT/install-for-user.sh"
chmod 755 "$SUPPORT/install-for-user.sh"

# 1.4.9.9 installed one system-wide runtime into the support folder; every
# later layout keeps only helper scripts there. That runtime is obsolete
# application code nothing launches, so it goes. Its uv cache is kept.
cat > "$SCRIPTS/postinstall" <<'EOF'
#!/bin/sh
set -eu
legacy="/Library/Application Support/spaCR"
for part in venv python bootstrap; do
    if [ -d "$legacy/$part" ] && [ ! -L "$legacy/$part" ]; then
        rm -rf "$legacy/$part"
    fi
done
mkdir -p /usr/local/bin
ln -sfn "/Applications/spaCR.app/Contents/MacOS/spaCR" /usr/local/bin/spacr
exit 0
EOF
chmod 755 "$SCRIPTS/postinstall"

python3 "$SIGNING_HELPER" sign-app "$APP"
# Installer relocates a relocatable bundle to any copy with the same bundle
# identifier that Spotlight knows of -- a backup, a Downloads copy, a second
# drive -- and leaves /Applications/spaCR.app old or missing. Pin it.
COMPONENTS="$WORK/components.plist"
pkgbuild --analyze --root "$ROOT" "$COMPONENTS"
count="$(python3 -c 'import plistlib,sys; print(len(plistlib.load(open(sys.argv[1],"rb"))))' "$COMPONENTS")"
for ((index = 0; index < count; index++)); do
    plutil -replace "$index.BundleIsRelocatable" -bool NO "$COMPONENTS"
done
pkgbuild \
    --root "$ROOT" \
    --component-plist "$COMPONENTS" \
    --scripts "$SCRIPTS" \
    --identifier "com.einarolafsson.spacr.online" \
    --version "$VERSION" \
    --install-location / \
    "$OUT"

python3 "$SIGNING_HELPER" sign-package "$OUT"

echo "Built $OUT"
