#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "The macOS online installer must be built on macOS." >&2
    exit 2
fi

VERSION="$(python3 setup.py --version)"
OUT_DIR="dist/online"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/spacr-online-pkg.XXXXXX")"
ROOT="$WORK/root"
SCRIPTS="$WORK/scripts"
APP="$ROOT/Applications/SpaCR.app"
SUPPORT="$ROOT/Library/Application Support/SpaCR"
OUT="$OUT_DIR/SpaCR-$VERSION-macOS-Universal-Online.pkg"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$SUPPORT" "$SCRIPTS" "$OUT_DIR"

sed "s/@SPACR_VERSION@/$VERSION/g" \
    packaging/online/install_spacr_unix.sh > "$SUPPORT/install-online.sh"
chmod 755 "$SUPPORT/install-online.sh"

cat > "$APP/Contents/MacOS/SpaCR" <<'EOF'
#!/bin/sh
exec "/Library/Application Support/SpaCR/venv/bin/python" -m spacr.qt "$@"
EOF
chmod 755 "$APP/Contents/MacOS/SpaCR"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key><string>spaCR</string>
  <key>CFBundleExecutable</key><string>SpaCR</string>
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
rm -f /usr/local/bin/spacr
rm -rf "/Applications/SpaCR.app"
rm -rf "/Library/Application Support/SpaCR"
echo "spaCR was removed. User-created data and preferences were left in place."
EOF
chmod 755 "$SUPPORT/uninstall-spacr.sh"

cat > "$SCRIPTS/postinstall" <<EOF
#!/bin/sh
set -eu
"/Library/Application Support/SpaCR/install-online.sh" \
  --platform macos \
  --install-root "/Library/Application Support/SpaCR" \
  --no-launch
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
