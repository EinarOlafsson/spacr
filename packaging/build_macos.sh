#!/usr/bin/env bash
# build_macos.sh — produce dist/SpaCR-<version>.dmg on macOS 11+
#
# Run from the spacr repo root:
#
#     ./packaging/build_macos.sh
#
# Prerequisites (checked below):
#   * macOS 11 (Big Sur) or newer
#   * python3.10+ on PATH
#   * spacr installed in the current environment
#   * pyinstaller >= 6.0
#   * hdiutil (ships with macOS)
#   * (optional) an Apple Developer ID for real code-signing — otherwise
#     the .app is signed ad-hoc and Gatekeeper will require right-click
#     "Open" on first launch.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script must run on macOS. For Windows use build_windows.ps1; for Debian use build_debian.sh." >&2
    exit 1
fi

echo "==> spacr macOS installer build"

# --- version ---
VERSION=$(python3 -c "import re,pathlib; s=pathlib.Path('setup.py').read_text(); m=re.search(r\"VERSION\s*=\s*['\\\"]([^'\\\"]+)\", s); print(m.group(1))")
echo "    version: $VERSION"

# --- clean previous outputs ---
rm -rf build dist

# --- deps ---
echo "==> installing build deps (pip)"
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade pyinstaller
python3 -m pip install -e .

# --- run PyInstaller ---
echo "==> running PyInstaller"
pyinstaller --noconfirm --clean packaging/spacr.spec

APP="dist/SpaCR.app"
if [[ ! -d "$APP" ]]; then
    echo "PyInstaller did not produce $APP" >&2
    exit 1
fi

# --- ad-hoc codesign so the app can launch without --disable-library-validation ---
echo "==> ad-hoc codesigning $APP"
codesign --force --deep --sign - "$APP"

# --- package into a .dmg via hdiutil ---
DMG_DIR=$(mktemp -d)
cp -R "$APP" "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"

DMG="dist/SpaCR-$VERSION.dmg"
echo "==> creating $DMG"
hdiutil create -fs HFS+ -volname "SpaCR $VERSION" \
    -srcfolder "$DMG_DIR" -ov -format UDZO \
    "$DMG"

rm -rf "$DMG_DIR"

echo "==> done: $DMG"
echo ""
echo "To publish for external users you must sign+notarize with your Apple Developer ID:"
echo "    codesign --deep --force --options runtime --sign 'Developer ID Application: YOUR NAME (TEAMID)' $APP"
echo "    xcrun notarytool submit $DMG --apple-id ... --team-id TEAMID --password ... --wait"
echo "    xcrun stapler staple $DMG"
