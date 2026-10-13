#!/usr/bin/env bash
# build_macos.sh — produce dist/spaCR-<version>.dmg on macOS 11+
#
# Run from the spacr repo root:
#
#     ./packaging/build_macos.sh
#
# Prerequisites (checked below):
#   * macOS 11 (Big Sur) or newer
#   * python3.9+ on PATH
#   * spacr installed in the current environment
#   * pyinstaller >= 6.10
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
if [[ ${SPACR_SKIP_BUILD_DEPENDENCIES:-0} != 1 ]]; then
    python3 -m pip install --upgrade pip
    python3 -m pip install 'pyinstaller>=6.10,<7'
    python3 -m pip install .
fi

# --- run PyInstaller ---
echo "==> running PyInstaller"
python3 -m PyInstaller --noconfirm --clean packaging/spacr.spec

APP="dist/spaCR.app"
if [[ ! -d "$APP" ]]; then
    echo "PyInstaller did not produce $APP" >&2
    exit 1
fi

python3 - "$APP" "$VERSION" <<'BUNDLE_VERSION'
"""Bind both native bundle version fields to the already selected release."""
from pathlib import Path
import plistlib
import re
import sys

path = Path(sys.argv[1]) / "Contents/Info.plist"
with path.open("rb") as stream:
    info = plistlib.load(stream)
version = sys.argv[2]
if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,3}", version):
    raise ValueError("unsupported release version for a native macOS bundle")
parts = [int(part) for part in version.split(".")]
parts += [0] * (4 - len(parts))
major, minor, patch, revision = parts
if not 1 <= major <= 99 or any(not 0 <= part <= 99 for part in parts[1:]):
    raise ValueError("native bundle version mapping supports major1..99 and remaining components0..99")
info["CFBundleShortVersionString"] = f"{major}.{minor}.{patch}"
info["CFBundleVersion"] = f"{major * 100 + minor}.{patch}.{revision}"
info["SPACRPackageVersion"] = version
with path.open("wb") as stream:
    plistlib.dump(info, stream)
BUNDLE_VERSION

# --- ad-hoc codesign so the app can launch without --disable-library-validation ---
echo "==> ad-hoc codesigning $APP"
codesign --force --deep --sign - "$APP"

# --- package into a .dmg via hdiutil ---
DMG_DIR=$(mktemp -d)
cp -R "$APP" "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"

DMG="dist/spaCR-$VERSION.dmg"
echo "==> creating $DMG"
hdiutil create -fs HFS+ -volname "spaCR $VERSION" \
    -srcfolder "$DMG_DIR" -ov -format UDZO \
    "$DMG"

rm -rf "$DMG_DIR"

echo "==> done: $DMG"
echo ""
echo "To publish for external users you must sign+notarize with your Apple Developer ID:"
echo "    codesign --deep --force --options runtime --sign 'Developer ID Application: YOUR NAME (TEAMID)' $APP"
echo "    xcrun notarytool submit $DMG --apple-id ... --team-id TEAMID --password ... --wait"
echo "    xcrun stapler staple $DMG"
