#!/usr/bin/env bash
# build_debian.sh — build a Debian package installable via apt.
#
# Run from the spacr repo root on Debian 12 / Ubuntu 22.04+ (or a
# `debian:12` docker image):
#
#     ./packaging/build_debian.sh
#
# Output:
#     dist/spacr_<version>_amd64.deb
#
# Install with:
#     sudo apt install ./dist/spacr_<version>_amd64.deb
#
# Prerequisites (installed automatically if missing):
#     python3 python3-pip python3-venv python3-stdeb
#     dpkg-dev debhelper dh-python fakeroot

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This script must run on Linux. For Windows use build_windows.ps1; for macOS use build_macos.sh." >&2
    exit 1
fi
if [[ ! -f /etc/debian_version ]]; then
    echo "This script targets Debian/Ubuntu. On other distros build a wheel with 'python -m build' and install via pip." >&2
    exit 1
fi

echo "==> spacr Debian package build"

# --- ensure build toolchain present ---
NEEDED_PKGS=(python3 python3-pip python3-venv python3-stdeb dpkg-dev debhelper dh-python fakeroot)
MISSING=()
for pkg in "${NEEDED_PKGS[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING+=("$pkg")
    fi
done
if (( ${#MISSING[@]} > 0 )); then
    echo "==> installing ${MISSING[*]} via apt (needs sudo)"
    sudo apt-get update
    sudo apt-get install -y "${MISSING[@]}"
fi

# --- version ---
VERSION=$(python3 -c "import re,pathlib; s=pathlib.Path('setup.py').read_text(); m=re.search(r\"VERSION\s*=\s*['\\\"]([^'\\\"]+)\", s); print(m.group(1))")
echo "    version: $VERSION"

rm -rf deb_dist dist/*.deb

# --- generate debian/ tree via stdeb ---
# stdeb reads setup.py's install_requires and pins the deps into the .deb.
# The runtime system libraries (libgl1 etc.) are added by the --extra-cfg-file
# below since they aren't Python deps.
cat > /tmp/spacr-stdeb.cfg <<'CFG'
[DEFAULT]
Package: spacr
Depends3: python3 (>= 3.10), libgl1, libglib2.0-0, libsm6, libxext6, libxrender1, libxft2, libtk8.6, python3-tk
Suggests3: python3-torch, python3-torchvision
XS-Python-Version: >= 3.10
CFG

echo "==> running stdeb"
python3 setup.py \
    --command-packages=stdeb.command \
    sdist_dsc --extra-cfg-file=/tmp/spacr-stdeb.cfg \
    bdist_deb

# --- move final .deb into dist/ ---
mkdir -p dist
DEB=$(ls deb_dist/*_amd64.deb 2>/dev/null | head -n1 || true)
if [[ -z "$DEB" ]]; then
    # fall back to any generated .deb
    DEB=$(ls deb_dist/*.deb 2>/dev/null | head -n1 || true)
fi
if [[ -z "$DEB" || ! -f "$DEB" ]]; then
    echo "stdeb did not produce a .deb — check deb_dist/ for errors" >&2
    exit 2
fi

TARGET="dist/spacr_${VERSION}_amd64.deb"
cp "$DEB" "$TARGET"
echo "==> wrote $TARGET"

echo ""
echo "Install with:"
echo "    sudo apt install ./$TARGET"
echo ""
echo "The GUI launcher is registered as:"
echo "    /usr/bin/spacr    (equivalent to python3 -m spacr)"
