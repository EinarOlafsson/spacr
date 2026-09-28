#!/usr/bin/env bash
# Build a self-contained CPU .deb from the shared onedir PyInstaller spec.
# Build on Ubuntu 22.04 to retain the advertised Ubuntu 22.04 / Debian 12 floor.
set -euo pipefail
[[ $(uname -s) == Linux && -f /etc/debian_version ]] || {
    echo 'This builder requires Debian/Ubuntu Linux.' >&2; exit 1;
}
[[ $(dpkg --print-architecture) == amd64 ]] || {
    echo 'This package currently supports amd64 only.' >&2; exit 1;
}
build_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$build_root"
apt_runner=()
[[ $EUID == 0 ]] || apt_runner=(sudo)
needed=(python3 python3-venv libpython3-dev binutils dpkg-dev libgl1 libegl1 libglib2.0-0 \
        libx11-6 libxcb1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 \
        libxcb-xinerama0 libxcb-xkb1 libfontconfig1 libfreetype6 libdbus-1-3 libgomp1)
missing=()
for package in "${needed[@]}"; do
    dpkg -s "$package" >/dev/null 2>&1 || missing+=("$package")
done
if (( ${#missing[@]} )); then
    "${apt_runner[@]}" apt-get update
    "${apt_runner[@]}" apt-get install -y "${missing[@]}"
fi
build_env=$(mktemp -d "${TMPDIR:-/tmp}/spacr-debian-build.XXXXXXXX")
trap 'rm -rf -- "$build_env"' EXIT
python3 -m venv "$build_env/venv"
build_python="$build_env/venv/bin/python"
"$build_python" -m pip install --upgrade pip
"$build_python" -m pip install 'pyinstaller>=6.10,<7'
"$build_python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$build_python" -m pip install .
"$build_python" -m pip check
"$build_python" -c 'import torch; assert torch.version.cuda is None'
version=$("$build_python" packaging/release.py version)
mkdir -p acceptance
"$build_python" -m pip freeze > acceptance/debian-build-dependencies.txt
"$build_python" -m PyInstaller --noconfirm --clean packaging/spacr.spec
[[ -x dist/spacr/spacr ]] || { echo 'The frozen executable is missing.' >&2; exit 1; }
stage=build/debian-root
rm -rf -- "$stage"
mkdir -p "$stage"/{DEBIAN,opt/spacr,usr/bin,usr/share/applications,usr/share/icons/hicolor}
cp -a dist/spacr/. "$stage/opt/spacr/"
cat > "$stage/usr/bin/spacr" <<'LAUNCHER'
#!/bin/sh
exec /opt/spacr/spacr "$@"
LAUNCHER
chmod 755 "$stage/usr/bin/spacr"
install -m644 packaging/linux/io.github.olafssonlab.spacr.desktop "$stage/usr/share/applications/"
cp -a packaging/linux/icons/hicolor/. "$stage/usr/share/icons/hicolor/"
installed_size=$(du -sk "$stage/opt/spacr" | cut -f1)
build_libc=$(getconf GNU_LIBC_VERSION | cut -d' ' -f2)
cp /etc/os-release acceptance/debian-build-os.txt
cat > "$stage/DEBIAN/control" <<CONTROL
Package: spacr
Version: $version
Section: science
Priority: optional
Architecture: amd64
Maintainer: Einar Olafsson <einar.olafsson@gmail.com>
Installed-Size: $installed_size
Depends: libc6 (>= $build_libc), libgcc-s1, libstdc++6, libgl1, libegl1, libglib2.0-0, libx11-6, libxcb1, libxkbcommon0, libxkbcommon-x11-0, libxcb-cursor0, libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-render-util0, libxcb-shape0, libxcb-xinerama0, libxcb-xkb1, libfontconfig1, libfreetype6, libdbus-1-3, libgomp1
Description: spaCR microscopy analysis application with a private CPU runtime
 Bundles Python, Qt and the declared scientific environment under /opt/spacr.
 System Python is not modified. User preferences and analysis outputs are
 outside the package and are preserved when it is removed.
CONTROL
cp "$stage/DEBIAN/control" acceptance/debian-control.txt
# dpkg owns every installed file; removal needs no custom destructive script.
dpkg-deb --root-owner-group --build "$stage" "dist/spacr_${version}_amd64.deb"
