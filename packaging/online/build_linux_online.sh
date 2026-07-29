#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "The Linux online installer must be built on Linux." >&2
    exit 2
fi

VERSION="$(python3 setup.py --version)"
OUT_DIR="dist/online"
OUT="$OUT_DIR/SpaCR-$VERSION-Linux-x86_64-Online.run"
mkdir -p "$OUT_DIR"
sed "s/@SPACR_VERSION@/$VERSION/g" \
    packaging/online/install_spacr_unix.sh > "$OUT"
chmod 755 "$OUT"
echo "Built $OUT"
