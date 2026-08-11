#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "The Linux online installer must be built on Linux." >&2
    exit 2
fi

VERSION="$(python3 -c 'import ast,pathlib; t=ast.parse(pathlib.Path("setup.py").read_text()); print(next(ast.literal_eval(n.value) for n in t.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "VERSION" for x in n.targets)))')"
OUT_DIR="dist/online"
OUT="$OUT_DIR/SpaCR-$VERSION-Linux-x86_64-Online.run"
mkdir -p "$OUT_DIR"
python3 packaging/i18n/render.py \
    --embed-unix packaging/online/install_spacr_unix.sh \
    --output "$OUT" \
    --version "$VERSION"
chmod 755 "$OUT"
echo "Built $OUT"
