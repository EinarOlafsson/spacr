#!/usr/bin/env bash
set -e

# 1) Ensure pdoc is installed
pip install --upgrade pdoc

# 2) Remove any old generated docs
rm -rf docs/site

# 3) Generate new HTML docs for the `spacr` package
pdoc -o docs/site spacr

echo "✅ Docs generated in docs/site"
