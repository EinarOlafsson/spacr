#!/bin/bash

# Set script to exit on error
set -e

# Define paths
PROJECT_ROOT="/home/carruthers/Documents/repo/spacr"
DOCS_SOURCE="$PROJECT_ROOT/docs/source"
MODULE_PATH="$PROJECT_ROOT/spacr"

# Clean old autodoc files
rm -f "$DOCS_SOURCE"/spacr*.rst

# Generate .rst files for all modules
sphinx-apidoc -o "$DOCS_SOURCE" "$MODULE_PATH" --force

# Build HTML docs
cd "$PROJECT_ROOT/docs"
make html

echo "Docs built successfully in: $PROJECT_ROOT/docs/build/html"