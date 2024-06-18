#!/bin/bash

# Exit if any subcommand fails
set -e

# Build the documentation
cd docs
make clean
make html

# Copy the generated files to the root of the docs directory
cp -r build/html/* .

# Add and commit the changes
git add .
git commit -m "Update documentation"
git push origin main

