#!/bin/bash
# Script to build the package with protocol buffer files

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure we're in the project directory
cd "$PROJECT_ROOT"

# Step 1: Generate protocol buffer files
echo "Generating protocol buffer files..."
./tools/protoc-build.sh

# Step 2: Install locally to make sure it works
echo "Installing package in development mode..."
uv pip install -e .

# Step 3: Clean any previous build artifacts
echo "Cleaning previous build artifacts..."
rm -rf dist/

# Step 4: Build the package
echo "Building package..."
uv build

echo "Build complete. Wheel package is available in the dist/ directory."
echo "To install: uv pip install dist/*.whl"