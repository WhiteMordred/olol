#!/bin/bash
# Script to regenerate protocol buffer files

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building protocol buffer files from $PROJECT_ROOT"

# Ensure we're in the project directory
cd "$PROJECT_ROOT"

# Clear any existing generated files
rm -f src/osync/ollama_pb2*.py
rm -f src/osync/async_impl/ollama_pb2*.py
rm -f src/ollama_pb2*.py  # Remove root-level pb2 files that might be causing conflicts
rm -rf dist/  # Clean dist directory

# Run the protoc tool
uv run -m osync.utils.protoc

echo "Protocol buffer files successfully generated"
echo "Don't forget to reinstall the package with: uv pip install -e ."