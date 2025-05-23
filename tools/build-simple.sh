#!/bin/bash
# Simple build script that directly generates and packages

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure we're in the project directory
cd "$PROJECT_ROOT"

# Clean up
rm -f src/osync/ollama_pb2*.py
rm -rf dist/
mkdir -p dist/

echo "Generating protobuf files..."
python -m grpc_tools.protoc \
    --proto_path=src/osync/proto \
    --python_out=src/osync \
    --grpc_python_out=src/osync \
    src/osync/proto/ollama.proto

echo "Fixing imports..."
sed -i 's/import ollama_pb2/from . import ollama_pb2/g' src/osync/ollama_pb2_grpc.py

echo "Building wheel..."
# Create wheel directory structure
WHEEL_DIR="dist/wheel_build"
mkdir -p "$WHEEL_DIR/osync"

# Copy Python files
cp -r src/osync/* "$WHEEL_DIR/osync/"

# Copy proto files
mkdir -p "$WHEEL_DIR/osync/proto"
cp src/osync/proto/*.proto "$WHEEL_DIR/osync/proto/"

# Create wheel metadata
mkdir -p "$WHEEL_DIR/osync-0.1.0.dist-info"
cat > "$WHEEL_DIR/osync-0.1.0.dist-info/METADATA" << EOF
Metadata-Version: 2.1
Name: osync
Version: 0.1.0
Summary: Ollama gRPC interface with sync/async support for distributed clustering
Author: Shane Macaulay
Author-email: ktwo@ktwo.ca
Requires-Python: >=3.12
Description-Content-Type: text/markdown
License-File: LICENSE
Requires-Dist: grpcio>=1.62.1
Requires-Dist: grpcio-tools>=1.62.1
Requires-Dist: protobuf>=4.25.3
Requires-Dist: grpclib>=0.4.7
Requires-Dist: flask>=2.0.0
Requires-Dist: aiohttp>=3.8.0

# Ollama Sync Load Balancing and Clustering

A distributed inference system that allows you to build a powerful multi-host cluster for Ollama AI models.
EOF

cat > "$WHEEL_DIR/osync-0.1.0.dist-info/WHEEL" << EOF
Wheel-Version: 1.0
Generator: osync-build
Root-Is-Purelib: true
Tag: py3-none-any
EOF

# Create entry point file
mkdir -p "$WHEEL_DIR/osync-0.1.0.dist-info/scripts"
cat > "$WHEEL_DIR/osync-0.1.0.dist-info/entry_points.txt" << EOF
[console_scripts]
osync = osync.__main__:main
osync-protoc = osync.utils.protoc:build
EOF

# Generate RECORD file (required for pip installation)
echo "Creating RECORD file..."
# Change to wheel directory
cd "$WHEEL_DIR"
# First generate a list of all files with placeholders for hashes
find osync osync-0.1.0.dist-info -type f | sort | while read -r file; do
    echo "$file,,0" >> "osync-0.1.0.dist-info/RECORD"
done
# Add the RECORD file itself
echo "osync-0.1.0.dist-info/RECORD,,0" >> "osync-0.1.0.dist-info/RECORD"
# Return to previous directory
cd - > /dev/null

# Create the wheel file (using zip)
cd "$WHEEL_DIR"
zip -r ../osync-0.1.0-py3-none-any.whl osync osync-0.1.0.dist-info
cd "$PROJECT_ROOT"

# Clean up temporary directory
rm -rf "$WHEEL_DIR"

echo ""
echo "Build complete!"
echo "Wheel file: dist/osync-0.1.0-py3-none-any.whl"
echo ""
echo "To install, run: uv pip install dist/osync-0.1.0-py3-none-any.whl"