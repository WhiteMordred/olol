#!/bin/bash
# Build a release-ready wheel package

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure we're in the project directory
cd "$SCRIPT_DIR"

# Run the build-wheel.py script
python tools/build-wheel.py

echo ""
echo "Release build complete!"
echo "Wheel file is in the 'dist' directory"
echo ""