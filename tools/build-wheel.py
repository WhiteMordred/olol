#!/usr/bin/env python
"""Build a wheel package with protocol buffer files included."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_package():
    """Clean up the package."""
    # Remove generated files
    for pattern in ["ollama_pb2.py", "ollama_pb2_grpc.py"]:
        for file_path in Path("src").glob(f"**/{pattern}"):
            print(f"Removing {file_path}")
            file_path.unlink()
    
    # Clean build artifacts
    if Path("dist").exists():
        shutil.rmtree("dist")
        print("Cleaned dist directory")
    
    # Clean build directories
    for path in ["build", "*.egg-info"]:
        for dir_path in Path(".").glob(path):
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
                print(f"Cleaned {dir_path}")


def generate_proto():
    """Generate protocol buffer files."""
    proto_dir = Path("src/olol/proto")
    output_dir = Path("src/olol")
    
    # Ensure proto directory exists
    if not proto_dir.exists():
        print(f"Error: Proto directory {proto_dir} does not exist")
        return False
    
    # Find proto files
    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        print(f"No .proto files found in {proto_dir}")
        return False
    
    # Generate Python code for sync implementation using grpcio-tools
    print("Generating protocol buffer files...")
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
    ] + [str(f) for f in proto_files]
    
    try:
        subprocess.check_call(cmd)
        print("Successfully generated protocol buffer files")
        
        # Fix imports in generated files
        fix_imports(output_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating protocol buffer files: {e}")
        return False


def fix_imports(output_dir):
    """Fix imports in generated files."""
    for pb2_grpc_file in output_dir.glob("*_pb2_grpc.py"):
        with open(pb2_grpc_file, "r") as f:
            content = f.read()
        
        # Fix imports - change absolute to relative
        fixed_content = content.replace(
            "import ollama_pb2 as ollama__pb2",
            "from . import ollama_pb2 as ollama__pb2"
        )
        
        if fixed_content != content:
            with open(pb2_grpc_file, "w") as f:
                f.write(fixed_content)
            print(f"Fixed imports in {pb2_grpc_file}")


def build_package():
    """Build the Python package."""
    print("Building package...")
    try:
        # Try using uv if available
        try:
            subprocess.check_call([sys.executable, "-m", "uv", "build", "--no-isolation"])
        except subprocess.CalledProcessError:
            # Fall back to python -m build
            print("Falling back to standard build...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "build"])
            subprocess.check_call([sys.executable, "-m", "build", "--wheel"])
        print("Package built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building package: {e}")
        return False


def main():
    """Main entry point."""
    # Make sure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Clean up
    clean_package()
    
    # Generate proto files
    if not generate_proto():
        return 1
    
    # Build the package
    if not build_package():
        return 1
    
    print("\nBuild completed successfully!")
    print("The wheel package is available in the 'dist' directory.")
    print("To install the package:")
    print("  uv pip install dist/*.whl")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())