#!/usr/bin/env python
"""Setup script for OLOL."""

import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def generate_proto_files(package_dir):
    """Generate protocol buffer Python files."""
    proto_dir = os.path.join(package_dir, "src", "olol", "proto")
    output_dir = os.path.join(package_dir, "src", "olol")
    
    proto_files = [f for f in os.listdir(proto_dir) if f.endswith(".proto")]
    if not proto_files:
        print("No .proto files found in {}".format(proto_dir))
        return
    
    try:
        # Try to generate the protobuf files
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            "--proto_path={}".format(proto_dir),
            "--python_out={}".format(output_dir),
            "--grpc_python_out={}".format(output_dir),
        ]
        
        for proto_file in proto_files:
            cmd.append(os.path.join(proto_dir, proto_file))
        
        print("Generating protobuf files with command: {}".format(" ".join(cmd)))
        subprocess.check_call(cmd)
        print("Successfully generated protobuf files")
    except Exception as e:
        print("Error generating protobuf files: {}".format(str(e)))
        print("You may need to install grpcio-tools and run the build again.")

class CustomInstall(install):
    def run(self):
        # Generate proto files before install
        generate_proto_files(os.getcwd())
        install.run(self)

class CustomDevelop(develop):
    def run(self):
        # Generate proto files before develop install
        generate_proto_files(os.getcwd())
        develop.run(self)

if __name__ == "__main__":
    setup(
        # All packaging info is read from pyproject.toml
        cmdclass={
            'install': CustomInstall,
            'develop': CustomDevelop,
        },
    )