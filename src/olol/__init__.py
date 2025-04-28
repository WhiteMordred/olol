"""OLOL - Ollama gRPC interface with sync/async support for distributed inference."""

__version__ = "0.1.0"

import importlib.util
import os
import sys

# Check if protobuf modules are generated, generate them if needed
if not os.path.exists(os.path.join(os.path.dirname(__file__), "ollama_pb2.py")):
    # The .proto file should be included in the package, if it exists, we can generate the pb2 files
    proto_file = os.path.join(os.path.dirname(__file__), "proto", "ollama.proto")
    if os.path.exists(proto_file):
        import logging
        import subprocess
        import warnings
        
        warnings.warn("Protobuf files not found. Attempting to generate them...")
        try:
            # Try to generate the protobuf files
            from grpc_tools import protoc
            
            package_dir = os.path.dirname(__file__)
            proto_dir = os.path.join(package_dir, "proto")
            
            # Generate sync implementation
            args = [
                protoc.__file__,
                f"--proto_path={proto_dir}",
                f"--python_out={package_dir}",
                f"--grpc_python_out={package_dir}",
                proto_file
            ]
            
            # Strip .py from protoc.__file__
            if args[0].endswith(".py"):
                args[0] = args[0][:-3]
                
            code = protoc.main(args)
            
            if code != 0:
                warnings.warn(f"Failed to generate protobuf files, code {code}")
        except Exception as e:
            warnings.warn(f"Error generating protobuf files: {str(e)}")

# Import will be handled within modules as needed to avoid circular dependencies
# We don't import the ollama_pb2 modules here to prevent import errors if they're not ready yet