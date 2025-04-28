# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "grpc_tools",
#     "protoc",
# ]
# ///

import hashlib
import os
import sys
import time

try:
    from grpc_tools import protoc
except ImportError:
    raise ImportError(
        "grpc_tools package is not installed. "
        "Please install it using: pip install grpcio-tools"
    )

# Current API version - increment when making proto changes
API_VERSION = "1.2.0"

def calculate_file_hash(file_path):
    """Calculate the SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None
        
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_proto_version(proto_path, generated_path):
    """Check if the proto file has changed since last generation."""
    # Check proto file hash vs stored hash
    proto_hash = calculate_file_hash(proto_path)
    
    # Path to hash storage file
    hash_file = os.path.join(os.path.dirname(generated_path), ".proto_hash")
    
    # Check if hash file exists
    if not os.path.exists(hash_file):
        # First time build
        return True
    
    # Read stored hash
    with open(hash_file, 'r') as f:
        stored_hash = f.read().strip()
    
    # Compare hashes
    if proto_hash != stored_hash:
        print(f"Proto file changed (hash: {proto_hash[:8]} != {stored_hash[:8]})")
        return True
    
    # Check if _pb2.py file exists
    if not os.path.exists(generated_path):
        print(f"Generated file {generated_path} missing")
        return True
    
    # All checks passed, no need to rebuild
    return False

def update_hash_file(proto_path, generated_path):
    """Update the hash file after successful build."""
    proto_hash = calculate_file_hash(proto_path)
    hash_file = os.path.join(os.path.dirname(generated_path), ".proto_hash")
    
    with open(hash_file, 'w') as f:
        f.write(proto_hash)
    
    print(f"Updated proto hash: {proto_hash[:8]}")

def build(ctx):
    """Build protocol buffer files with version checks."""
    proto_file = "src/olol/proto/ollama.proto"
    generated_file = "src/olol/proto/ollama_pb2.py"
    
    # Check if we need to rebuild the proto files
    if not check_proto_version(proto_file, generated_file):
        print(f"Proto files up to date (API version: {API_VERSION})")
        return 0
    
    print(f"Building proto files (API version: {API_VERSION})...")
    
    # Add versioning tag to the proto file
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(proto_file, 'r') as f:
        proto_content = f.read()
    
    # Make sure version in file matches API_VERSION
    if f"Current version: v{API_VERSION}" not in proto_content:
        print(f"WARNING: Proto file version doesn't match API_VERSION ({API_VERSION})")
        
    # protoc.main expects a list of arguments
    protoc_args = [
        "grpc_tools.protoc",
        "-I=src/olol",  # include path
        "--python_out=src/olol",
        "--grpc_python_out=src/olol",
        proto_file
    ]
    
    # Invoke protoc
    result = protoc.main(protoc_args)
    
    if result == 0:
        # Success - update hash file
        update_hash_file(proto_file, generated_file)
        print(f"Proto build successful: {generated_file}")
    else:
        print(f"Proto build failed with code: {result}", file=sys.stderr)
    
    return result

if __name__ == "__main__":
    sys.exit(build(None))

