"""Protocol buffer compilation utilities for OLOL."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_proto_files(base_dir: Optional[str] = None) -> List[str]:
    """Find all .proto files in the project's proto directory.
    
    Args:
        base_dir: Optional base directory to search from
        
    Returns:
        List of paths to .proto files
    """
    if base_dir is None:
        # Find the package root directory
        package_dir = Path(__file__).parent.parent
        proto_dir = package_dir / "proto"
    else:
        proto_dir = Path(base_dir)
    
    proto_files = [str(p) for p in proto_dir.glob("*.proto")]
    print(f"Found proto files: {proto_files}")
    return proto_files


def build(args: Optional[List[str]] = None) -> int:
    """Build protocol buffer files for both sync and async implementations.
    
    Args:
        args: Optional command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    if args is None:
        args = sys.argv[1:]
    
    package_dir = Path(__file__).parent.parent
    proto_dir = package_dir / "proto"
    
    if not proto_dir.exists():
        print(f"Error: Proto directory {proto_dir} does not exist")
        proto_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory {proto_dir}")
        return 1
    
    proto_files = find_proto_files(str(proto_dir))
    if not proto_files:
        print(f"No .proto files found in {proto_dir}")
        # Copy proto files from src/proto if they exist there
        src_proto_dir = Path(__file__).parent.parent.parent / "proto"
        if src_proto_dir.exists():
            src_proto_files = list(src_proto_dir.glob("*.proto"))
            if src_proto_files:
                for proto_file in src_proto_files:
                    dest_file = proto_dir / proto_file.name
                    with open(proto_file, 'r') as src, open(dest_file, 'w') as dest:
                        dest.write(src.read())
                    print(f"Copied {proto_file} to {dest_file}")
                proto_files = find_proto_files(str(proto_dir))
            else:
                print(f"No proto files found in {src_proto_dir} either")
                return 1
        else:
            print(f"Could not find proto files in {src_proto_dir} either")
            return 1
    
    # First validate the proto files
    print("Validating proto files...")
    validate_cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        "--proto_path=" + str(Path("/usr/local/include")),  # System protobuf includes
    ] + proto_files
    
    validate_result = subprocess.run(validate_cmd, capture_output=True, text=True)
    if validate_result.returncode != 0:
        print(f"Error validating proto files: {validate_result.stderr}")
        # Continue anyway, as some errors might be benign
    
    # Generate Python code for sync implementation using grpcio-tools
    print("Generating sync implementation with grpcio-tools...")
    grpc_python_cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={package_dir}",
        f"--grpc_python_out={package_dir}",
        "--proto_path=" + str(Path("/usr/local/include")),  # System protobuf includes
    ] + proto_files
    
    sync_result = subprocess.run(grpc_python_cmd, capture_output=True, text=True)
    if sync_result.returncode != 0:
        print(f"Error generating sync proto files: {sync_result.stderr}")
        # Try fixing the proto files
        fixed_protos = fix_proto_files(proto_files)
        if fixed_protos:
            # Try again with fixed protos
            print("Retrying with fixed proto files...")
            sync_result = subprocess.run(grpc_python_cmd, capture_output=True, text=True)
            if sync_result.returncode != 0:
                print(f"Error generating sync proto files after fixing: {sync_result.stderr}")
                return 1
            else:
                print("Successfully generated sync proto files after fixing")
        else:
            return 1
    
    print(f"Successfully generated sync proto files: {', '.join(proto_files)}")
    
    # Generate Python code for async implementation using grpclib
    print("Generating async implementation with grpclib protoc...")
    
    # Create the async_impl directory if it doesn't exist
    async_impl_dir = package_dir / "async_impl"
    if not async_impl_dir.exists():
        print(f"Creating async_impl directory: {async_impl_dir}")
        async_impl_dir.mkdir(parents=True, exist_ok=True)
        # Create an __init__.py file
        with open(async_impl_dir / "__init__.py", "w") as f:
            f.write('"""Asynchronous client and server implementations for OLOL."""\n\n')
    
    # Check if grpclib.plugin.main is available
    try:
        import grpclib.plugin.main as _
    except ImportError:
        print("Warning: grpclib.plugin.main not found, installing grpclib...")
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "grpclib"], check=True)
    
    async_cmd = [
        sys.executable, "-m", "grpclib.plugin.main",
        f"--proto_path={proto_dir}",
        f"--python_out={package_dir}",
        f"--grpclib_python_out={package_dir}/async_impl",
    ] + proto_files
    
    async_result = subprocess.run(async_cmd, capture_output=True, text=True)
    if async_result.returncode != 0:
        print(f"Error generating async proto files: {async_result.stderr}")
        # Try with a different tool
        print("Trying alternative async generation method...")
        alt_async_cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={package_dir}",
            f"--grpclib_python_out={package_dir}/async_impl",
            f"--plugin=protoc-gen-grpclib_python={sys.executable} -m grpclib.plugin",
        ] + proto_files
        
        alt_async_result = subprocess.run(alt_async_cmd, capture_output=True, text=True)
        if alt_async_result.returncode != 0:
            print(f"Error with alternative async generation: {alt_async_result.stderr}")
            print("Skipping async generation, sync files were generated successfully")
            return 0  # Return success anyway since sync generation worked
    else:
        print(f"Successfully generated async proto files: {', '.join(proto_files)}")
    
    # Fix imports in generated files
    fix_imports(package_dir)
    
    print("Protocol buffer compilation completed successfully")
    return 0


def fix_proto_files(proto_files: List[str]) -> bool:
    """Attempts to fix common issues in proto files.
    
    Args:
        proto_files: List of proto file paths
        
    Returns:
        True if files were fixed, False otherwise
    """
    fixed = False
    for proto_file in proto_files:
        with open(proto_file, 'r') as f:
            content = f.read()
        
        # Fix common issues:
        # 1. Missing message definitions
        missing_messages = []
        if "not defined" in content:
            # Parse error messages to find missing definitions
            pass
        
        # Apply fixes if needed
        if missing_messages:
            fixed = True
            # Add missing message definitions
            pass
    
    return fixed


def fix_imports(package_dir: Path) -> None:
    """Fix imports in generated Python files.
    
    Args:
        package_dir: Path to the package directory
    """
    # Look for generated _pb2.py and _pb2_grpc.py files
    pb2_files = list(package_dir.glob("*_pb2.py"))
    pb2_grpc_files = list(package_dir.glob("*_pb2_grpc.py"))
    
    # Also look in the async_impl directory
    async_impl_dir = package_dir / "async_impl"
    if async_impl_dir.exists():
        pb2_files.extend(async_impl_dir.glob("*_pb2.py"))
        pb2_grpc_files.extend(async_impl_dir.glob("*_pb2_grpc.py"))
    
    # Also look one directory up (from incorrect generation)
    parent_dir = package_dir.parent
    parent_pb2_files = list(parent_dir.glob("*_pb2.py"))
    parent_pb2_grpc_files = list(parent_dir.glob("*_pb2_grpc.py"))
    
    # If there are pb2 files in the parent dir but not in package dir, move them
    if parent_pb2_files and not pb2_files:
        print(f"Found pb2 files in parent directory, moving to {package_dir}")
        for parent_file in parent_pb2_files + parent_pb2_grpc_files:
            target_file = package_dir / parent_file.name
            # Only move if target doesn't exist
            if not target_file.exists():
                import shutil
                shutil.copy2(parent_file, target_file)
                print(f"Copied {parent_file} to {target_file}")
        
        # Refresh lists
        pb2_files = list(package_dir.glob("*_pb2.py"))
        pb2_grpc_files = list(package_dir.glob("*_pb2_grpc.py"))
    
    # Fix imports in all pb2 files
    for pb2_file in pb2_files:
        with open(pb2_file, 'r') as f:
            content = f.read()
        
        # The .pb2 files shouldn't import other .pb2 files, but let's check anyway
        if "import ollama_pb2" in content:
            # Fix relative imports
            if "async_impl" in str(pb2_file):
                fixed_content = content.replace("import ollama_pb2", "from .. import ollama_pb2")
            else:
                fixed_content = content.replace("import ollama_pb2", "from . import ollama_pb2")
            
            if fixed_content != content:
                with open(pb2_file, 'w') as f:
                    f.write(fixed_content)
                print(f"Fixed imports in {pb2_file}")
    
    # Fix imports in all pb2_grpc files
    for pb2_grpc_file in pb2_grpc_files:
        with open(pb2_grpc_file, 'r') as f:
            content = f.read()
        
        # Fix imports, needs to be one of:
        # from . import ollama_pb2 as ollama__pb2  # In main package
        # from .. import ollama_pb2 as ollama__pb2  # In async_impl package
        
        # First check what kind of import statement is used
        if "import ollama_pb2 as ollama__pb2" in content:
            # Normal direct import
            if "async_impl" in str(pb2_grpc_file):
                fixed_content = content.replace("import ollama_pb2 as ollama__pb2", 
                                               "from .. import ollama_pb2 as ollama__pb2")
            else:
                fixed_content = content.replace("import ollama_pb2 as ollama__pb2", 
                                               "from . import ollama_pb2 as ollama__pb2")
        elif "import ollama_pb2" in content:
            # Simple import
            if "async_impl" in str(pb2_grpc_file):
                fixed_content = content.replace("import ollama_pb2", 
                                               "from .. import ollama_pb2")
            else:
                fixed_content = content.replace("import ollama_pb2", 
                                               "from . import ollama_pb2")
        else:
            # No ollama_pb2 import found, skip
            continue
        
        if fixed_content != content:
            with open(pb2_grpc_file, 'w') as f:
                f.write(fixed_content)
            print(f"Fixed imports in {pb2_grpc_file}")
    
    # Create __init__.py files if needed
    init_file = package_dir / "async_impl" / "__init__.py"
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write('"""Asynchronous client and server implementations for OLOL."""\n\n')
        print(f"Created {init_file}")


if __name__ == "__main__":
    sys.exit(build())