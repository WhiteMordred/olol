"""Main entry point for the OLOL package."""

import argparse
import asyncio
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OLOL - Ollama gRPC interface with sync/async support")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start gRPC server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    server_parser.add_argument("--port", type=int, default=50051, help="Server port")
    server_parser.add_argument("--async", dest="async_mode", action="store_true", 
                              help="Run in async mode (default: sync)")
    server_parser.add_argument("--ollama-host", default="http://localhost:11434",
                              help="Ollama API host URL")
    
    # Proxy command
    proxy_parser = subparsers.add_parser("proxy", help="Start load balancing proxy")
    proxy_parser.add_argument("--host", default="0.0.0.0", help="Proxy host address")
    proxy_parser.add_argument("--port", type=int, default=8000, help="Proxy port")
    proxy_parser.add_argument("--servers", default="localhost:50051",
                             help="Comma-separated list of gRPC servers")
    proxy_parser.add_argument("--distributed", action="store_true",
                             help="Enable distributed inference mode")
    proxy_parser.add_argument("--rpc-servers", 
                             help="Comma-separated list of RPC servers for distributed inference")
    proxy_parser.add_argument("--auto-distribute-large", action="store_true", default=True,
                             help="Automatically use distributed inference for large models")
    proxy_parser.add_argument("--no-auto-distribute", action="store_false", dest="auto_distribute_large",
                             help="Disable automatic distribution for large models")
    proxy_parser.add_argument("--discovery", action="store_true", default=True,
                             help="Enable auto-discovery of servers")
    proxy_parser.add_argument("--no-discovery", action="store_false", dest="discovery",
                             help="Disable auto-discovery")
    proxy_parser.add_argument("--interface", 
                             help="Preferred network interface IP address for connections")
    proxy_parser.add_argument("--ui", action="store_true", default=True,
                             help="Enable console UI (default: enabled)")
    proxy_parser.add_argument("--no-ui", action="store_false", dest="ui",
                             help="Disable console UI")
    proxy_parser.add_argument("--show-config", action="store_true", 
                             help="Show configuration and continue (compatible with --no-ui)")
    proxy_parser.add_argument("--verbose", action="store_true",
                             help="Enable verbose logging and detailed UI status updates")
    proxy_parser.add_argument("--debug", action="store_true",
                             help="Enable debug mode with maximum verbosity")
    
    # Client command
    client_parser = subparsers.add_parser("client", help="Run a client example")
    client_parser.add_argument("--host", default="localhost", help="Server host address")
    client_parser.add_argument("--port", type=int, default=50051, help="Server port")
    client_parser.add_argument("--async", dest="async_mode", action="store_true",
                              help="Use async client (default: sync)")
    client_parser.add_argument("--model", default="llama2", help="Model name")
    client_parser.add_argument("--prompt", default="Hello, how are you?", help="Prompt text")
    
    # RPC server command
    rpc_server_parser = subparsers.add_parser("rpc-server", help="Start distributed RPC inference server")
    rpc_server_parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    rpc_server_parser.add_argument("--port", type=int, default=50052, help="Server port")
    rpc_server_parser.add_argument("--device", default="auto", 
                                  choices=["auto", "cpu", "cuda", "rocm", "metal"],
                                  help="Device type (auto, cpu, cuda, rocm, metal)")
    rpc_server_parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    rpc_server_parser.add_argument("--ollama-host", default="http://localhost:11434",
                                  help="Ollama API host URL")
    rpc_server_parser.add_argument("--start-ollama", action="store_true",
                                  help="Start Ollama if not running")
    # Ollama performance options
    rpc_server_parser.add_argument("--flash-attention", action="store_true", default=True,
                                  help="Enable FlashAttention for faster inference")
    rpc_server_parser.add_argument("--no-flash-attention", action="store_false", dest="flash_attention",
                                  help="Disable FlashAttention")
    rpc_server_parser.add_argument("--context-window", type=int, default=8192,
                                  help="Default context window size")
    rpc_server_parser.add_argument("--quantize", default="q8_0", 
                                  choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
                                  help="Quantization level for models")
    rpc_server_parser.add_argument("--health-check-interval", type=int, default=30,
                                  help="Interval for Ollama health checks (seconds)")
    rpc_server_parser.add_argument("--debug", action="store_true",
                                  help="Enable debug mode with additional logging")
    rpc_server_parser.add_argument("--interface", 
                                  help="Preferred network interface IP address for connections")
    
    # Distributed inference command
    dist_parser = subparsers.add_parser("dist", help="Run distributed inference")
    dist_parser.add_argument("--servers", default="localhost:50052",
                           help="Comma-separated list of RPC server addresses")
    dist_parser.add_argument("--model", default="llama2", help="Model name")
    dist_parser.add_argument("--prompt", default="Hello, how are you?", help="Prompt text")
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    return parser.parse_args()


async def async_main(args) -> None:
    """Run the async version of the application."""
    if args.command == "server":
        # Import like this to avoid 'async' keyword conflict
        async_server = __import__("olol.async_impl.server", fromlist=["serve"]).serve
        await async_server()
        
    elif args.command == "client":
        # Import like this to avoid 'async' keyword conflict
        AsyncOllamaClient = __import__("olol.async_impl.client", fromlist=["AsyncOllamaClient"]).AsyncOllamaClient
        
        client = AsyncOllamaClient(host=args.host, port=args.port)
        try:
            print(f"Running model {args.model} with prompt: {args.prompt}")
            async for response in client.generate(args.model, args.prompt):
                if not response.done:
                    print(response.response, end="", flush=True)
                else:
                    print(f"\nCompleted in {response.total_duration}ms")
        finally:
            await client.close()


def sync_main(args) -> None:
    """Run the sync version of the application."""
    if args.command == "server":
        from .service import serve
        serve()
        
    elif args.command == "client":
        from .sync.client import OllamaClient
        
        client = OllamaClient(host=args.host, port=args.port)
        try:
            print(f"Running model {args.model} with prompt: {args.prompt}")
            for response in client.generate(args.model, args.prompt):
                if not response.done:
                    print(response.response, end="", flush=True)
                else:
                    print(f"\nCompleted in {response.total_duration}ms")
        finally:
            client.close()
    
    elif args.command == "proxy":
        from .proxy import run_proxy
        
        # Parse server addresses
        server_addresses = args.servers.split(",")
        
        # Parse RPC server addresses if provided
        rpc_servers = None
        if hasattr(args, "rpc_servers") and args.rpc_servers:
            rpc_servers = args.rpc_servers.split(",")
        
        # UI option is passed to run_proxy below
        
        # Set log level based on verbosity
        log_level = logging.INFO
        if hasattr(args, "debug") and args.debug:
            log_level = logging.DEBUG
        elif hasattr(args, "verbose") and args.verbose:
            log_level = logging.INFO
        
        logging.getLogger().setLevel(log_level)
        
        # Show configuration if requested
        if hasattr(args, "show_config") and args.show_config:
            print("\n===== OLOL Proxy Configuration =====")
            print(f"Host: {args.host}")
            print(f"Port: {args.port}")
            print(f"Servers: {server_addresses}")
            print(f"Distributed mode: {'Enabled' if getattr(args, 'distributed', False) else 'Disabled'}")
            if rpc_servers:
                print(f"RPC servers: {rpc_servers}")
            print(f"Auto-distribute large models: {'Enabled' if getattr(args, 'auto_distribute_large', True) else 'Disabled'}")
            print(f"Discovery: {'Enabled' if getattr(args, 'discovery', True) else 'Disabled'}")
            if hasattr(args, "interface") and args.interface:
                print(f"Preferred interface: {args.interface}")
            print(f"Console UI: {'Enabled' if getattr(args, 'ui', True) else 'Disabled'}")
            print(f"Verbose mode: {'Enabled' if getattr(args, 'verbose', False) else 'Disabled'}")
            print(f"Debug mode: {'Enabled' if getattr(args, 'debug', False) else 'Disabled'}")
            print("===================================\n")
        
        # Run the proxy with distributed inference if enabled
        run_proxy(
            host=args.host, 
            port=args.port, 
            server_addresses=server_addresses,
            enable_distributed=args.distributed if hasattr(args, "distributed") else False,
            auto_distribute_large=args.auto_distribute_large if hasattr(args, "auto_distribute_large") else True,
            rpc_servers=rpc_servers,
            enable_discovery=args.discovery if hasattr(args, "discovery") else True,
            preferred_interface=args.interface if hasattr(args, "interface") else None,
            enable_ui=args.ui if hasattr(args, "ui") else True,
            verbose=args.verbose if hasattr(args, "verbose") else False,
            debug=args.debug if hasattr(args, "debug") else False
        )


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.command == "version":
        from . import __version__
        print(f"OLOL version {__version__}")
        return
    
    if args.command is None:
        print("Error: Command is required. Use --help for available commands.")
        sys.exit(1)
    
    if args.command == "rpc-server":
        run_rpc_server(args)
    elif args.command == "dist":
        run_distributed_inference(args)
    elif getattr(args, "async_mode", False):
        asyncio.run(async_main(args))
    else:
        sync_main(args)

# Direct entry points for uv script commands

def run_proxy_entrypoint() -> None:
    """Direct entry point for olol-proxy command."""
    import sys

    from .proxy import run_proxy
    
    # Default values
    host = "0.0.0.0"
    port = 8000
    server_addresses = ["localhost:50051"]
    enable_distributed = False
    auto_distribute_large = True
    rpc_servers = None
    enable_discovery = True
    interface = None
    enable_ui = True
    show_config = False
    verbose = False
    debug = False
    
    # Parse command line arguments
    args = sys.argv[1:]
    i = 0
    
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--servers" and i + 1 < len(args):
            server_addresses = args[i + 1].split(",")
            i += 2
        elif args[i] == "--distributed":
            enable_distributed = True
            i += 1
        elif args[i] == "--no-auto-distribute":
            auto_distribute_large = False
            i += 1
        elif args[i] == "--auto-distribute-large":
            auto_distribute_large = True
            i += 1
        elif args[i] == "--rpc-servers" and i + 1 < len(args):
            rpc_servers = args[i + 1].split(",")
            i += 2
        elif args[i] == "--discovery":
            enable_discovery = True
            i += 1
        elif args[i] == "--no-discovery":
            enable_discovery = False
            i += 1
        elif args[i] == "--interface" and i + 1 < len(args):
            interface = args[i + 1]
            i += 2
        elif args[i] == "--ui":
            enable_ui = True
            i += 1
        elif args[i] == "--no-ui":
            # Explicitly set enable_ui to False
            enable_ui = False
            i += 1
        elif args[i] == "--show-config":
            show_config = True
            i += 1
        elif args[i] == "--verbose":
            verbose = True
            i += 1
        elif args[i] == "--debug":
            debug = True
            verbose = True  # Debug implies verbose
            i += 1
        elif args[i] == "--help" or args[i] == "-h":
            print("Usage: olol-proxy [OPTIONS]")
            print("\nOptions:")
            print("  --host HOST                  Proxy host address (default: 0.0.0.0)")
            print("  --port PORT                  Proxy port (default: 8000)")
            print("  --servers SERVERS            Comma-separated list of gRPC servers (default: localhost:50051)")
            print("  --distributed                Enable distributed inference mode")
            print("  --rpc-servers SERVERS        Comma-separated list of RPC servers for distributed inference")
            print("  --auto-distribute-large      Automatically use distributed inference for large models (default)")
            print("  --no-auto-distribute         Disable automatic distribution for large models")
            print("  --discovery                  Enable auto-discovery of servers (default)")
            print("  --no-discovery               Disable auto-discovery")
            print("  --interface IP               Preferred network interface IP address for connections")
            print("  --ui                         Enable console UI (default)")
            print("  --no-ui                      Disable console UI")
            print("  --show-config                Show configuration and continue")
            print("  --verbose                    Enable verbose logging and detailed UI status updates")
            print("  --debug                      Enable debug mode with maximum verbosity")
            print("  --help, -h                   Show this help message and exit")
            return
        else:
            print(f"Unknown argument: {args[i]}")
            i += 1
    
    # Set log level based on verbosity
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    
    logging.getLogger().setLevel(log_level)
    
    # Show configuration if requested
    if show_config:
        print("\n===== OLOL Proxy Configuration =====")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Servers: {server_addresses}")
        print(f"Distributed mode: {'Enabled' if enable_distributed else 'Disabled'}")
        if rpc_servers:
            print(f"RPC servers: {rpc_servers}")
        print(f"Auto-distribute large models: {'Enabled' if auto_distribute_large else 'Disabled'}")
        print(f"Discovery: {'Enabled' if enable_discovery else 'Disabled'}")
        if interface:
            print(f"Preferred interface: {interface}")
        print(f"Console UI: {'Enabled' if enable_ui else 'Disabled'}")
        print(f"Verbose mode: {'Enabled' if verbose else 'Disabled'}")
        print(f"Debug mode: {'Enabled' if debug else 'Disabled'}")
        print("===================================\n")
    
    # Run the proxy
    run_proxy(
        host=host,
        port=port,
        server_addresses=server_addresses,
        enable_distributed=enable_distributed,
        auto_distribute_large=auto_distribute_large,
        rpc_servers=rpc_servers,
        enable_discovery=enable_discovery,
        preferred_interface=interface,
        enable_ui=enable_ui,
        verbose=verbose,
        debug=debug
    )

def run_server_entrypoint() -> None:
    """Direct entry point for olol-server command."""
    import sys

    from .server import serve
    
    # Default values
    host = "0.0.0.0"
    port = 50051
    ollama_host = "http://localhost:11434"
    async_mode = False
    
    # Parse command line arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--ollama-host" and i + 1 < len(args):
            ollama_host = args[i + 1]
            i += 2
        elif args[i] == "--async":
            async_mode = True
            i += 1
        elif args[i] == "--help" or args[i] == "-h":
            print("Usage: olol-server [OPTIONS]")
            print("\nOptions:")
            print("  --host HOST                  Server host address (default: 0.0.0.0)")
            print("  --port PORT                  Server port (default: 50051)")
            print("  --ollama-host HOST           Ollama API host URL (default: http://localhost:11434)")
            print("  --async                      Run in async mode (default: sync)")
            print("  --help, -h                   Show this help message and exit")
            return
        else:
            print(f"Unknown argument: {args[i]}")
            i += 1
    
    # Run the server
    if async_mode:
        import asyncio

        from .async_impl.server import serve as async_serve
        asyncio.run(async_serve(host, port, ollama_host))
    else:
        from .service import serve
        serve(host, port, ollama_host)

def run_rpc_server_entrypoint() -> None:
    """Direct entry point for olol-rpc command."""
    import sys

    from .rpc.server import serve
    
    # Default values
    host = "0.0.0.0"
    port = 50052
    device_type = "auto"
    device_id = 0
    ollama_host = "http://localhost:11434"
    flash_attention = True
    context_window = 8192
    quantize = "q8_0"
    debug = False
    discovery = True
    health_check_interval = 30
    interface = None
    
    # Parse command line arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--device" and i + 1 < len(args):
            device_type = args[i + 1]
            i += 2
        elif args[i] == "--device-id" and i + 1 < len(args):
            device_id = int(args[i + 1])
            i += 2
        elif args[i] == "--ollama-host" and i + 1 < len(args):
            ollama_host = args[i + 1]
            i += 2
        elif args[i] == "--flash-attention":
            flash_attention = True
            i += 1
        elif args[i] == "--no-flash-attention":
            flash_attention = False
            i += 1
        elif args[i] == "--context-window" and i + 1 < len(args):
            context_window = int(args[i + 1])
            i += 2
        elif args[i] == "--quantize" and i + 1 < len(args):
            quantize = args[i + 1]
            i += 2
        elif args[i] == "--debug":
            debug = True
            i += 1
        elif args[i] == "--discovery":
            discovery = True
            i += 1
        elif args[i] == "--no-discovery":
            discovery = False
            i += 1
        elif args[i] == "--health-check-interval" and i + 1 < len(args):
            health_check_interval = int(args[i + 1])
            i += 2
        elif args[i] == "--interface" and i + 1 < len(args):
            interface = args[i + 1]
            i += 2
        elif args[i] == "--help" or args[i] == "-h":
            print("Usage: olol-rpc [OPTIONS]")
            print("\nOptions:")
            print("  --host HOST                  Server host address (default: 0.0.0.0)")
            print("  --port PORT                  Server port (default: 50052)")
            print("  --device TYPE                Device type: auto, cpu, cuda, rocm, metal (default: auto)")
            print("  --device-id ID               Device ID (default: 0)")
            print("  --ollama-host HOST           Ollama API host URL (default: http://localhost:11434)")
            print("  --flash-attention            Enable FlashAttention for faster inference (default)")
            print("  --no-flash-attention         Disable FlashAttention")
            print("  --context-window SIZE        Default context window size (default: 8192)")
            print("  --quantize LEVEL             Quantization level: q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32 (default: q8_0)")
            print("  --debug                      Enable debug mode with additional logging")
            print("  --discovery                  Enable auto-discovery to find proxy servers (default)")
            print("  --no-discovery               Disable auto-discovery")
            print("  --health-check-interval SEC  Interval for Ollama health checks in seconds (default: 30)")
            print("  --interface IP               Preferred network interface IP address for connections")
            print("  --help, -h                   Show this help message and exit")
            return
        else:
            print(f"Unknown argument: {args[i]}")
            i += 1
    
    # Prepare environment variables
    ollama_env = {
        "OLLAMA_FLASH_ATTENTION": "1" if flash_attention else "0",
        "OLLAMA_CONTEXT_WINDOW": str(context_window),
        "OLLAMA_QUANTIZE": quantize
    }
    
    if debug:
        ollama_env["OLLAMA_DEBUG"] = "1"
        ollama_env["OLLAMA_LOG_LEVEL"] = "debug"
        # Also increase Python logging level
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle auto device detection
    if device_type == "auto":
        device_type = _auto_detect_device_type()
    
    # Run the RPC server
    serve(
        host=host,
        port=port,
        device_type=device_type,
        device_id=device_id,
        ollama_host=ollama_host,
        ollama_env=ollama_env,
        health_check_interval=health_check_interval,
        enable_discovery=discovery,
        preferred_interface=interface
    )

def run_dist_entrypoint() -> None:
    """Direct entry point for olol-dist command."""
    import sys
    
    # Default values
    servers = "localhost:50052"
    model = "llama2"
    prompt = "Hello, how are you?"
    
    # Parse command line arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--servers" and i + 1 < len(args):
            servers = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--prompt" and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
        elif args[i] == "--help" or args[i] == "-h":
            print("Usage: olol-dist [OPTIONS]")
            print("\nOptions:")
            print("  --servers SERVERS            Comma-separated list of RPC server addresses (default: localhost:50052)")
            print("  --model MODEL                Model name (default: llama2)")
            print("  --prompt PROMPT              Prompt text (default: 'Hello, how are you?')")
            print("  --help, -h                   Show this help message and exit")
            return
        else:
            print(f"Unknown argument: {args[i]}")
            i += 1
    
    # Run distributed inference
    try:
        from .rpc.coordinator import InferenceCoordinator
        
        # Parse server addresses
        server_addresses = servers.split(",")
        
        # Create coordinator
        coordinator = InferenceCoordinator(server_addresses)
        
        try:
            # Run inference
            print(f"Running distributed inference with model {model}")
            print(f"Prompt: {prompt}")
            print("Generating response...")
            
            response = coordinator.generate(
                model=model,
                prompt=prompt
            )
            
            print("\nResponse:")
            print(response)
            
        finally:
            # Clean up
            coordinator.close()
            
    except ImportError as e:
        print(f"Error loading RPC coordinator module: {e}")
        print("Make sure numpy and requests are installed: pip install numpy requests")
        sys.exit(1)
    except Exception as e:
        print(f"Error during distributed inference: {e}")
        sys.exit(1)

def run_client_entrypoint() -> None:
    """Direct entry point for olol-client command."""
    import sys
    
    # Default values
    host = "localhost"
    port = 50051
    model = "llama2"
    prompt = "Hello, how are you?"
    async_mode = False
    
    # Parse command line arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--prompt" and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
        elif args[i] == "--async":
            async_mode = True
            i += 1
        elif args[i] == "--help" or args[i] == "-h":
            print("Usage: olol-client [OPTIONS]")
            print("\nOptions:")
            print("  --host HOST                  Server host address (default: localhost)")
            print("  --port PORT                  Server port (default: 50051)")
            print("  --model MODEL                Model name (default: llama2)")
            print("  --prompt PROMPT              Prompt text (default: 'Hello, how are you?')")
            print("  --async                      Use async client (default: sync)")
            print("  --help, -h                   Show this help message and exit")
            return
        else:
            print(f"Unknown argument: {args[i]}")
            i += 1
    
    # Run client
    if async_mode:
        import asyncio
        async def async_client():
            from .async_impl.client import AsyncOllamaClient
            
            client = AsyncOllamaClient(host=host, port=port)
            try:
                print(f"Running model {model} with prompt: {prompt}")
                async for response in client.generate(model, prompt):
                    if not response.done:
                        print(response.response, end="", flush=True)
                    else:
                        print(f"\nCompleted in {response.total_duration}ms")
            finally:
                await client.close()
                
        asyncio.run(async_client())
    else:
        from .sync.client import OllamaClient
        
        client = OllamaClient(host=host, port=port)
        try:
            print(f"Running model {model} with prompt: {prompt}")
            for response in client.generate(model, prompt):
                if not response.done:
                    print(response.response, end="", flush=True)
                else:
                    print(f"\nCompleted in {response.total_duration}ms")
        finally:
            client.close()
        
def run_rpc_server(args) -> None:
    """Run RPC server for distributed inference."""
    try:
        from .rpc.server import serve
        
        # If device is set to auto, try to auto-detect the best device
        device_type = args.device
        if device_type == "auto":
            device_type = _auto_detect_device_type()
        
        # Prepare Ollama environment variables
        ollama_env = {}
        
        # Set performance options
        if hasattr(args, "flash_attention"):
            ollama_env["OLLAMA_FLASH_ATTENTION"] = "1" if args.flash_attention else "0"
        
        if hasattr(args, "context_window"):
            ollama_env["OLLAMA_CONTEXT_WINDOW"] = str(args.context_window)
            
        if hasattr(args, "quantize"):
            ollama_env["OLLAMA_QUANTIZE"] = args.quantize
            
        if hasattr(args, "debug") and args.debug:
            ollama_env["OLLAMA_DEBUG"] = "1"
            ollama_env["OLLAMA_LOG_LEVEL"] = "debug"
            # Also increase Python logging level
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Health check interval
        health_check_interval = getattr(args, "health_check_interval", 30)
        
        serve(
            host=args.host,
            port=args.port,
            device_type=device_type,
            device_id=args.device_id,
            ollama_host=args.ollama_host,
            ollama_env=ollama_env,
            health_check_interval=health_check_interval
        )
    except ImportError as e:
        print(f"Error loading RPC server module: {e}")
        print("Make sure numpy and requests are installed: pip install numpy requests")
        sys.exit(1)
        
def _auto_detect_device_type() -> str:
    """Auto-detect the best available device type.
    
    Returns:
        String with device type: "cuda", "rocm", "metal", or "cpu"
    """
    # Try CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            print(f"Auto-detected CUDA device with {torch.cuda.device_count()} devices")
            return "cuda"
    except ImportError:
        pass
    
    # Try ROCm
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            print(f"Auto-detected ROCm device with {torch.hip.device_count()} devices")
            return "rocm"
    except ImportError:
        pass
        
    # Try direct ROCm detection
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showdevice"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            print("Auto-detected ROCm device using rocm-smi")
            return "rocm"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
        
    # Try Metal (macOS only)
    if sys.platform == "darwin":
        try:
            # Check if PyTorch with Metal is available
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Auto-detected Metal device")
                return "metal"
        except ImportError:
            pass
            
    # Default to CPU
    print("No GPU detected, using CPU")
    return "cpu"
        
def run_distributed_inference(args) -> None:
    """Run distributed inference across multiple servers."""
    try:
        from .rpc.coordinator import InferenceCoordinator
        
        # Parse server addresses
        server_addresses = args.servers.split(",")
        
        # Create coordinator
        coordinator = InferenceCoordinator(server_addresses)
        
        try:
            # Run inference
            print(f"Running distributed inference with model {args.model}")
            print(f"Prompt: {args.prompt}")
            print("Generating response...")
            
            response = coordinator.generate(
                model=args.model,
                prompt=args.prompt
            )
            
            print("\nResponse:")
            print(response)
            
        finally:
            # Clean up
            coordinator.close()
            
    except ImportError as e:
        print(f"Error loading RPC coordinator module: {e}")
        print("Make sure numpy is installed: pip install numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error during distributed inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()