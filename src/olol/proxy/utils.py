"""Utility functions for Ollama proxy server."""

import logging
from typing import Dict, Any

from ..sync.client import OllamaClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_grpc_client(server_address: str) -> OllamaClient:
    """Create a new gRPC client for a given server.
    
    Args:
        server_address: Server address in "host:port" or "[IPv6]:port" format
        
    Returns:
        OllamaClient instance
    """
    try:
        # Cas simple - format host:port
        if server_address.count(':') == 1:
            host, port_str = server_address.split(':')
            port = int(port_str)
            logger.debug(f"Connexion vers {host}:{port} (format IPv4/hostname)")
            return OllamaClient(host=host, port=port)
        
        # Cas IPv6 - format [IPv6]:port
        elif ']' in server_address:
            host = server_address[1:server_address.index(']')]
            port_str = server_address.split(']:', 1)[1]
            port = int(port_str)
            logger.debug(f"Connexion vers {host}:{port} (format IPv6 avec crochets)")
            return OllamaClient(host=host, port=port)
        
        # Cas IPv6 sans crochets
        else:
            # Trouver le dernier ':' qui sépare l'adresse du port
            last_colon = server_address.rindex(':')
            host = server_address[:last_colon]
            port_str = server_address[last_colon+1:]
            port = int(port_str)
            logger.debug(f"Connexion vers {host}:{port} (format IPv6 sans crochets)")
            return OllamaClient(host=host, port=port)
            
    except (ValueError, IndexError) as e:
        error_msg = f"Format d'adresse de serveur invalide: {server_address} - {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def adjust_context_length(model_name: str, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze input and adjust context length for optimal performance.
    
    Args:
        model_name: Name of the model being used
        prompt: Input prompt or combined message content
        options: Original request options
        
    Returns:
        Adjusted options with optimized context length
    """
    # Import here to avoid circular imports
    from proxy import cluster
    
    # Make a copy of options to avoid modifying the original
    adjusted = options.copy() if options else {}
    
    # Check if user explicitly set context_length or num_ctx
    user_specified = (
        'context_length' in adjusted or 
        'num_ctx' in adjusted or 
        'num_ctx_tokens' in adjusted
    )
    
    if user_specified:
        # User specified a value - respect it and don't change
        return adjusted
    
    # Estimate input token count (rough approximation - 4 chars per token)
    # This is just an approximation, real tokenization varies by model
    estimated_tokens = len(prompt) // 4
    
    # Get the optimal context length for this input
    if cluster and cluster.model_manager:
        recommended_ctx = cluster.model_manager.estimate_optimal_context_length(
            model_name, estimated_tokens
        )
        
        # Set context length in all the formats Ollama might use
        adjusted['context_length'] = recommended_ctx
        adjusted['num_ctx'] = recommended_ctx
        
        logger.debug(f"Adjusted context length for {model_name}: {recommended_ctx} (est. tokens: {estimated_tokens})")
    
    return adjusted


def format_server_address(ip: str, port: int) -> str:
    """Format server address according to IPv4/IPv6 conventions.
    
    Args:
        ip: IP address (IPv4 or IPv6)
        port: Port number
        
    Returns:
        Properly formatted server address
    """
    # IPv6 address needs square brackets
    if ':' in ip and not ip.startswith('localhost'):
        return f"[{ip}]:{port}"
    else:
        return f"{ip}:{port}"


def parse_server_address(server_address: str) -> tuple:
    """Parse server address into host and port.
    
    Args:
        server_address: Server address in "host:port" or "[IPv6]:port" format
        
    Returns:
        Tuple of (host, port)
    """
    # IPv6 with brackets: [IPv6]:port
    if server_address.startswith('[') and ']' in server_address:
        host = server_address[1:server_address.index(']')]
        port_str = server_address.split(']:', 1)[1]
        port = int(port_str)
        return host, port
    
    # IPv4 or hostname: host:port
    elif server_address.count(':') == 1:
        host, port_str = server_address.split(':')
        port = int(port_str)
        return host, port
    
    # IPv6 without brackets (not recommended)
    else:
        last_colon = server_address.rindex(':')
        host = server_address[:last_colon]
        port_str = server_address[last_colon+1:]
        port = int(port_str)
        return host, port