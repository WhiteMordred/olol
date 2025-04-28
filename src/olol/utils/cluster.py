"""Cluster management utilities for distributed Ollama instances."""

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class TensorPartitioner:
    """Manages the distribution of tensor operations across multiple servers.
    
    This class implements a sharding approach similar to the llama.cpp RPC system,
    where tensor computations can be distributed across multiple backend servers.
    """
    
    def __init__(self) -> None:
        """Initialize the tensor partitioner."""
        self.device_capabilities: Dict[str, Dict[str, Any]] = {}
        
    def register_device(self, server_id: str, capabilities: Dict[str, Any]) -> None:
        """Register a server with its hardware capabilities.
        
        Args:
            server_id: Unique identifier for the server
            capabilities: Dict containing device info like:
                - backend_type: "cuda", "metal", "cpu", etc.
                - memory: Available memory in bytes
                - compute_capability: For CUDA devices
                - cores: Number of cores/units
        """
        self.device_capabilities[server_id] = capabilities
        logger.info(f"Registered server {server_id} with capabilities: {capabilities}")
    
    def partition_model(self, 
                        model_size: int, 
                        layer_count: int) -> Dict[str, List[int]]:
        """Create a partitioning plan for a model.
        
        Args:
            model_size: Size of the model in bytes
            layer_count: Number of layers in the model
            
        Returns:
            Dictionary mapping server IDs to the layers they should process
        """
        if not self.device_capabilities:
            raise ValueError("No devices registered for partitioning")
            
        # Simple heuristic: distribute layers proportionally to device memory
        total_memory = sum(dev["memory"] for dev in self.device_capabilities.values())
        layer_assignment: Dict[str, List[int]] = {server_id: [] for server_id in self.device_capabilities}
        
        for layer_idx in range(layer_count):
            # Find best server for this layer based on current load and capabilities
            server_loads = {
                server_id: len(layers) / (dev["memory"] / total_memory)
                for server_id, dev in self.device_capabilities.items()
                for layers in [layer_assignment[server_id]]
            }
            
            # Assign to least loaded server
            best_server = min(server_loads.items(), key=lambda x: x[1])[0]
            layer_assignment[best_server].append(layer_idx)
            
        return layer_assignment
    
    def get_device_for_tensor(self, 
                              tensor_id: str, 
                              tensor_size: int,
                              operation: str) -> str:
        """Determine the best device for a specific tensor operation.
        
        Args:
            tensor_id: Identifier for the tensor
            tensor_size: Size of tensor in bytes
            operation: Type of operation ("matmul", "attention", etc.)
            
        Returns:
            Server ID that should handle this tensor
        """
        # Simple selection based on available memory and operation type
        eligible_servers = []
        
        for server_id, capabilities in self.device_capabilities.items():
            # Only consider devices with enough memory
            if capabilities["memory"] >= tensor_size:
                # Prefer GPU for matmul operations
                if operation == "matmul" and capabilities["backend_type"] in ["cuda", "rocm", "metal"]:
                    eligible_servers.append((server_id, 2.0))  # Higher weight for GPU
                else:
                    eligible_servers.append((server_id, 1.0))
        
        if not eligible_servers:
            raise ValueError(f"No device can handle tensor of size {tensor_size} bytes")
            
        # Choose randomly weighted by score
        # Handle numpy import gracefully in case it's not available
        try:
            import numpy as np
            servers, weights = zip(*eligible_servers, strict=False)
            weights = np.array(weights) / sum(weights)
            return np.random.choice(servers, p=weights)
        except ImportError:
            # Fallback if numpy isn't available
            import random
            servers = [server for server, _ in eligible_servers]
            return random.choice(servers)


class ModelManager:
    """Manager for model availability and synchronization across servers.
    
    Tracks which models are available on which servers and facilitates
    model sharing between servers.
    """
    
    def __init__(self) -> None:
        """Initialize the model manager."""
        # Map model names to details
        self.models: Dict[str, Dict[str, Any]] = {}
        
        # Map model to server list
        self.model_server_map: Dict[str, List[str]] = {}
        
        # Map model name to supported context lengths
        self.model_context_lengths: Dict[str, Dict[str, int]] = {}
        
        # Map model name to embedding dimensions
        self.model_embedding_dims: Dict[str, int] = {}
        
        # Map model name to parameter counts
        self.model_parameters: Dict[str, int] = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def register_model(self, model_name: str, server: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Register a model as available on a server.
        
        Args:
            model_name: Name of the model
            server: Server address
            details: Optional model details like size, quantization, etc.
        """
        with self.lock:
            # Update model details
            if model_name not in self.models:
                self.models[model_name] = details or {}
            elif details:
                # Merge with existing details, preferring new ones
                self.models[model_name].update(details)
                
            # Update server mapping
            if model_name not in self.model_server_map:
                self.model_server_map[model_name] = []
                
            if server not in self.model_server_map[model_name]:
                self.model_server_map[model_name].append(server)
                logger.info(f"Model {model_name} registered on server {server}")
    
    def unregister_model(self, model_name: str, server: str) -> None:
        """Unregister a model from a server.
        
        Args:
            model_name: Name of the model
            server: Server address
        """
        with self.lock:
            if model_name in self.model_server_map and server in self.model_server_map[model_name]:
                self.model_server_map[model_name].remove(server)
                logger.info(f"Model {model_name} unregistered from server {server}")
                
                # Clean up if no servers have this model
                if not self.model_server_map[model_name]:
                    del self.model_server_map[model_name]
                    if model_name in self.models:
                        del self.models[model_name]
    
    def get_servers_for_model(self, model_name: str) -> List[str]:
        """Get all servers that have a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of server addresses
        """
        with self.lock:
            return self.model_server_map.get(model_name, []).copy()
    
    def update_server_models(self, server: str, available_models: List[str]) -> None:
        """Update the models available on a server.
        
        Args:
            server: Server address
            available_models: List of model names available on this server
        """
        with self.lock:
            # First get all models this server currently has
            current_models = [
                model for model, servers in self.model_server_map.items()
                if server in servers
            ]
            
            # Remove server from models it no longer has
            for model in current_models:
                if model not in available_models:
                    self.unregister_model(model, server)
            
            # Add server to models it now has
            for model in available_models:
                if model not in current_models:
                    self.register_model(model, server)
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """Get all models and the servers they're on.
        
        Returns:
            Dict mapping model names to lists of server addresses
        """
        with self.lock:
            return {model: servers.copy() for model, servers in self.model_server_map.items()}
    
    def get_model_details(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get details about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with model details or None if not found
        """
        with self.lock:
            return self.models.get(model_name)
            
    def set_model_context_length(self, model_name: str, context_length: int, max_length: Optional[int] = None) -> None:
        """Set the context length for a model.
        
        Args:
            model_name: Name of the model
            context_length: Current context window size
            max_length: Maximum supported context window size (if known)
        """
        with self.lock:
            if model_name not in self.model_context_lengths:
                self.model_context_lengths[model_name] = {}
                
            self.model_context_lengths[model_name]["current"] = context_length
            
            if max_length is not None:
                self.model_context_lengths[model_name]["max"] = max_length
                
            # Update model details as well
            if model_name in self.models:
                self.models[model_name]["context_length"] = context_length
                if max_length is not None:
                    self.models[model_name]["max_context_length"] = max_length
            
            logger.info(f"Model {model_name} context length set to {context_length}" +
                        (f" (max: {max_length})" if max_length is not None else ""))
            
    def get_model_context_length(self, model_name: str) -> Dict[str, int]:
        """Get the context length for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with current and max context lengths if known
        """
        with self.lock:
            return self.model_context_lengths.get(model_name, {}).copy()
            
    def detect_context_length_from_modelfile(self, model_name: str, modelfile_content: str) -> Optional[int]:
        """Try to detect context length from a modelfile.
        
        Looks for patterns like "context_length: 8192" or "parameter context_length 8192"
        
        Args:
            model_name: Name of the model
            modelfile_content: Content of the Modelfile
            
        Returns:
            Detected context length or None if not found
        """
        # Different patterns to check
        patterns = [
            r'context_length:?\s*(\d+)',          # YAML style
            r'parameter\s+context_length\s+(\d+)', # Parameter style
            r'n_ctx:?\s*(\d+)',                   # n_ctx parameter
            r'Context\s+Length:?\s*(\d+)',        # Human readable
            r'context\s+window:?\s*(\d+)',        # Another variant
            r'max_seq_len:?\s*(\d+)'              # Max sequence length
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, modelfile_content, re.IGNORECASE)
            if matches:
                try:
                    context_length = int(matches[0])
                    self.set_model_context_length(model_name, context_length, context_length)
                    return context_length
                except (ValueError, IndexError):
                    continue
        
        # Also look for embedding dimension patterns
        embedding_patterns = [
            r'embedding_length:?\s*(\d+)',       # YAML style
            r'parameter\s+embedding_length\s+(\d+)', # Parameter style 
            r'embedding_dim(?:ension)?:?\s*(\d+)',  # Common notation
            r'dim(?:ension)?:?\s*(\d+)',         # Short form
            r'n_embed(?:ding)?:?\s*(\d+)'        # Model parameter style
        ]
        
        # Try each embedding pattern
        for pattern in embedding_patterns:
            matches = re.findall(pattern, modelfile_content, re.IGNORECASE)
            if matches:
                try:
                    embedding_dim = int(matches[0])
                    self.set_embedding_dimension(model_name, embedding_dim)
                except (ValueError, IndexError):
                    continue
                
        return None
        
    def set_embedding_dimension(self, model_name: str, embedding_dim: int) -> None:
        """Set the embedding dimension for a model.
        
        Args:
            model_name: Name of the model
            embedding_dim: Embedding vector dimension
        """
        with self.lock:
            self.model_embedding_dims[model_name] = embedding_dim
            
            # Update model details as well
            if model_name in self.models:
                self.models[model_name]["embedding_dimension"] = embedding_dim
                
            logger.info(f"Model {model_name} embedding dimension set to {embedding_dim}")
            
    def get_embedding_dimension(self, model_name: str) -> Optional[int]:
        """Get the embedding dimension for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Embedding dimension or None if not known
        """
        with self.lock:
            return self.model_embedding_dims.get(model_name)
        
    def set_model_parameter_count(self, model_name: str, parameter_count: int) -> None:
        """Set the parameter count for a model.
        
        Args:
            model_name: Name of the model
            parameter_count: Number of parameters in billions or as integer
        """
        with self.lock:
            # Handle various formats like "7B" or 7000000000
            if isinstance(parameter_count, str):
                if parameter_count.endswith('B'):
                    try:
                        # Convert "7B" to 7 billion
                        self.model_parameters[model_name] = float(parameter_count[:-1]) * 1_000_000_000
                    except ValueError:
                        # On conversion error, store as-is
                        self.model_parameters[model_name] = parameter_count
                else:
                    try:
                        # Try to convert to int
                        self.model_parameters[model_name] = int(parameter_count)
                    except ValueError:
                        # On conversion error, store as-is
                        self.model_parameters[model_name] = parameter_count
            else:
                # Store as-is if already numeric
                self.model_parameters[model_name] = parameter_count
                
            # Update model details as well
            if model_name in self.models:
                self.models[model_name]["parameters"] = parameter_count
                
    def estimate_optimal_context_length(self, model_name: str, input_tokens: int) -> int:
        """Estimate the optimal context length for a model based on input size.
        
        Args:
            model_name: Name of the model
            input_tokens: Number of tokens in the input (estimated or actual)
            
        Returns:
            Recommended context length
        """
        with self.lock:
            # Get known context lengths if available
            context_info = self.model_context_lengths.get(model_name, {})
            current = context_info.get("current", 4096)  # Default to 4096 if unknown
            max_length = context_info.get("max", 32768)  # Default to 32K if unknown max
            
            # Ensure max length is reasonable
            if max_length > 1_000_000:  # Sanity check
                max_length = 32768
            
            # Calculate the recommended length - at least double the input plus 1000 for output
            recommended = max(current, min(input_tokens * 2 + 1000, max_length))
            
            # Round to nearest multiple of 512
            recommended = ((recommended + 511) // 512) * 512
            
            # For very large inputs, we might want to be more generous
            if input_tokens > 4000:
                # For large inputs, give even more space (4x)
                large_recommendation = min(input_tokens * 4, max_length)
                large_recommendation = ((large_recommendation + 511) // 512) * 512
                recommended = max(recommended, large_recommendation)
            
            return recommended


class OllamaCluster:
    """Manager for a cluster of Ollama instances.
    
    Provides load balancing, server health tracking, and model availability
    across a cluster of Ollama servers.
    """
    
    def __init__(self, server_addresses: List[str]) -> None:
        """Initialize the cluster manager.
        
        Args:
            server_addresses: List of server addresses in "host:port" format
        """
        self.server_addresses = server_addresses
        self.server_loads = {server: 0 for server in server_addresses}
        self.server_lock = threading.Lock()
        
        # Enhanced model management
        self.model_manager = ModelManager()
        
        # For backward compatibility, keep this reference
        self.model_server_map: Dict[str, List[str]] = self.model_manager.model_server_map
        self.model_lock = self.model_manager.lock
        
        # Maps session IDs to their assigned server
        self.session_server_map: Dict[str, str] = {}
        self.session_lock = threading.Lock()
        
        # Server health status
        self.server_health: Dict[str, bool] = {server: True for server in server_addresses}
        self.health_lock = threading.Lock()
        
        # Extended server information for complex networks
        # Maps server_id -> connection details
        self.server_connections: Dict[str, Dict[str, Any]] = {}
        self.connections_lock = threading.Lock()
        
        # Server capabilities
        self.server_capabilities: Dict[str, Dict[str, Any]] = {}
        self.capabilities_lock = threading.Lock()
    
    def select_server(self, model_name: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Select the best server for a request.
        
        Args:
            model_name: Optional model name to filter servers
            session_id: Optional session ID to maintain server affinity
            
        Returns:
            Selected server address
        """
        # If session already exists, use the same server
        if session_id:
            with self.session_lock:
                if session_id in self.session_server_map:
                    server = self.session_server_map[session_id]
                    logger.debug(f"Using existing server {server} for session {session_id}")
                    return server
        
        # If model specified, find servers with this model
        if model_name:
            with self.model_lock:
                if model_name in self.model_server_map:
                    available_servers = [
                        s for s in self.model_server_map[model_name] 
                        if self.server_health.get(s, False)
                    ]
                    if available_servers:
                        # Get the least loaded server among those with the model
                        with self.server_lock:
                            server_options = [(s, self.server_loads[s]) for s in available_servers]
                            selected_server = min(server_options, key=lambda x: x[1])[0]
                            self.server_loads[selected_server] += 1
                            
                            # Record session mapping if provided
                            if session_id:
                                with self.session_lock:
                                    self.session_server_map[session_id] = selected_server
                                    
                            logger.debug(f"Selected server {selected_server} for model {model_name}")
                            return selected_server
        
        # Otherwise, use least loaded healthy server
        with self.server_lock, self.health_lock:
            healthy_servers = [s for s in self.server_addresses if self.server_health.get(s, False)]
            if not healthy_servers:
                # If no healthy servers, try any server
                logger.warning("No healthy servers available, trying any server")
                healthy_servers = self.server_addresses
                
            server_options = [(s, self.server_loads[s]) for s in healthy_servers]
            selected_server = min(server_options, key=lambda x: x[1])[0]
            self.server_loads[selected_server] += 1
            
            # Record selected server for new session
            if session_id:
                with self.session_lock:
                    self.session_server_map[session_id] = selected_server
            
            # Record model availability if specified
            if model_name:
                with self.model_lock:
                    if model_name not in self.model_server_map:
                        self.model_server_map[model_name] = []
                    if selected_server not in self.model_server_map[model_name]:
                        self.model_server_map[model_name].append(selected_server)
                        
            logger.debug(f"Selected least loaded server {selected_server}")
            return selected_server
    
    def release_server(self, server: str) -> None:
        """Release load counter for a server after request completes.
        
        Args:
            server: Server address to release
        """
        with self.server_lock:
            if server in self.server_loads:
                self.server_loads[server] = max(0, self.server_loads[server] - 1)
    
    def mark_server_health(self, server: str, healthy: bool, force: bool = False) -> None:
        """Update health status for a server.
        
        Args:
            server: Server address to update
            healthy: Whether the server is healthy
            force: Whether to force the health state even if it would downgrade a connection
        """
        with self.health_lock:
            # Critical: Never mark a previously healthy server as unhealthy
            # once it has successfully connected, unless force=True
            if not healthy and server in self.server_health and self.server_health[server] and not force:
                logger.warning(f"Ignoring temporary health check failure for previously healthy server: {server}")
                return
                
            # Only log changes in health status
            if server not in self.server_health or self.server_health[server] != healthy:
                self.server_health[server] = healthy
                logger.info(f"Server {server} health status: {'healthy' if healthy else 'unhealthy'}")
            else:
                # Still update, but don't log
                self.server_health[server] = healthy
    
    def update_model_availability(self, server: str, models: List[str], 
                              model_details: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Update which models are available on which servers.
        
        Args:
            server: Server address
            models: List of model names available on this server
            model_details: Optional dictionary mapping model names to their details
        """
        # Use the new model manager for the update
        self.model_manager.update_server_models(server, models)
        
        # If we have details about specific models, register them
        if model_details:
            for model_name, details in model_details.items():
                if model_name in models:  # Only register if the model is actually available
                    self.model_manager.register_model(model_name, server, details)
        
        # Update server capabilities with model info
        with self.capabilities_lock:
            if server in self.server_capabilities:
                self.server_capabilities[server]["models"] = models
                
        # Log the updated model availability
        logger.debug(f"Server {server} now has {len(models)} models: {', '.join(models[:5])}" + 
                    (f" and {len(models)-5} more" if len(models) > 5 else ""))
    
    def remove_session(self, session_id: str) -> None:
        """Remove a session from the tracking map.
        
        Args:
            session_id: Session ID to remove
        """
        with self.session_lock:
            if session_id in self.session_server_map:
                del self.session_server_map[session_id]
                
    def get_cluster_status(self) -> Dict:
        """Get a summary of the cluster status.
        
        Returns:
            Dict with cluster status information
        """
        # Gather locks in a specific order to avoid deadlocks
        with self.health_lock:
            with self.server_lock:
                with self.model_lock:
                    with self.session_lock:
                        with self.capabilities_lock:
                            # Basic server information
                            server_info = {
                                server: {
                                    "load": self.server_loads[server],
                                    "healthy": self.server_health.get(server, False),
                                } for server in self.server_addresses
                            }
                            
                            # Add capability information where available
                            for server, info in server_info.items():
                                if server in self.server_capabilities:
                                    # Include selected capability highlights
                                    caps = self.server_capabilities[server]
                                    info["device_type"] = caps.get("device_type", "unknown")
                                    
                                    # Include device info if available
                                    if "device_info" in caps:
                                        info["device_info"] = caps["device_info"]
                                        
                                    # Include model count
                                    if "models" in caps:
                                        info["model_count"] = len(caps["models"])
                                        
                            # Get enhanced model information
                            model_info = {}
                            for model_name, servers in self.model_server_map.items():
                                # Get model details if available
                                details = self.model_manager.get_model_details(model_name) or {}
                                
                                # Create entry with both servers and available details
                                model_info[model_name] = {
                                    "servers": servers,
                                    "details": details
                                }
                            
                            # Build the complete status response
                            return {
                                "servers": server_info,
                                "models": model_info,
                                "model_count": len(model_info),
                                "server_count": len(server_info),
                                "healthy_server_count": sum(1 for s in server_info.values() if s.get("healthy", False)),
                                "sessions": len(self.session_server_map)
                            }
            
    def add_server(self, server_address: str, connection_details: Optional[Dict[str, Any]] = None) -> None:
        """Add a new server to the cluster.
        
        Args:
            server_address: Server address in "host:port" format
            connection_details: Optional additional connection information for complex networks
        """
        with self.server_lock, self.health_lock:
            if server_address not in self.server_addresses:
                logger.info(f"Adding new server to cluster: {server_address}")
                self.server_addresses.append(server_address)
                self.server_loads[server_address] = 0
                self.server_health[server_address] = True
                
                # Register connection details if provided
                if connection_details:
                    with self.connections_lock:
                        self.server_connections[server_address] = connection_details
                        
                        # Log all endpoints for debugging
                        if "connection_endpoints" in connection_details:
                            endpoints = connection_details["connection_endpoints"]
                            logger.debug(f"Server {server_address} has {len(endpoints)} connection endpoints")
                            
    def register_connection_details(self, server_address: str, connection_details: Dict[str, Any]) -> None:
        """Register detailed connection information for a server.
        
        Args:
            server_address: Server address in "host:port" format
            connection_details: Dict containing connection information like:
                - connection_endpoints: List of possible endpoints
                - reachable_ips: List of reachable IP addresses
                - best_ip: Best IP address to use
                - source_port: Source port for NAT traversal
        """
        with self.connections_lock:
            self.server_connections[server_address] = connection_details
            logger.debug(f"Registered connection details for {server_address}")
            
        # Extract capabilities if present
        capabilities = connection_details.get("capabilities", {})
        if capabilities:
            with self.capabilities_lock:
                self.server_capabilities[server_address] = capabilities
                
    def register_server_capabilities(self, server_address: str, capabilities: Dict[str, Any]) -> None:
        """Register capabilities of a server.
        
        Args:
            server_address: Server address in "host:port" format
            capabilities: Dict with server capabilities (CPU, GPU, memory, etc.)
        """
        with self.capabilities_lock:
            self.server_capabilities[server_address] = capabilities
            logger.info(f"Registered capabilities for {server_address}: {capabilities.get('device_type', 'unknown')}")
            
    def request_model_transfer(self, model_name: str, source_server: str, target_server: str) -> bool:
        """Request a model transfer from source to target server.
        
        Args:
            model_name: Name of the model to transfer
            source_server: Source server that has the model
            target_server: Target server that needs the model
            
        Returns:
            True if transfer was initiated, False otherwise
        """
        # Check if source server has the model
        available_servers = self.model_manager.get_servers_for_model(model_name)
        if source_server not in available_servers:
            logger.warning(f"Source server {source_server} does not have model {model_name}")
            return False
            
        # Check if target server already has the model
        if target_server in available_servers:
            logger.info(f"Target server {target_server} already has model {model_name}")
            return True
            
        # Get connection details for the target server
        target_endpoint = self.get_best_connection_endpoint(target_server)
        
        # Mark the transfer as requested in capabilities
        with self.capabilities_lock:
            if target_server in self.server_capabilities:
                if "pending_transfers" not in self.server_capabilities[target_server]:
                    self.server_capabilities[target_server]["pending_transfers"] = {}
                    
                # Record source and timestamp
                self.server_capabilities[target_server]["pending_transfers"][model_name] = {
                    "source": source_server,
                    "requested_at": time.time()
                }
        
        logger.info(f"Requested transfer of model {model_name} from {source_server} to {target_server}")
        return True
        
    def get_server_capabilities(self, server_address: str) -> Dict[str, Any]:
        """Get capabilities of a server.
        
        Args:
            server_address: Server address in "host:port" format
            
        Returns:
            Dict with server capabilities or empty dict if not found
        """
        with self.capabilities_lock:
            return self.server_capabilities.get(server_address, {}).copy()
            
    def get_best_connection_endpoint(self, server_address: str) -> str:
        """Get the best connection endpoint for a server.
        
        This handles complex network environments where direct addressing
        might not work due to tunnels, NAT, etc.
        
        Args:
            server_address: Server address in "host:port" format
            
        Returns:
            The optimal connection endpoint or the original address if no details
        """
        with self.connections_lock:
            if server_address in self.server_connections:
                details = self.server_connections[server_address]
                
                # First try a best_ip if available
                if "best_ip" in details:
                    best_ip = details["best_ip"]
                    port = server_address.split(":")[-1]
                    
                    # Format differently for IPv6
                    if ':' in best_ip and not best_ip.startswith('localhost'):
                        return f"[{best_ip}]:{port}"
                    else:
                        return f"{best_ip}:{port}"
                
                # Next try any connection endpoints
                if "connection_endpoints" in details and details["connection_endpoints"]:
                    return details["connection_endpoints"][0]  # Return first endpoint
                
            # Fall back to original address
            return server_address