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
    
    def update_model_info(self, model_name: str, details: Dict[str, Any]) -> None:
        """Update the information about a model with new details.
        
        Args:
            model_name: Name of the model
            details: Dictionary containing model details to update
        """
        with self.lock:
            if model_name not in self.models:
                self.models[model_name] = {}
                
            # Update the model details
            self.models[model_name].update(details)
            
            # Extract and update special fields if present
            if "context_length" in details:
                self.set_model_context_length(model_name, details["context_length"],
                                         details.get("max_context_length"))
                
            if "embedding_dimension" in details:
                self.set_embedding_dimension(model_name, details["embedding_dimension"])
                
            if "parameters" in details:
                self.set_model_parameter_count(model_name, details["parameters"])
                
            logger.debug(f"Updated model info for {model_name}: {details}")
    
    def get_all_models_with_details(self) -> Dict[str, Dict[str, Any]]:
        """Get all models with their details.
        
        Returns:
            Dict mapping model names to their details
        """
        with self.lock:
            models_with_details = {}
            
            for model_name in self.model_server_map:
                # Start with basic info - servers list
                servers = self.model_server_map[model_name].copy()
                model_info = {"servers": servers}
                
                # Add details from models dictionary
                if model_name in self.models:
                    model_info.update(self.models[model_name])
                    
                # Add context length information if available
                if model_name in self.model_context_lengths:
                    model_info["context"] = self.model_context_lengths[model_name].copy()
                    
                # Add embedding dimension if available
                if model_name in self.model_embedding_dims:
                    model_info["embedding_dim"] = self.model_embedding_dims[model_name]
                    
                # Add parameter count if available
                if model_name in self.model_parameters:
                    model_info["parameter_count"] = self.model_parameters[model_name]
                    
                models_with_details[model_name] = model_info
                
            return models_with_details


class OllamaCluster:
    """Manage a cluster of Ollama servers for load balancing and failover."""
    
    def __init__(self, server_addresses: List[str]):
        """Initialize the cluster with a list of server addresses."""
        # Fix hint errors for missing imports
        from typing import Dict, Any, Set, List

        # Server list and load information
        self.server_addresses = server_addresses
        self.server_loads: Dict[str, int] = {}
        self.server_status: Dict[str, bool] = {}
        self.server_health: Dict[str, bool] = {}
        self.server_connection_details: Dict[str, Dict[str, Any]] = {}
        
        # Mapping models to servers
        self.model_server_map: Dict[str, List[str]] = {}
        
        # Discovery callbacks
        self.discovery_callbacks = []
        
        # Thread locks to avoid racing conditions
        self.server_lock = threading.RLock()
        self.health_lock = threading.RLock()
        self.model_lock = threading.RLock()

        # Initialize server loads
        with self.server_lock:
            for server in server_addresses:
                self.server_loads[server] = 0
                
        # Initialize server health status
        with self.health_lock:
            for server in server_addresses:
                self.server_health[server] = False
                
        # Initialize model-server mappings
        # This sera rempli plus tard lorsque les serveurs seront découverts
        
        # Start model discovery for each server
        threading.Thread(target=self._discover_all_servers, daemon=True).start()
        
    def _discover_all_servers(self):
        """Discover models on all servers."""
        with self.server_lock:
            servers = list(self.server_addresses)
        
        # Ajouter un message de log pour le débogage
        logger.info(f"Découverte des modèles sur {len(servers)} serveurs")
        
        for server in servers:
            # Exécuter immédiatement pour le premier serveur, puis un par un
            self._discover_server_models(server)
            # Ajouter un petit délai pour éviter de surcharger les serveurs
            time.sleep(0.5)
            
        # Ajouter un log pour indiquer la fin de la découverte
        with self.model_lock:
            model_count = len(self.model_server_map)
            logger.info(f"Découverte terminée: {model_count} modèles trouvés")
    
    def _discover_server_models(self, server_address: str):
        """Discover models available on a server."""
        try:
            # Format host:port simple
            if server_address.count(':') == 1:
                host, port_str = server_address.split(':')
                port = int(port_str)
                
                from olol.sync.client import OllamaClient
                client = OllamaClient(host=host, port=port)
                
                try:
                    # Vérifier la santé du serveur
                    is_healthy = client.check_health()
                    
                    # Mettre à jour l'état de santé
                    with self.health_lock:
                        self.server_health[server_address] = is_healthy
                    
                    # Si sain, récupérer les modèles
                    if is_healthy:
                        models_response = client.list_models()
                        
                        if hasattr(models_response, 'models'):
                            # Mise à jour de la carte des modèles
                            with self.model_lock:
                                for model_info in models_response.models:
                                    model_name = model_info.name
                                    
                                    if model_name not in self.model_server_map:
                                        self.model_server_map[model_name] = []
                                        
                                    if server_address not in self.model_server_map[model_name]:
                                        self.model_server_map[model_name].append(server_address)
                except Exception:
                    # Si une erreur se produit, marquer le serveur comme non sain
                    with self.health_lock:
                        self.server_health[server_address] = False
                finally:
                    # Fermer le client pour libérer les ressources
                    client.close()
        except Exception:
            # En cas d'erreur de connexion, marquer le serveur comme non sain
            with self.health_lock:
                self.server_health[server_address] = False