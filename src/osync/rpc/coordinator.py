"""Coordinator for distributed LLM inference."""

import logging
import threading
from typing import Any, Dict, List, Optional

import requests

from ..utils.cluster import TensorPartitioner
from .client import RPCClient
from .server import check_ollama_running, ensure_ollama_running

logger = logging.getLogger(__name__)


class InferenceCoordinator:
    """Coordinates distributed inference across multiple servers.
    
    This class manages the distribution of model layers across servers
    and coordinates the execution of inference requests.
    """
    
    def __init__(self, server_addresses: List[str]) -> None:
        """Initialize the inference coordinator.
        
        Args:
            server_addresses: List of server addresses in "host:port" format
        """
        self.client = RPCClient(server_addresses)
        self.partitioner = TensorPartitioner()
        self.model_partitions: Dict[str, Dict[str, List[int]]] = {}
        self._actual_model: Optional[str] = None  # Used when substituting models
        self._server_capabilities: Dict[str, Dict[str, Any]] = {}  # Server capabilities cache
        self._servers_lock = threading.Lock()  # Lock for thread-safe operations
        
        # Initialize with server capabilities
        self._init_partitioner()
        
    def _init_partitioner(self) -> None:
        """Initialize the partitioner with server capabilities."""
        # In a real implementation, this would query each server's capabilities
        # For now, just use default capabilities
        for server in self.client.server_addresses:
            capabilities = {
                "backend_type": "cpu" if "cpu" in server else "cuda",
                "memory": 8 * 1024 * 1024 * 1024,  # 8 GB default
                "cores": 8
            }
            self.partitioner.register_device(server, capabilities)
            
    def generate(self, 
                model: str, 
                prompt: str, 
                options: Optional[Dict[str, Any]] = None) -> str:
        """Generate text using distributed inference.
        
        This method distributes the inference across multiple servers and
        combines the results.
        
        Args:
            model: Model name to use
            prompt: Text prompt to send
            options: Optional parameters
            
        Returns:
            Generated text
        """
        try:
            # Reset any previous model substitution
            self._actual_model = None
            
            # Ensure the model is loaded on all servers
            # This may set self._actual_model if the requested model isn't available
            self._ensure_model_loaded(model)
            
            # Use the actual model (either the requested one or a substitute)
            actual_model = self._actual_model or model
            
            # Get or create partitioning plan for this model
            if actual_model not in self.model_partitions:
                model_info = self._get_model_info(actual_model)
                self.model_partitions[actual_model] = self.partitioner.partition_model(
                    model_size=model_info.get("size", 0),
                    layer_count=model_info.get("layers", 0)
                )
                logger.info(f"Created partitioning plan for {actual_model}: {self.model_partitions[actual_model]}")
                
            # Log the distribution plan
            server_count = len(self.client.server_addresses)
            logger.info(f"Running distributed inference across {server_count} servers")
                
            # Use RPCClient to execute distributed inference
            results = self.client.distributed_generate(
                model=actual_model,
                prompt=prompt,
                options=options
            )
            
            # Extract the response text from the results
            response_text = ""
            for response in results:
                if "response" in response:
                    response_text += response["response"]
                    
            # Log some stats about the distribution
            if len(results) > 0 and "server_count" in results[0]:
                server_count = results[0]["server_count"]
                logger.info(f"Inference completed using {server_count} servers")
                
            # If we're using a substitute model, inform the user
            if self._actual_model:
                # Check if it's just a different quantization of the same model
                base_requested = model.split(':')[0]
                base_actual = self._actual_model.split(':')[0]
                
                if base_requested == base_actual:
                    # Same model, different quantization
                    response_prefix = f"[Using {self._actual_model} (alternative quantization)] "
                else:
                    # Completely different model
                    response_prefix = f"[Using {self._actual_model} instead of {model}] "
                    
                response_text = response_prefix + response_text
                
            return response_text
            
        except Exception as e:
            logger.error(f"Error in distributed generation: {e}")
            return f"Error: {str(e)}"
            
    def _ensure_model_loaded(self, model: str) -> None:
        """Ensure the model is loaded on all servers.
        
        If the model isn't available on a server, it will pull it.
        If the model isn't available anywhere, it will try to use
        a common model that is available on all servers.
        
        Args:
            model: Model name to load
        """
        # First, get all available models on all servers
        server_models = {}
        common_models = set()
        first_server = True
        
        for address in self.client.server_addresses:
            try:
                # Extract host and port
                host, port = address.split(":")
                
                # Ensure Ollama is running on each server
                ollama_host = f"http://{host}:11434"
                
                # Try to start Ollama if it's not running and it's on localhost
                if not check_ollama_running(ollama_host):
                    logger.info(f"Ollama not running on {host}, attempting to start it")
                    if host in ('localhost', '127.0.0.1', '0.0.0.0', '::1'):
                        ensure_ollama_running(ollama_host)
                    else:
                        logger.warning(f"Ollama not running on remote host {host}, and cannot be started remotely")
                
                # Check available models
                response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    server_models[address] = models
                    
                    # Initialize or update the set of common models
                    if first_server:
                        common_models = set(models)
                        first_server = False
                    else:
                        common_models &= set(models)  # Intersection
                else:
                    logger.warning(f"Could not check models on {address}: HTTP {response.status_code}")
                    server_models[address] = []
            except Exception as e:
                logger.warning(f"Error checking models on {address}: {e}")
                server_models[address] = []
                
        # Check if requested model exists and is loaded on all servers
        if all(model in models for models in server_models.values()):
            logger.info(f"Model {model} is available on all servers")
            return
        
        # Check if we need to handle substitution
        use_model = model
        
        # If model doesn't exist on any server, try to find a common substitute
        if not any(model in models for models in server_models.values()):
            # Check if there's a model with a different quantization level of the same model family
            base_model_name = model.split(':')[0] if ':' in model else model
            model_with_different_quant = None
            
            for addr, models_list in server_models.items():
                # Find models that match the base name but have different quantization
                same_family_models = [m for m in models_list if base_model_name in m and m != model]
                if same_family_models:
                    model_with_different_quant = same_family_models[0]
                    logger.info(f"Found model {model_with_different_quant} from same family as {model}")
                    break
            
            # If found, use the alternative quantization
            if model_with_different_quant:
                use_model = model_with_different_quant
                logger.info(f"Using {use_model} (different quantization) instead of {model}")
            
            # If no alternative quantization, try to reload with appropriate quantization
            elif self._can_reload_with_quant(model):
                quant_model = self._reload_with_appropriate_quant(model)
                if quant_model:
                    use_model = quant_model
                    logger.info(f"Reloaded model as {use_model} with appropriate quantization")
                    # Update model info in all servers
                    for address in self.client.server_addresses:
                        try:
                            server_models[address].append(use_model)
                        except KeyError:
                            pass
            
            # Otherwise, choose a substitute from common models if available
            elif common_models:
                # Prefer smaller models in this priority order
                preferred_models = [
                    "phi", "gemma", "mistral", "llama2:7b", "llama2", 
                    "llama3", "mixtral", "mpt"
                ]
                
                for preferred in preferred_models:
                    matches = [m for m in common_models if preferred in m.lower()]
                    if matches:
                        use_model = matches[0]
                        logger.info(f"Model {model} not found, using {use_model} instead")
                        break
                        
                # If no preferred model found, use any common model
                if use_model == model and common_models:
                    use_model = list(common_models)[0]
                    logger.info(f"Model {model} not found, using {use_model} instead")
            
            # If still no common model, use whatever is available
            if use_model == model:
                # Find any model on any server and use that
                for address, models in server_models.items():
                    if models:
                        use_model = models[0]
                        logger.info(f"No common models found, using {use_model} from {address}")
                        break
        
        # Pull the model to servers that need it
        for address, models in server_models.items():
            # Skip if the server already has the model
            if use_model in models:
                continue
                
            try:
                # Extract host and port
                host, port = address.split(":")
                ollama_host = f"http://{host}:11434"
                
                # Pull the model
                logger.info(f"Pulling model {use_model} to server {address}")
                pull_response = requests.post(
                    f"{ollama_host}/api/pull",
                    json={"name": use_model},
                    timeout=30
                )
                
                if pull_response.status_code == 200:
                    logger.info(f"Successfully started pulling {use_model} to {address}")
                else:
                    logger.warning(f"Failed to pull {use_model} to {address}: HTTP {pull_response.status_code}")
            except Exception as e:
                logger.warning(f"Error pulling model to {address}: {e}")
                
        # Update the model name if we're using a different one
        if use_model != model:
            # Store the substitute model name for actual use
            self._actual_model = use_model
    
    def _can_reload_with_quant(self, model: str) -> bool:
        """Check if a model can be reloaded with a different quantization level.
        
        Args:
            model: Model name
            
        Returns:
            True if model can be reloaded with different quantization
        """
        # Extract base model name and requested quantization (if any)
        parts = model.split(':')
        base_name = parts[0]
        
        # If model already specifies quantization, we can try other levels
        if len(parts) > 1 and any(q in parts[1] for q in ['q4', 'q5', 'q8', 'f16']):
            return True
            
        # For known model families, we can try different quantizations
        known_model_families = [
            'llama2', 'llama3', 'mistral', 'mixtral', 'phi', 
            'gemma', 'mpt', 'falcon', 'orca'
        ]
        
        return any(family in base_name.lower() for family in known_model_families)
        
    def _reload_with_appropriate_quant(self, model: str) -> Optional[str]:
        """Reload a model with appropriate quantization based on environment.
        
        This tries to reload the model with the quantization level that was
        requested, or with an appropriate fallback.
        
        Args:
            model: Model name with optional quantization
            
        Returns:
            New model name with quantization, or None if reload failed
        """
        try:
            # Parse model name and requested quantization
            parts = model.split(':')
            base_name = parts[0]
            
            # Get quantization from model name or default
            requested_quant = None
            model_size = None
            
            if len(parts) > 1:
                # Check for quantization in the model name
                for quant in ['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0', 'f16']:
                    if quant in parts[1]:
                        requested_quant = quant
                        break
                        
                # Check for model size
                if any(size in parts[1] for size in ['7b', '13b', '70b', '34b']):
                    model_size = next(size for size in ['7b', '13b', '70b', '34b'] if size in parts[1])
            
            # Quantization compatibility: smaller quantizations can use larger caches
            # e.g. a model loaded with q8_0 can fulfill requests for q4_0
            quant_compatibility = {
                'q4_0': ['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'],  # q4_0 can use any higher quant
                'q4_1': ['q4_1', 'q5_0', 'q5_1', 'q8_0'],          # q4_1 can use q4_1+ caches
                'q5_0': ['q5_0', 'q5_1', 'q8_0'],                  # q5_0 can use q5_0+ caches
                'q5_1': ['q5_1', 'q8_0'],                          # q5_1 can use q5_1+ caches
                'q8_0': ['q8_0'],                                  # q8_0 only works with q8_0
                'f16': ['f16']                                     # f16 only works with f16
            }
            
            # Check if a compatible quantization is already available
            for address in self.client.server_addresses:
                try:
                    # Query Ollama directly to check loaded models
                    host, port = address.split(':')
                    ollama_host = f"http://{host}:11434"
                    
                    response = requests.get(f"{ollama_host}/api/tags")
                    if response.status_code != 200:
                        continue
                        
                    data = response.json()
                    for loaded_model in data.get("models", []):
                        loaded_name = loaded_model.get("name", "")
                        
                        # Check if this is the same base model
                        if base_name in loaded_name and loaded_name != model:
                            # Check if it has compatible quantization
                            loaded_quant = None
                            for quant in ['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0', 'f16']:
                                if quant in loaded_name:
                                    loaded_quant = quant
                                    break
                                    
                            # Check compatibility even if no specific quantization was requested
                            if loaded_quant:
                                # If no specific quantization requested, any loaded quantization is compatible
                                if not requested_quant:
                                    logger.info(f"Found compatible model: {loaded_name} for requested {model}")
                                    return loaded_name
                                    
                                # Otherwise check if loaded quantization is compatible with requested
                                elif loaded_quant in quant_compatibility.get(requested_quant, []):
                                    logger.info(f"Found compatible quantization: {loaded_name} for requested {model}")
                                    return loaded_name
                except Exception as e:
                    logger.warning(f"Error checking loaded models on {address}: {e}")
            
            # Try to reload with appropriate quantization
            # First, determine best quantization level for the device
            best_quant = self._determine_best_quant(model_size)
            
            # If requested quantization is already the best, use it
            if requested_quant and requested_quant == best_quant:
                new_model = f"{base_name}:{best_quant}"
                if model_size:
                    new_model = f"{base_name}:{model_size}:{best_quant}"
            else:
                # Otherwise use the best determined quantization
                new_model = f"{base_name}:{best_quant}"
                if model_size:
                    new_model = f"{base_name}:{model_size}:{best_quant}"
            
            # Attempt to load the model with the new quantization
            loaded = False
            for address in self.client.server_addresses:
                try:
                    host, port = address.split(':')
                    ollama_host = f"http://{host}:11434"
                    
                    # Check if the model is already available
                    response = requests.get(f"{ollama_host}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        if new_model in models:
                            logger.info(f"Model {new_model} already loaded on {address}")
                            loaded = True
                            break
                    
                    # Try to load the model with the new quantization
                    logger.info(f"Attempting to load {new_model} with appropriate quantization on {address}")
                    
                    # Use Ollama pull API to load the model
                    pull_data = {"name": new_model}
                    pull_response = requests.post(f"{ollama_host}/api/pull", json=pull_data, timeout=30)
                    
                    if pull_response.status_code == 200:
                        logger.info(f"Successfully started pulling {new_model} on {address}")
                        loaded = True
                        break
                except Exception as e:
                    logger.warning(f"Failed to load model with appropriate quantization on {address}: {e}")
            
            if loaded:
                return new_model
            
            return None
            
        except Exception as e:
            logger.error(f"Error reloading model with appropriate quantization: {e}")
            return None
            
    def _determine_best_quant(self, model_size: Optional[str] = None) -> str:
        """Determine the best quantization level based on hardware and model size.
        
        Args:
            model_size: Size of the model if known (e.g., '7b', '13b')
            
        Returns:
            Best quantization level (e.g., 'q8_0', 'q4_0')
        """
        # Check hardware capabilities
        gpu_available = False
        gpu_memory = 0
        
        for address, capabilities in self.partitioner.devices.items():
            # Skip CPU devices
            if capabilities.get("backend_type", "cpu") == "cpu":
                continue
                
            # Found a GPU device
            gpu_available = True
            gpu_memory = max(gpu_memory, capabilities.get("memory", 0))
        
        # If no GPU or very limited memory, use aggressive quantization
        if not gpu_available or gpu_memory < 4 * 1024 * 1024 * 1024:  # < 4GB
            return "q4_0"  # Most aggressive quantization
            
        # For large models on modest GPUs, use q5_0 as a balance
        if model_size in ["13b", "70b", "34b"] and gpu_memory < 12 * 1024 * 1024 * 1024:  # < 12GB
            return "q5_0"  # Good balance for large models
            
        # For large models on good GPUs, use q5_1 or q8_0
        if model_size in ["13b", "70b", "34b"] and gpu_memory < 24 * 1024 * 1024 * 1024:  # < 24GB
            return "q5_1"  # Better quality while still saving memory
            
        # For smaller models or very good GPUs, use q8_0
        return "q8_0"  # Best quality that still saves some memory
        
    def _get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model information for partitioning.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with model information
        """
        # Check for model size in the name
        model_lower = model.lower()
        
        if "70b" in model_lower:
            return {
                "name": model,
                "size": 70 * 1024 * 1024 * 1024,  # 70 GB
                "layers": 80,
                "embedding_dim": 8192
            }
        elif "34b" in model_lower or "35b" in model_lower:
            return {
                "name": model,
                "size": 34 * 1024 * 1024 * 1024,  # 34 GB
                "layers": 60,
                "embedding_dim": 6656
            }
        elif "13b" in model_lower or "14b" in model_lower:
            return {
                "name": model,
                "size": 13 * 1024 * 1024 * 1024,  # 13 GB
                "layers": 40,
                "embedding_dim": 5120
            }
        elif "7b" in model_lower or "8b" in model_lower:
            return {
                "name": model,
                "size": 7 * 1024 * 1024 * 1024,  # 7 GB
                "layers": 32,
                "embedding_dim": 4096
            }
        # Support for smaller models
        elif "3b" in model_lower:
            return {
                "name": model,
                "size": 3 * 1024 * 1024 * 1024,  # 3 GB
                "layers": 26,
                "embedding_dim": 3072
            }
        elif "2b" in model_lower:
            return {
                "name": model,
                "size": 2 * 1024 * 1024 * 1024,  # 2 GB
                "layers": 24,
                "embedding_dim": 2560
            }
        elif "1b" in model_lower:
            return {
                "name": model,
                "size": 1 * 1024 * 1024 * 1024,  # 1 GB
                "layers": 16,
                "embedding_dim": 2048
            }
        else:
            # Default for unknown models - estimate based on model family
            model_family = model.split(':')[0].lower()
            
            # Larger default for known large models
            if any(m in model_family for m in ['llama3', 'llama-3', 'mixtral', 'falcon']):
                return {
                    "name": model,
                    "size": 8 * 1024 * 1024 * 1024,  # 8 GB estimate
                    "layers": 32,
                    "embedding_dim": 4096
                }
            # Medium default for known medium models
            elif any(m in model_family for m in ['llama2', 'llama-2', 'mistral']):
                return {
                    "name": model,
                    "size": 6 * 1024 * 1024 * 1024,  # 6 GB estimate
                    "layers": 28,
                    "embedding_dim": 4096
                }
            # Smaller default for known small models
            elif any(m in model_family for m in ['phi', 'gemma', 'orca']):
                return {
                    "name": model,
                    "size": 2 * 1024 * 1024 * 1024,  # 2 GB estimate
                    "layers": 24,
                    "embedding_dim": 2560
                }
            # Fallback default
            else:
                return {
                    "name": model,
                    "size": 1 * 1024 * 1024 * 1024,  # 1 GB estimate
                    "layers": 16,
                    "embedding_dim": 768
                }
    
    def close(self) -> None:
        """Close connections to all servers."""
        self.client.close()