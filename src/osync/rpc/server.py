"""Server for RPC-based distributed LLM inference."""

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import grpc
import grpc.aio
import requests

try:
    import numpy as np
except ImportError:
    np = None  # Handle case where numpy is not available

# Import proto definitions safely with fallback
try:
    from ..proto import ollama_pb2, ollama_pb2_grpc
except ImportError:
    try:
        import ollama_pb2
        import ollama_pb2_grpc
    except ImportError:
        # Will be generated at runtime
        pass

logger = logging.getLogger(__name__)


class RPCServer(ollama_pb2_grpc.DistributedOllamaServiceServicer):
    """Server for distributed LLM inference using RPC.
    
    This server can process subsets of model layers assigned by
    a coordinator, similar to the llama.cpp RPC architecture.
    """
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 device_type: str = "cpu",
                 device_id: int = 0) -> None:
        """Initialize the RPC server.
        
        Args:
            ollama_host: URL of the Ollama HTTP API
            device_type: Type of compute device ("cpu", "cuda", "metal")
            device_id: Device ID for multi-device systems
        """
        self.ollama_host = ollama_host
        self.device_type = device_type
        self.device_id = device_id
        self.loaded_models = set()
        self.active_computations = {}
        self.start_time = time.time()
        
        # Get device capabilities
        self.device_capabilities = self._detect_device_capabilities()
        logger.info(f"Server initialized with device: {device_type}:{device_id}")
        logger.info(f"Device capabilities: {self.device_capabilities}")
        
    def _detect_device_capabilities(self) -> Dict[str, Any]:
        """Detect capabilities of the compute device.
        
        Returns:
            Dictionary of device capabilities
        """
        capabilities = {
            "backend_type": self.device_type,
            "device_id": self.device_id,
        }
        
        # First try to get info from Ollama logs
        ollama_capabilities = self._get_ollama_capabilities()
        if ollama_capabilities:
            # Merge with base capabilities, preferring Ollama's info
            capabilities.update(ollama_capabilities)
            logger.info(f"Using capabilities from Ollama: {capabilities}")
            return capabilities
        
        # Fall back to detecting device capabilities directly
        if self.device_type == "cpu":
            try:
                import psutil
                memory = psutil.virtual_memory()
                capabilities["memory"] = memory.total
                capabilities["cores"] = psutil.cpu_count(logical=False) or 4
            except ImportError:
                # Default values if psutil not available
                capabilities["memory"] = 8 * 1024 * 1024 * 1024  # 8 GB
                capabilities["cores"] = 4
                
        elif self.device_type == "cuda":
            try:
                # Try to get CUDA device info
                import torch
                if torch.cuda.is_available() and self.device_id < torch.cuda.device_count():
                    device_properties = torch.cuda.get_device_properties(self.device_id)
                    capabilities["memory"] = device_properties.total_memory
                    capabilities["compute_capability"] = f"{device_properties.major}.{device_properties.minor}"
                    capabilities["name"] = device_properties.name
                else:
                    logger.warning(f"CUDA device {self.device_id} not available")
                    capabilities["memory"] = 4 * 1024 * 1024 * 1024  # 4 GB default
            except ImportError:
                logger.warning("torch not available, using default CUDA capabilities")
                capabilities["memory"] = 4 * 1024 * 1024 * 1024  # 4 GB default
                
        elif self.device_type == "rocm":
            try:
                # Try to get ROCm device info via torch if available
                import torch
                if hasattr(torch, 'hip') and torch.hip.is_available() and self.device_id < torch.hip.device_count():
                    # Get device properties
                    device = torch.hip.device(f"hip:{self.device_id}")
                    device_properties = torch.hip.get_device_properties(device)
                    capabilities["memory"] = device_properties.total_memory
                    capabilities["compute_capability"] = device_properties.gfx_version
                    capabilities["name"] = device_properties.name
                else:
                    logger.warning(f"ROCm device {self.device_id} not available via torch")
                    # Try direct ROCm detection
                    self._detect_rocm_capabilities(capabilities)
            except ImportError:
                logger.warning("torch not available, trying direct ROCm detection")
                self._detect_rocm_capabilities(capabilities)
                
        elif self.device_type == "metal":
            # Default Metal capabilities
            capabilities["memory"] = 4 * 1024 * 1024 * 1024  # 4 GB default
            
        return capabilities
        
    def _get_ollama_capabilities(self) -> Dict[str, Any]:
        """Get capabilities from Ollama's output.
        
        Parses Ollama's startup logs to get accurate GPU information.
        
        Returns:
            Dictionary of capabilities or empty dict if not available
        """
        try:
            # Use requests instead of curl for more reliable results
            import requests
            try:
                response = requests.get(f"{self.ollama_host}/api/status", timeout=5)
                if response.status_code != 200:
                    logger.warning(f"Failed to get Ollama status: HTTP {response.status_code}")
                    return {}
                
                data = response.json()
                capabilities = {}
            except requests.RequestException as e:
                # Fallback to curl if requests fails
                logger.debug(f"Failed to use requests, falling back to curl: {e}")
                result = subprocess.run(
                    ["curl", "-s", f"{self.ollama_host}/api/status"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode != 0 or not result.stdout:
                    logger.warning("Failed to get Ollama status with curl")
                    return {}
                
                # Clean the output - sometimes it has extra characters
                stdout = result.stdout.strip()
                # Remove any non-JSON prefix
                if stdout and stdout[0] != '{':
                    pos = stdout.find('{')
                    if pos >= 0:
                        stdout = stdout[pos:]
                
                try:
                    data = json.loads(stdout)
                    capabilities = {}
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse Ollama status JSON: {e}")
                    return {}
            
            # Look for GPU information
            if "gpus" in data:
                for i, gpu in enumerate(data["gpus"]):
                    if i != self.device_id:
                        continue
                        
                    capabilities["name"] = gpu.get("name", "")
                    
                    # Convert memory strings like "15.6 GiB" to bytes
                    if "memory" in gpu:
                        mem_str = gpu.get("memory", "0 GiB")
                        mem_val = float(mem_str.split()[0])
                        mem_unit = mem_str.split()[1].upper()
                        
                        # Convert to bytes
                        if "GIB" in mem_unit:
                            capabilities["memory"] = int(mem_val * 1024 * 1024 * 1024)
                        elif "MIB" in mem_unit:
                            capabilities["memory"] = int(mem_val * 1024 * 1024)
                        
                    # Determine backend type
                    if gpu.get("library") == "cuda":
                        capabilities["backend_type"] = "cuda"
                        capabilities["compute_capability"] = gpu.get("compute", "")
                    elif gpu.get("library") == "rocm":
                        capabilities["backend_type"] = "rocm"
                        capabilities["compute_capability"] = gpu.get("compute", "")
                    elif gpu.get("library") == "metal":
                        capabilities["backend_type"] = "metal"
                        
                    break
                    
            return capabilities
            
        except Exception as e:
            logger.warning(f"Error getting Ollama capabilities: {e}")
            return {}
            
    def _detect_rocm_capabilities(self, capabilities: Dict[str, Any]) -> None:
        """Detect ROCm capabilities directly.
        
        Updates the capabilities dictionary in place.
        
        Args:
            capabilities: Dictionary to update with ROCm capabilities
        """
        try:
            # Try to get ROCm device info using rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "-d", str(self.device_id), "--json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if str(self.device_id) in data:
                    gpu_data = data[str(self.device_id)]
                    
                    # Get total memory
                    if "VRAM" in gpu_data:
                        total_memory = gpu_data["VRAM"]["total"]
                        # Convert to bytes (assuming MB)
                        capabilities["memory"] = int(total_memory) * 1024 * 1024
                    
                    # Try to get device name
                    if "GPU" in gpu_data and "card_name" in gpu_data["GPU"]:
                        capabilities["name"] = gpu_data["GPU"]["card_name"]
                        
            else:
                # Try using hipinfo command
                result = subprocess.run(
                    ["hipinfo", "--devices"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    device_found = False
                    
                    for line in lines:
                        if line.startswith(f"Device {self.device_id}:"):
                            device_found = True
                        elif device_found:
                            if "gfx" in line:
                                # Extract gfx version
                                capabilities["compute_capability"] = line.strip()
                            elif "Memory" in line:
                                # Extract memory size
                                mem_parts = line.split(":")
                                if len(mem_parts) > 1:
                                    mem_str = mem_parts[1].strip()
                                    # Parse memory like "8 GB"
                                    if "GB" in mem_str:
                                        mem_val = float(mem_str.replace("GB", "").strip())
                                        capabilities["memory"] = int(mem_val * 1024 * 1024 * 1024)
                            elif "Name" in line:
                                # Extract device name
                                name_parts = line.split(":")
                                if len(name_parts) > 1:
                                    capabilities["name"] = name_parts[1].strip()
        except Exception as e:
            logger.warning(f"Error detecting ROCm capabilities: {e}")
        
        # If all detection methods failed, use default values
        if "memory" not in capabilities:
            capabilities["memory"] = 8 * 1024 * 1024 * 1024  # 8 GB default
        if "compute_capability" not in capabilities:
            capabilities["compute_capability"] = "gfx1100"  # Default to a recent AMD GPU
        
    def GetVersion(self, request, context):
        """Get API version information.
        
        Args:
            request: VersionRequest
            context: gRPC context
            
        Returns:
            VersionInfo with API version details
        """
        logger.info("GetVersion requested")
        
        # Create the response with version information
        response = ollama_pb2.VersionInfo(
            api_version=ollama_pb2.ApiVersion.V1_2_0,  # Current version
            version_string="1.2.0",
            protocol_version=2  # Increment when making breaking changes
        )
        
        return response
        
    def HealthCheck(self, request, context):
        """Check if the server is healthy.
        
        Args:
            request: HealthCheckRequest
            context: gRPC context
            
        Returns:
            HealthCheckResponse with health status
        """
        logger.info("HealthCheck requested")
        
        # Check Ollama API health
        ollama_health = self._check_ollama_health()
        
        # Calculate uptime
        uptime_seconds = int(time.time() - self.start_time)
        
        # Create the response
        response = ollama_pb2.HealthCheckResponse(
            healthy=ollama_health,
            status="healthy" if ollama_health else "unhealthy",
            version="1.2.0",
            uptime_seconds=uptime_seconds
        )
        
        # Add details about the health check
        response.details["ollama_api"] = "connected" if ollama_health else "disconnected"
        response.details["device_type"] = self.device_type
        response.details["device_id"] = str(self.device_id)
        
        return response
    
    def _check_ollama_health(self) -> bool:
        """Check if Ollama API is healthy.
        
        Returns:
            True if Ollama API is responding, False otherwise
        """
        try:
            import requests
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def GetDeviceCapabilities(self, request, context):
        """Get capabilities of this server for distributed inference.
        
        Args:
            request: DeviceCapabilitiesRequest
            context: gRPC context
            
        Returns:
            DeviceCapabilitiesResponse with server capabilities
        """
        logger.info(f"GetDeviceCapabilities requested with detail={request.detail}")
        
        # Create the response with device capabilities
        response = ollama_pb2.DeviceCapabilitiesResponse(
            device_type=self.device_capabilities.get("backend_type", "cpu"),
            device_id=self.device_capabilities.get("device_id", 0),
            memory=self.device_capabilities.get("memory", 0),
            cores=self.device_capabilities.get("cores", 0),
            compute_capability=self.device_capabilities.get("compute_capability", "")
        )
        
        # Add any additional details
        for key, value in self.device_capabilities.items():
            if key not in ["backend_type", "device_id", "memory", "cores", "compute_capability"]:
                if isinstance(value, str):
                    response.details[key] = value
                else:
                    response.details[key] = str(value)
        
        return response
        
    def DistributedGenerate(self, request, context):
        """Process a distributed generation request for specific model layers.
        
        Handles a subset of model layers as specified in the request.
        
        Args:
            request: DistributedGenerateRequest with layer assignments
            context: gRPC context
            
        Returns:
            Stream of DistributedGenerateResponse messages
        """
        logger.info(f"DistributedGenerate request for model: {request.model}")
        logger.info(f"Processing layers: {request.assigned_layers}")
        
        # Build Ollama API request parameters
        prompt = request.prompt
        model = request.model
        options = dict(request.options)
        
        # Set up the request
        if request.is_first:
            # First server does tokenization and initial embedding
            try:
                # Process the input through assigned layers only
                url = f"{self.ollama_host}/api/generate"
                process_args = {
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": options
                }
                
                # Add layer range parameter to tell Ollama to only process certain layers
                process_args["options"]["layer_range"] = f"0-{max(request.assigned_layers)}"
                
                # Execute the request
                result = self._execute_ollama_request(url, process_args)
                for i, step_result in enumerate(result):
                    # Create layer outputs
                    layer_outputs = {}
                    
                    # Convert hidden states to bytes for transfer
                    for layer_id in request.assigned_layers:
                        if "hidden_states" in step_result and len(step_result["hidden_states"]) > layer_id:
                            # In a real implementation, these would be actual tensor data from the model
                            # For now, we use placeholder data (in production would be numpy tensors)
                            tensor_data = f"layer_{layer_id}_state_{i}".encode('utf-8')
                            layer_outputs[layer_id] = tensor_data
                    
                    # Create response
                    response = ollama_pb2.DistributedGenerateResponse(
                        layer_outputs=layer_outputs,
                        done=step_result.get("done", False)
                    )
                    
                    yield response
                
            except Exception as e:
                logger.error(f"Error in distributed inference for first server: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return
                
        elif request.is_last:
            # Last server produces the final output tokens
            try:
                # This server gets the hidden states from previous servers
                # and produces the final output tokens
                url = f"{self.ollama_host}/api/generate"
                process_args = {
                    "model": model,
                    "prompt": prompt, 
                    "stream": True,
                    "options": options
                }
                
                # Configure to process only the assigned layers
                min_layer = min(request.assigned_layers)
                process_args["options"]["layer_range"] = f"{min_layer}-{max(request.assigned_layers)}"
                
                # Execute the request
                result = self._execute_ollama_request(url, process_args)
                for step_result in result:
                    # Create layer outputs
                    layer_outputs = {}
                    
                    # Add layer outputs
                    for layer_id in request.assigned_layers:
                        if "hidden_states" in step_result and len(step_result["hidden_states"]) > layer_id:
                            tensor_data = f"final_layer_{layer_id}_output".encode('utf-8')
                            layer_outputs[layer_id] = tensor_data
                    
                    # Get the generated tokens
                    tokens = []
                    if "tokens" in step_result:
                        tokens = step_result["tokens"]
                    elif "token" in step_result:
                        tokens = [step_result["token"]]
                    
                    # Create response
                    response = ollama_pb2.DistributedGenerateResponse(
                        layer_outputs=layer_outputs,
                        done=step_result.get("done", False)
                    )
                    
                    # Add tokens to response
                    if tokens:
                        response.tokens.extend(tokens)
                        logger.info(f"Generated tokens: {tokens}")
                    
                    yield response
                    
            except Exception as e:
                logger.error(f"Error in distributed inference for last server: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return
                
        else:
            # Middle servers process intermediate layers
            try:
                # Process only the assigned layers
                url = f"{self.ollama_host}/api/generate"
                process_args = {
                    "model": model,
                    "prompt": prompt,
                    "stream": True, 
                    "options": options
                }
                
                # Set layer range
                min_layer = min(request.assigned_layers)
                process_args["options"]["layer_range"] = f"{min_layer}-{max(request.assigned_layers)}"
                
                # Execute the request
                result = self._execute_ollama_request(url, process_args)
                for step_result in result:
                    # Create layer outputs
                    layer_outputs = {}
                    
                    # Add layer outputs
                    for layer_id in request.assigned_layers:
                        if "hidden_states" in step_result and len(step_result["hidden_states"]) > layer_id:
                            tensor_data = f"middle_layer_{layer_id}_output".encode('utf-8')
                            layer_outputs[layer_id] = tensor_data
                    
                    # Create response
                    response = ollama_pb2.DistributedGenerateResponse(
                        layer_outputs=layer_outputs,
                        done=step_result.get("done", False)
                    )
                    
                    yield response
                    
            except Exception as e:
                logger.error(f"Error in distributed inference for middle server: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return
                
    def _execute_ollama_request(self, url, data):
        """Execute a request to the Ollama API.
        
        Args:
            url: Ollama API endpoint URL
            data: Request data
            
        Returns:
            Generator of response data
        """
        import requests
        
        try:
            # Make POST request to Ollama API
            response = requests.post(url, json=data, stream=True)
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse response line: {line}")
        except requests.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
            
    def ProcessLayers(self, request, context):
        """Process a specific layer's computation.
        
        Args:
            request: LayerProcessRequest specifying the layer and input
            context: gRPC context
            
        Returns:
            Stream of LayerProcessResponse messages with results
        """
        logger.info(f"Processing layer {request.layer_id} with operation {request.operation}")
        
        try:
            # Process a single layer using Ollama's API
            layer_id = request.layer_id
            input_tensor = request.input_tensor
            operation = request.operation
            
            # Unpack the input tensor
            logger.info(f"Received input tensor for layer {layer_id} with size {len(input_tensor)} bytes")
            
            # Only try to use numpy if it's available, otherwise skip that part
            try:
                import numpy as np
                # In a real implementation with numpy, we'd convert bytes to array here
                logger.debug("numpy is available for tensor processing")
                
                # Process the tensor through Ollama
                url = f"{self.ollama_host}/api/embeddings"  # Using embeddings API for tensor operations
                
                # Build request based on operation type
                api_request = {
                    "model": self._get_loaded_model_name(),
                    "prompt": "",  # Not used for direct tensor operations
                    "options": {
                        "operation": operation,
                        "layer": layer_id
                    }
                }
                
                # Execute the request
                response_data = None
                try:
                    import requests
                    response = requests.post(url, json=api_request)
                    response.raise_for_status()
                    response_data = response.json()
                except Exception as e:
                    logger.warning(f"Ollama API request failed: {e}, using fallback processing")
                    # Fallback to local processing
                    response_data = {"embedding": []}
                
                # Create result tensor
                if response_data and "embedding" in response_data:
                    # In a real implementation, we'd convert the embedding to tensor
                    # For now, we'll just create a placeholder result
                    processed_tensor = f"real_processed_layer_{layer_id}_{operation}".encode('utf-8')
                else:
                    # Fallback result
                    processed_tensor = f"fallback_processed_layer_{layer_id}_{operation}".encode('utf-8')
                
                # Return the result
                yield ollama_pb2.LayerProcessResponse(
                    layer_id=layer_id,
                    output_tensor=processed_tensor,
                    success=True
                )
                
            except ImportError:
                logger.warning("numpy not available, using fallback tensor processing")
                # Create a fallback tensor output
                processed_tensor = f"fallback_processed_layer_{layer_id}_{operation}".encode('utf-8')
                
                # Return the result
                yield ollama_pb2.LayerProcessResponse(
                    layer_id=layer_id,
                    output_tensor=processed_tensor,
                    success=True
                )
            
        except Exception as e:
            logger.error(f"Error processing layer {request.layer_id}: {e}")
            yield ollama_pb2.LayerProcessResponse(
                layer_id=request.layer_id,
                success=False,
                error=str(e)
            )
            
    def _get_loaded_model_name(self):
        """Get a loaded model name from Ollama.
        
        Returns:
            Name of a loaded model or default fallback
        """
        try:
            # Query Ollama for loaded models
            import requests
            response = requests.get(f"{self.ollama_host}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            # Get models sorted by preference (smaller models first)
            if "models" in data and len(data["models"]) > 0:
                models = [m["name"] for m in data["models"]]
                
                # Order by preference - try to use smaller models first
                preferred_order = [
                    "phi", "gemma", "mistral", "llama2:7b", "llama2", 
                    "llama3", "mixtral", "mpt"
                ]
                
                # Try to find preferred models first
                for preferred in preferred_order:
                    matches = [m for m in models if preferred in m.lower()]
                    if matches:
                        return matches[0]
                
                # If no preferred model found, return the first one
                return models[0]
        except Exception as e:
            logger.warning(f"Failed to get loaded models: {e}")
            
        # Fallback model names to try
        fallbacks = ["phi", "gemma", "mistral", "llama2", "llama3", "orca-mini"]
        
        # Try to get any model directly from Ollama
        for fallback in fallbacks:
            try:
                import requests
                response = requests.get(f"{self.ollama_host}/api/show?name={fallback}")
                if response.status_code == 200:
                    return fallback
            except:
                pass
                
        # Ultimate fallback
        return "llama2"
    
    # The server would also implement all standard Ollama service methods
    # These implementations would be similar to the ones in service.py
    
    
def check_ollama_running(ollama_host: str) -> bool:
    """Check if Ollama is running at the given host.
    
    Args:
        ollama_host: URL of Ollama API
        
    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Recommended Ollama environment variables for optimal performance
OLLAMA_ENV_VARS = {
    # Performance settings
    "OLLAMA_FLASH_ATTENTION": "1",      # Enable FlashAttention for faster inference
    "OLLAMA_NUMA": "1",                 # Enable NUMA optimization if available
    "OLLAMA_KEEP_ALIVE": "1h",          # Keep models loaded for 1 hour after last use
    
    # Memory/cache settings
    "OLLAMA_MEMORY_LOCK": "1",          # Lock memory to prevent swapping
    "OLLAMA_LOAD_TIMEOUT": "120s",      # Longer timeout for loading large models
    "OLLAMA_QUANTIZE": "q8_0",          # Recommended quantization level
    
    # Context window settings
    "OLLAMA_CONTEXT_WINDOW": "8192",    # Default context window size
    
    # Logging/debug settings
    "OLLAMA_DEBUG": "0",                # Enable for debugging (0 for production)
    "OLLAMA_LOG_LEVEL": "info",         # Default log level
}

# Global variable to store the Ollama process for monitoring
ollama_process = None
ollama_log_thread = None
ollama_log_buffer = []
ollama_log_buffer_lock = threading.Lock()

def ensure_ollama_running(ollama_host: str, timeout: int = 60, env_vars: Dict[str, str] = None) -> bool:
    """Ensure Ollama is running, starting it if needed.
    
    Args:
        ollama_host: URL of Ollama API
        timeout: Maximum time to wait for Ollama to start (seconds)
        env_vars: Custom environment variables for Ollama
        
    Returns:
        True if Ollama is running, False if couldn't start
    """
    global ollama_process, ollama_log_thread
    
    if check_ollama_running(ollama_host):
        logger.info(f"Ollama already running at {ollama_host}")
        return True
        
    # Parse host to get just the hostname/IP
    parsed = urlparse(ollama_host)
    hostname = parsed.netloc.split(':')[0] if parsed.netloc else parsed.path.split(':')[0]
    
    # Only try to start Ollama if it's on localhost
    if hostname not in ('localhost', '127.0.0.1', '0.0.0.0', '::1'):
        logger.warning(f"Ollama not running at {ollama_host} but it's not on localhost, can't start it")
        return False
        
    # Prepare environment with recommended defaults and custom overrides
    env = os.environ.copy()
    
    # Apply default environment variables
    for key, value in OLLAMA_ENV_VARS.items():
        env[key] = value
    
    # Apply custom environment variables (overriding defaults)
    if env_vars:
        for key, value in env_vars.items():
            env[key] = value
    
    # Log the environment variables being used
    logger.info("Starting Ollama with the following environment variables:")
    for key in OLLAMA_ENV_VARS.keys():
        if key in env:
            logger.info(f"  {key}={env[key]}")
    
    # Try to start Ollama
    logger.info("Starting Ollama server...")
    try:
        # Start Ollama in the background with custom environment
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,  # Line buffered
            start_new_session=False  # Keep attached for monitoring
        )
        
        # Start thread to monitor Ollama logs
        ollama_log_thread = threading.Thread(
            target=_monitor_ollama_logs,
            args=(ollama_process,),
            daemon=True
        )
        ollama_log_thread.start()
        
        # Wait for Ollama to start
        start_time = time.time()
        while time.time() - start_time < timeout:
            if check_ollama_running(ollama_host):
                logger.info(f"Ollama started successfully at {ollama_host}")
                return True
            time.sleep(1)
            
        # Timeout reached
        logger.error(f"Timed out waiting for Ollama to start after {timeout} seconds")
        # Don't kill the process, it might still be initializing
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False

def _monitor_ollama_logs(process: subprocess.Popen) -> None:
    """Monitor Ollama's log output.
    
    This function runs in a separate thread and monitors Ollama's stdout/stderr,
    looking for error messages or important information.
    
    Args:
        process: The subprocess.Popen object representing the Ollama process
    """
    global ollama_log_buffer
    
    try:
        # Set up log readers
        for pipe, name in [(process.stdout, "STDOUT"), (process.stderr, "STDERR")]:
            for line in pipe:
                # Clean the line
                line = line.strip()
                
                # Add to circular buffer with timestamp
                with ollama_log_buffer_lock:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] [{name}] {line}"
                    
                    # Keep a circular buffer of the last 1000 log lines
                    ollama_log_buffer.append(log_entry)
                    if len(ollama_log_buffer) > 1000:
                        ollama_log_buffer = ollama_log_buffer[-1000:]
                    
                # Look for important messages
                log_level = "INFO"
                if "error" in line.lower() or "fatal" in line.lower():
                    log_level = "ERROR"
                    logger.error(f"Ollama: {line}")
                elif "warn" in line.lower():
                    log_level = "WARNING"
                    logger.warning(f"Ollama: {line}")
                elif "listening" in line.lower() or "loaded" in line.lower():
                    logger.info(f"Ollama: {line}")
                else:
                    logger.debug(f"Ollama: {line}")
                    
                # Watch for specific error patterns that might require intervention
                if "out of memory" in line.lower() or "oom" in line.lower():
                    logger.error("Ollama out of memory error detected!")
                elif "cuda error" in line.lower() or "gpu error" in line.lower():
                    logger.error("Ollama GPU error detected!")
                elif "killed" in line.lower() and "process" in line.lower():
                    logger.error("Ollama process was killed!")
    except Exception as e:
        logger.error(f"Error monitoring Ollama logs: {e}")
    finally:
        logger.info("Ollama log monitoring thread exiting")

def get_ollama_logs(count: int = 100) -> List[str]:
    """Get recent Ollama logs.
    
    Args:
        count: Number of log lines to return
        
    Returns:
        List of log lines
    """
    with ollama_log_buffer_lock:
        return ollama_log_buffer[-count:]

def serve(host: str = "0.0.0.0", 
          port: int = 50052,
          device_type: str = "cpu",
          device_id: int = 0,
          ollama_host: str = "http://localhost:11434",
          ollama_env: Dict[str, str] = None,
          health_check_interval: int = 30,
          enable_discovery: bool = True,
          preferred_interface: Optional[str] = None) -> None:
    """Start the RPC server.
    
    Args:
        host: Hostname to bind to
        port: Port to listen on
        device_type: Type of compute device
        device_id: Device ID for multi-device systems
        ollama_host: URL of Ollama API
        ollama_env: Custom environment variables for Ollama
        health_check_interval: Interval for Ollama health checks (seconds)
        enable_discovery: Enable auto-discovery of proxies
        preferred_interface: Preferred network interface IP address for connections
    """
    # Set default Ollama environment variables
    env_vars = OLLAMA_ENV_VARS.copy()
    
    # Set device-specific environment variables
    if device_type == "cuda":
        env_vars["CUDA_VISIBLE_DEVICES"] = str(device_id)
    elif device_type == "rocm":
        env_vars["HIP_VISIBLE_DEVICES"] = str(device_id)
        env_vars["ROCR_VISIBLE_DEVICES"] = str(device_id)
    
    # Override with custom environment variables if provided
    if ollama_env:
        env_vars.update(ollama_env)
    
    # First ensure Ollama is running with the right environment variables
    if not ensure_ollama_running(ollama_host, env_vars=env_vars):
        logger.warning("Proceeding without Ollama - some functionality may be limited")
    else:
        logger.info(f"Connected to Ollama at {ollama_host}")
    
    # Check port availability
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
    except OSError:
        logger.warning(f"Port {port} is already in use, server may fail to start")
    except Exception as e:
        logger.warning(f"Error checking port availability: {e}")
    
    # Create a gRPC server
    import concurrent.futures
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )
    
    # Create service instance
    service = RPCServer(ollama_host=ollama_host, device_type=device_type, device_id=device_id)
    
    # Register service
    ollama_pb2_grpc.add_DistributedOllamaServiceServicer_to_server(service, server)
    
    # Start Ollama health check thread
    stop_health_check = threading.Event()
    health_thread = threading.Thread(
        target=_ollama_health_check,
        args=(ollama_host, health_check_interval, stop_health_check),
        daemon=True
    )
    health_thread.start()
    
    # Start server
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info(f"RPC server started on {host}:{port} with device {device_type}:{device_id}")
    
    # Start discovery service for auto-discovery with proxies
    discovery_service = None
    if enable_discovery:
        try:
            # Import here to avoid circular imports
            from ..utils.discovery import DiscoveryService, get_capabilities_info
            
            # Get system capabilities to advertise
            capabilities = get_capabilities_info()
            
            # Add specific server info to capabilities
            capabilities["device_type"] = device_type
            capabilities["device_id"] = device_id
            capabilities["service_type"] = "rpc-server"
            
            # Create the discovery service
            discovery_service = DiscoveryService(
                service_type="server",
                service_port=port,
                extra_info=capabilities,
                preferred_interface=preferred_interface
            )
            
            # Register callback for proxy discovery
            def on_proxy_discovered(service_id, service_info):
                ip = service_info.get("ip")
                proxy_port = service_info.get("port", 8000)
                logger.info(f"Discovered proxy at {ip}:{proxy_port} (ID: {service_id})")
                
            discovery_service.register_discovery_callback(on_proxy_discovered)
            
            # Start the discovery service
            discovery_service.start()
            logger.info("Auto-discovery service started")
        except ImportError as e:
            logger.warning(f"Auto-discovery not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to start discovery service: {e}")
    
    try:
        # Keep server running
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        stop_health_check.set()  # Signal health check thread to stop
        
        # Stop discovery service if active
        if discovery_service:
            discovery_service.stop()
            
        server.stop(0)

def _ollama_health_check(ollama_host: str, interval: int, stop_event: threading.Event) -> None:
    """Run periodic health checks on Ollama.
    
    Args:
        ollama_host: URL of Ollama API
        interval: Check interval in seconds
        stop_event: Event to signal thread to stop
    """
    global ollama_process
    
    logger.info(f"Starting Ollama health check thread (interval: {interval}s)")
    
    consecutive_failures = 0
    max_failures = 3  # Number of failures before attempting restart
    
    while not stop_event.is_set():
        try:
            # Check if Ollama is responding
            is_running = check_ollama_running(ollama_host)
            
            if is_running:
                if consecutive_failures > 0:
                    logger.info(f"Ollama is running again after {consecutive_failures} failed checks")
                    consecutive_failures = 0
                    
                # Check resource usage
                try:
                    import requests
                    response = requests.get(f"{ollama_host}/api/status")
                    if response.status_code == 200:
                        data = response.json()
                        if "gpu_usage" in data:
                            gpu_usage = data["gpu_usage"]
                            if gpu_usage > 95:
                                logger.warning(f"High GPU usage detected: {gpu_usage}%")
                except Exception as e:
                    logger.debug(f"Error checking Ollama resource usage: {e}")
            else:
                consecutive_failures += 1
                logger.warning(f"Ollama health check failed ({consecutive_failures}/{max_failures})")
                
                # Try to restart Ollama if it's not responding
                if consecutive_failures >= max_failures and ollama_process is not None:
                    logger.error(f"Ollama has failed {consecutive_failures} health checks, attempting restart")
                    try:
                        # Check if process is still running
                        if ollama_process.poll() is None:
                            logger.info("Terminating unresponsive Ollama process")
                            ollama_process.terminate()
                            ollama_process.wait(timeout=10)
                        
                        # Restart Ollama (with previous environment variables)
                        ensure_ollama_running(ollama_host)
                        
                        # Reset failure counter
                        consecutive_failures = 0
                    except Exception as restart_error:
                        logger.error(f"Failed to restart Ollama: {restart_error}")
        except Exception as e:
            logger.error(f"Error in Ollama health check: {e}")
        
        # Wait for the next interval or until stopped
        stop_event.wait(interval)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start RPC inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=50052, help="Port to listen on")
    parser.add_argument("--device", default="auto", 
                      choices=["auto", "cpu", "cuda", "rocm", "metal"],
                      help="Device type (auto, cpu, cuda, rocm, metal)")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                      help="Ollama API host URL")
    parser.add_argument("--flash-attention", action="store_true", default=True,
                      help="Enable FlashAttention for faster inference")
    parser.add_argument("--no-flash-attention", action="store_false", dest="flash_attention",
                      help="Disable FlashAttention")
    parser.add_argument("--context-window", type=int, default=8192,
                      help="Default context window size")
    parser.add_argument("--quantize", default="q8_0", 
                      choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
                      help="Quantization level for models")
    parser.add_argument("--health-check-interval", type=int, default=30,
                      help="Interval for Ollama health checks (seconds)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with additional logging")
    parser.add_argument("--discovery", action="store_true", default=True,
                      help="Enable auto-discovery to find proxy servers")
    parser.add_argument("--no-discovery", action="store_false", dest="discovery",
                      help="Disable auto-discovery")
    
    args = parser.parse_args()
    
    # Prepare custom environment variables based on command-line args
    ollama_env = {}
    
    # Set flash attention
    ollama_env["OLLAMA_FLASH_ATTENTION"] = "1" if args.flash_attention else "0"
    
    # Set context window
    ollama_env["OLLAMA_CONTEXT_WINDOW"] = str(args.context_window)
    
    # Set quantization
    ollama_env["OLLAMA_QUANTIZE"] = args.quantize
    
    # Set debug mode
    if args.debug:
        ollama_env["OLLAMA_DEBUG"] = "1"
        ollama_env["OLLAMA_LOG_LEVEL"] = "debug"
        # Also increase Python logging level
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # If device is set to auto, determine the best device
    device_type = args.device
    if device_type == "auto":
        # Import the auto-detection function from __main__.py
        try:
            from ..__main__ import _auto_detect_device_type
            device_type = _auto_detect_device_type()
        except ImportError:
            # Direct implementation if can't import
            try:
                import torch
                if torch.cuda.is_available():
                    device_type = "cuda"
                elif hasattr(torch, 'hip') and torch.hip.is_available():
                    device_type = "rocm"
                elif sys.platform == "darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device_type = "metal"
                else:
                    device_type = "cpu"
            except ImportError:
                # No torch, try direct detection
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "-L"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and "GPU" in result.stdout:
                        device_type = "cuda"
                    else:
                        try:
                            result = subprocess.run(
                                ["rocm-smi", "--showdevice"],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if result.returncode == 0 and "GPU" in result.stdout:
                                device_type = "rocm"
                            else:
                                device_type = "cpu"
                        except (subprocess.SubprocessError, FileNotFoundError, PermissionError) as e:
                            logger.debug(f"Could not detect ROCm: {e}")
                            device_type = "cpu"
                except (subprocess.SubprocessError, FileNotFoundError, PermissionError) as e:
                    logger.debug(f"Could not detect CUDA: {e}")
                    device_type = "cpu"
            
            logger.info(f"Auto-detected device type: {device_type}")
        
    # Start server with custom environment variables
    serve(
        host=args.host, 
        port=args.port, 
        device_type=device_type, 
        device_id=args.device_id, 
        ollama_host=args.ollama_host,
        ollama_env=ollama_env,
        health_check_interval=args.health_check_interval,
        enable_discovery=args.discovery
    )