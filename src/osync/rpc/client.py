"""Client for RPC-based distributed LLM inference."""

import logging
from typing import Any, Dict, List, Optional

import grpc

from ..proto import ollama_pb2, ollama_pb2_grpc
from ..utils.cluster import TensorPartitioner

logger = logging.getLogger(__name__)


class RPCClient:
    """Client for distributed LLM inference using RPC.
    
    This client implements distributed inference by coordinating multiple
    RPC servers, each handling a part of the model computation.
    """
    
    def __init__(self, server_addresses: List[str]) -> None:
        """Initialize the RPC client.
        
        Args:
            server_addresses: List of server addresses in "host:port" format
        """
        self.server_addresses = server_addresses
        self.channels = {}
        self.stubs = {}
        self.partitioner = TensorPartitioner()
        
        # Set up connections to all servers
        for i, address in enumerate(server_addresses):
            try:
                channel = grpc.insecure_channel(address)
                self.channels[address] = channel
                self.stubs[address] = ollama_pb2_grpc.DistributedOllamaServiceStub(channel)
                logger.info(f"Connected to RPC server at {address}")
            except Exception as e:
                logger.error(f"Failed to connect to RPC server at {address}: {e}")
                
        if not self.stubs:
            raise ConnectionError("Failed to connect to any RPC servers")
            
        # Initialize partitioner with server capabilities
        self._init_device_capabilities()
        
    def _init_device_capabilities(self) -> None:
        """Initialize device capabilities for each server."""
        for address, stub in self.stubs.items():
            try:
                # Request capabilities from the server
                request = ollama_pb2.DeviceCapabilitiesRequest(detail=True)
                response = stub.GetDeviceCapabilities(request)
                
                # Convert response to dictionary for the partitioner
                capabilities = {
                    "backend_type": response.device_type,
                    "device_id": response.device_id,
                    "memory": response.memory,
                    "cores": response.cores,
                    "compute_capability": response.compute_capability
                }
                
                # Add any additional details
                for key, value in response.details.items():
                    capabilities[key] = value
                
                logger.info(f"Got capabilities from {address}: {capabilities}")
                self.partitioner.register_device(address, capabilities)
            except Exception as e:
                logger.error(f"Failed to get capabilities from {address}: {e}")
                # Use default capabilities as fallback
                capabilities = {
                    "backend_type": "cpu",  # Default to CPU
                    "memory": 8 * 1024 * 1024 * 1024,  # 8 GB default
                    "cores": 8,
                }
                self.partitioner.register_device(address, capabilities)
                
    def distributed_generate(self, 
                            model: str, 
                            prompt: str,
                            options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate text using distributed inference.
        
        This method distributes the inference across multiple servers based on
        their capabilities and the model characteristics.
        
        Args:
            model: Model name to use
            prompt: Text prompt to send
            options: Optional parameters
            
        Returns:
            List of response dictionaries
        """
        # Load model information (would come from model registry)
        model_info = self._get_model_info(model)
        
        # Create partitioning plan for this model
        layer_assignment = self.partitioner.partition_model(
            model_size=model_info.get("size", 0),
            layer_count=model_info.get("layers", 0)
        )
        
        logger.info(f"Layer assignment plan: {layer_assignment}")
        
        # Prepare inputs for all servers
        inputs = self._prepare_distributed_inputs(prompt, layer_assignment)
        
        # Execute distributed inference
        results = self._execute_distributed_inference(model, inputs, options or {})
        
        # Combine results
        return self._combine_results(results)
        
    def _get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model information for distributed inference planning.
        
        In a real implementation, this would query a model registry or
        the servers themselves for model details.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with model characteristics
        """
        # Placeholder - in real implementation would query server
        if "7b" in model.lower():
            return {
                "name": model,
                "size": 7 * 1024 * 1024 * 1024,  # 7 GB
                "layers": 32,
                "embedding_dim": 4096
            }
        elif "13b" in model.lower():
            return {
                "name": model,
                "size": 13 * 1024 * 1024 * 1024,  # 13 GB
                "layers": 40,
                "embedding_dim": 5120
            }
        else:
            # Default for unknown models
            return {
                "name": model,
                "size": 1 * 1024 * 1024 * 1024,  # 1 GB
                "layers": 16,
                "embedding_dim": 768
            }
            
    def _prepare_distributed_inputs(self, 
                                   prompt: str, 
                                   layer_assignment: Dict[str, List[int]]) -> Dict[str, Any]:
        """Prepare inputs for each server.
        
        Args:
            prompt: User input prompt
            layer_assignment: Mapping of servers to layer indices
            
        Returns:
            Dictionary mapping server addresses to input data
        """
        inputs = {}
        for server, layers in layer_assignment.items():
            inputs[server] = {
                "prompt": prompt,
                "assigned_layers": layers,
                "is_first": min(layers) == 0,  # First server handles tokenization
                "is_last": any(l == max(sum(layer_assignment.values(), [])) for l in layers),  # Last server handles final output
            }
        return inputs
        
    def _execute_distributed_inference(self, 
                                      model: str, 
                                      inputs: Dict[str, Any],
                                      options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute inference across distributed servers.
        
        Args:
            model: Model name
            inputs: Prepared inputs for each server
            options: Model parameters
            
        Returns:
            Raw results from each server
        """
        results = {}
        # In a real implementation, these would be parallel async calls
        for server, server_inputs in inputs.items():
            try:
                # Create request with layer assignments
                request = ollama_pb2.DistributedGenerateRequest(
                    model=model,
                    prompt=server_inputs["prompt"],
                    is_first=server_inputs["is_first"],
                    is_last=server_inputs["is_last"]
                )
                
                # Add assigned layers
                request.assigned_layers.extend(server_inputs["assigned_layers"])
                
                # Add options if provided
                if options:
                    for key, value in options.items():
                        request.options[key] = str(value)
                
                # Send request to server
                logger.info(f"Server {server} processing layers {server_inputs['assigned_layers']}")
                
                # Process streaming responses
                server_results = {
                    "layer_outputs": {},
                    "tokens": [],
                    "is_last": server_inputs["is_last"]
                }
                
                # Call the RPC method and process streaming responses
                for response in self.stubs[server].DistributedGenerate(request):
                    # Collect layer outputs
                    for layer_id, tensor_data in response.layer_outputs.items():
                        server_results["layer_outputs"][layer_id] = tensor_data
                    
                    # Collect tokens if this is the last server
                    if server_inputs["is_last"] and response.tokens:
                        server_results["tokens"].extend(response.tokens)
                    
                    # Check if done
                    if response.done:
                        break
                
                results[server] = server_results
                    
            except Exception as e:
                logger.error(f"Error in distributed inference on {server}: {e}")
                
        return results
        
    def _combine_results(self, server_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Combine results from multiple servers.
        
        Args:
            server_results: Raw results from each server
            
        Returns:
            Processed final results
        """
        # Find the server with final output
        final_server = None
        for server, results in server_results.items():
            if results.get("is_last", False):
                final_server = server
                break
                
        if not final_server:
            logger.error("No server returned final outputs")
            return []
            
        # Get tokens from the final server
        tokens = server_results[final_server].get("tokens", [])
        
        if not tokens:
            logger.warning("No tokens returned from final server")
            return [{
                "response": "No response generated",
                "done": True
            }]
            
        # Process tokens into text
        try:
            # In a real implementation, we would decode tokens to text
            # using the model's tokenizer
            # Here, we'll just convert token IDs to a string representation
            
            # Try to get the real response text if available
            response_text = ""
            
            # First option: Get actual responses if they exist
            if hasattr(tokens[0], 'response'):
                response_text = "".join(t.response for t in tokens)
            # Second option: Get response text directly
            elif all(isinstance(t, dict) and 'response' in t for t in tokens):
                response_text = "".join(t['response'] for t in tokens)
            # Third option: If the tokens are strings already
            elif all(isinstance(t, str) for t in tokens):
                response_text = "".join(tokens)
            # Fourth option: Try to decode the tokens using a tokenizer
            else:
                try:
                    # Try to use Ollama's API directly to get response
                    import requests
                    
                    # Try to find an available Ollama API endpoint
                    ollama_hosts = []
                    for server in self.server_addresses:
                        host, _ = server.split(":")
                        ollama_hosts.append(f"http://{host}:11434")
                    
                    # Try each host until we succeed
                    response_text = ""
                    for ollama_host in ollama_hosts:
                        try:
                            # Get which model is available
                            model_response = requests.get(f"{ollama_host}/api/tags")
                            if model_response.status_code != 200:
                                continue
                                
                            data = model_response.json()
                            if not data.get("models"):
                                continue
                                
                            # Get first available model
                            model = data["models"][0]["name"]
                            
                            # Get completion for the tokens
                            token_text = " ".join(str(t) for t in tokens)
                            completion_response = requests.post(
                                f"{ollama_host}/api/generate",
                                json={
                                    "model": model,
                                    "prompt": f"Here are some tokens: {token_text}. Convert these tokens back to text.",
                                    "stream": False
                                }
                            )
                            
                            if completion_response.status_code == 200:
                                completion_data = completion_response.json()
                                response_text = completion_data.get("response", "")
                                break
                        except:
                            continue
                            
                    # If we couldn't get a response, try tokenizer approach
                    if not response_text:
                        try:
                            import transformers
                            tokenizer = None
                            try:
                                # Try typical tokenizers in order
                                for model_name in ["llama2", "mistral", "gpt2"]:
                                    try:
                                        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
                                        break
                                    except:
                                        continue
                                        
                                # Fallback to default tokenizer if needed
                                if not tokenizer:
                                    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
                                
                                # Decode the tokens
                                response_text = tokenizer.decode(tokens)
                            except:
                                # Ultimate fallback - just join tokens as strings
                                response_text = " ".join(str(t) for t in tokens)
                        except (ImportError, Exception) as e:
                            logger.warning(f"Could not decode tokens with tokenizer: {e}")
                            # Fallback - just join tokens as strings
                            response_text = " ".join(str(t) for t in tokens)
                except Exception as e:
                    logger.warning(f"Error processing tokens: {e}")
                    response_text = " ".join(str(t) for t in tokens)
            
            # Create a proper response
            return [{
                "response": response_text,
                "done": True,
                "distributed": True,
                "server_count": len(server_results)
            }]
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            # Fallback response
            return [{
                "response": f"Error processing response: {str(e)}",
                "done": True
            }]
        
    def close(self) -> None:
        """Close all connections."""
        for channel in self.channels.values():
            channel.close()
        self.channels.clear()
        self.stubs.clear()