"""Synchronous client implementation for Ollama service."""

import logging
from typing import Dict, Iterator, List, Optional

import grpc

from ..proto import ollama_pb2, ollama_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Synchronous client for interacting with Ollama service."""
    
    def __init__(self, host: str = 'localhost', port: int = 50051) -> None:
        """Initialize the Ollama client.
        
        Args:
            host: Server hostname or IP
            port: Server port number
        """
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = ollama_pb2_grpc.OllamaServiceStub(self.channel)
        self.host = host
        self.port = port
        
    def generate(self, model: str, prompt: str, 
                stream: bool = True, 
                options: Optional[Dict[str, str]] = None) -> Iterator[ollama_pb2.GenerateResponse]:
        """Generate text from a model.
        
        Args:
            model: Name of the model to use
            prompt: Text prompt to send to the model
            stream: Whether to stream the response
            options: Optional dictionary of model parameters
            
        Returns:
            Iterator of responses if streaming, otherwise a single response
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        try:
            request = ollama_pb2.GenerateRequest(
                model=model,
                prompt=prompt,
                stream=stream,
                options=options or {}
            )
            responses = self.stub.Generate(request)
            return responses
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            raise
    
    def chat(self, model: str, messages: List[Dict[str, str]], 
            stream: bool = True,
            options: Optional[Dict[str, str]] = None) -> Iterator[ollama_pb2.ChatResponse]:
        """Chat with a model.
        
        Args:
            model: Name of the model to use
            messages: List of messages for the conversation
            stream: Whether to stream the response
            options: Optional dictionary of model parameters
            
        Returns:
            Iterator of responses if streaming, otherwise a single response
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        try:
            proto_messages = [
                ollama_pb2.Message(
                    role=msg["role"],
                    content=msg["content"]
                ) for msg in messages
            ]
            
            request = ollama_pb2.ChatRequest(
                model=model,
                messages=proto_messages,
                stream=stream,
                options=options or {}
            )
            
            responses = self.stub.Chat(request)
            return responses
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            raise
    
    def list_models(self) -> ollama_pb2.ListResponse:
        """List available models.
        
        Returns:
            ListResponse with available models
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        try:
            request = ollama_pb2.ListRequest()
            response = self.stub.List(request)
            return response
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            raise
            
    def embeddings(self, model: str, prompt: str, options: Optional[Dict[str, str]] = None) -> ollama_pb2.EmbeddingsResponse:
        """Get embeddings for text.
        
        Args:
            model: Name of the model to use
            prompt: Text to get embeddings for
            options: Optional parameters
            
        Returns:
            EmbeddingsResponse containing the embeddings
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        try:
            request = ollama_pb2.EmbeddingsRequest(
                model=model,
                prompt=prompt,
                options=options or {}
            )
            response = self.stub.Embeddings(request)
            return response
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            raise
            
    def pull_model(self, model: str) -> Iterator[ollama_pb2.PullResponse]:
        """Pull a model.
        
        Args:
            model: Name of the model to pull
            
        Returns:
            Iterator of PullResponse with progress updates
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        try:
            request = ollama_pb2.PullRequest(model=model)
            return self.stub.Pull(request)
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            raise
            
    def get_version(self) -> ollama_pb2.VersionInfo:
        """Get API version information.
        
        Returns:
            VersionInfo with API version details
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        try:
            request = ollama_pb2.VersionRequest(detail=True)
            try:
                # Try the new version API first
                response = self.stub.GetVersion(request)
                return response
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                    # Fallback for older servers - create a default version response
                    logger.warning("Server does not support GetVersion API, using fallback")
                    return ollama_pb2.VersionInfo(
                        api_version=ollama_pb2.ApiVersion.V1_0_0,
                        version_string="unknown",
                        protocol_version=1
                    )
                else:
                    # Some other gRPC error occurred
                    raise
        except Exception as e:
            logger.error(f"Version check failed: {str(e)}")
            raise
    
    def check_health(self) -> bool:
        """Check if the server is healthy.
        
        Uses dedicated health check API if available, with fallback to listing models.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # First try dedicated health check API
            request = ollama_pb2.HealthCheckRequest(detail=True)
            try:
                response = self.stub.HealthCheck(request)
                return response.healthy
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                    # Fallback for older servers - try list_models instead
                    logger.debug("Server does not support HealthCheck API, using fallback")
                    try:
                        self.list_models()
                        return True
                    except Exception as list_err:
                        logger.error(f"Fallback health check failed: {str(list_err)}")
                        return False
                else:
                    # Some other gRPC error occurred
                    logger.error(f"Health check failed: {e.code()}: {e.details()}")
                    return False
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close the channel."""
        self.channel.close()


def main() -> None:
    """Example usage of the OllamaClient."""
    client = OllamaClient()
    try:
        print("Available models:")
        models = client.list_models()
        for model in models.models:
            print(f"- {model.name} ({model.parameter_size}, {model.quantization_level})")
        
        # Simple generation
        model_name = "llama2"  # replace with an available model
        prompt = "What is the capital of France?"
        
        print(f"\nGenerating response for: {prompt}")
        for response in client.generate(model_name, prompt):
            if not response.done:
                print(response.response, end='', flush=True)
            else:
                print("\nGeneration completed in", response.total_duration, "ms")
    finally:
        client.close()


if __name__ == "__main__":
    main()