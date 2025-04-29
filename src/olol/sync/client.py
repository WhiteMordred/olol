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
        
        This method automatically falls back to RunModel if the server doesn't support Generate.
        
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
            try:
                # Essayer d'abord la méthode moderne Generate
                responses = self.stub.Generate(request)
                return responses
            except grpc.RpcError as e:
                # Si la commande "generate" n'est pas reconnue, utiliser RunModel
                if "unknown command" in e.details() and "generate" in e.details():
                    logger.warning(f"Server doesn't support Generate API, falling back to RunModel. Error: {e.details()}")
                    return self._generate_with_runmodel(model, prompt, stream, options)
                else:
                    # Autre erreur gRPC, la propager
                    logger.error(f"gRPC error: {e.code()}: {e.details()}")
                    raise
        except Exception as e:
            logger.error(f"Generate error: {str(e)}")
            raise
            
    def _generate_with_runmodel(self, model: str, prompt: str, 
                              stream: bool = True, 
                              options: Optional[Dict[str, str]] = None) -> Iterator[ollama_pb2.GenerateResponse]:
        """Implementation fallback using RunModel instead of Generate.
        
        Args:
            model: Name of the model to use
            prompt: Text prompt to send to the model
            stream: Whether to stream the response (ignored in this implementation)
            options: Optional dictionary of model parameters
            
        Returns:
            Iterator with a single GenerateResponse
        """
        try:
            # Convertir les options en paramètres pour RunModel
            parameters = {}
            if options:
                parameters = options
                
            # Créer une requête RunModel
            request = ollama_pb2.ModelRequest(
                model_name=model,
                prompt=prompt,
                parameters=parameters
            )
            
            # Exécuter RunModel (non-streaming)
            response = self.stub.RunModel(request)
            
            # Créer une réponse compatible avec l'interface Generate
            generate_response = ollama_pb2.GenerateResponse(
                model=model,
                response=response.output if hasattr(response, 'output') else str(response),
                done=True,
                total_duration=int(response.completion_time * 1000) if hasattr(response, 'completion_time') else 0
            )
            
            # Créer un itérateur simple qui ne renvoie qu'une seule réponse
            def response_iterator():
                yield generate_response
                
            return response_iterator()
        except Exception as e:
            logger.error(f"RunModel fallback error: {str(e)}")
            raise
            
    def chat(self, model: str, messages: List[Dict[str, str]], 
            stream: bool = True,
            options: Optional[Dict[str, str]] = None) -> Iterator[ollama_pb2.ChatResponse]:
        """Chat with a model.
        
        This method automatically falls back to LegacyChat or RunModel if the server doesn't support Chat.
        
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
            
            try:
                responses = self.stub.Chat(request)
                return responses
            except grpc.RpcError as e:
                # Si la commande "chat" n'est pas reconnue, essayer avec LegacyChat ou RunModel
                if "unknown command" in e.details() and "chat" in e.details():
                    logger.warning(f"Server doesn't support Chat API, falling back to RunModel. Error: {e.details()}")
                    return self._chat_with_runmodel(model, messages, stream, options)
                else:
                    # Autre erreur gRPC, la propager
                    logger.error(f"gRPC error: {e.code()}: {e.details()}")
                    raise
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            raise
            
    def _chat_with_runmodel(self, model: str, messages: List[Dict[str, str]], 
                          stream: bool = True, 
                          options: Optional[Dict[str, str]] = None) -> Iterator[ollama_pb2.ChatResponse]:
        """Implementation fallback using RunModel instead of Chat.
        
        Args:
            model: Name of the model to use
            messages: List of messages for the conversation
            stream: Whether to stream the response
            options: Optional parameters
            
        Returns:
            Iterator with single ChatResponse containing a single assistant message
        """
        try:
            # Extraire uniquement le dernier message utilisateur pour la compatibilité
            user_message = ""
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                    break
                    
            if not user_message:
                raise ValueError("No user message found in conversation")
                
            # Convertir les options en paramètres pour RunModel
            parameters = {}
            if options:
                parameters = options
                
            # Créer une requête RunModel
            request = ollama_pb2.ModelRequest(
                model_name=model,
                prompt=user_message,
                parameters=parameters
            )
            
            # Exécuter RunModel (non-streaming)
            response = self.stub.RunModel(request)
            
            # Créer une réponse compatible avec l'interface Chat
            assistant_message = ollama_pb2.Message(
                role="assistant",
                content=response.output if hasattr(response, 'output') else str(response)
            )
            
            chat_response = ollama_pb2.ChatResponse(
                model=model,
                message=assistant_message,
                done=True,
                total_duration=int(response.completion_time * 1000) if hasattr(response, 'completion_time') else 0
            )
            
            # Créer un itérateur simple qui ne renvoie qu'une seule réponse
            def response_iterator():
                yield chat_response
                
            return response_iterator()
        except Exception as e:
            logger.error(f"RunModel fallback for chat error: {str(e)}")
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
        
        This method automatically falls back to a compatible approach if Embeddings is not supported.
        
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
            try:
                response = self.stub.Embeddings(request)
                return response
            except grpc.RpcError as e:
                # Si la commande "embeddings" n'est pas reconnue, utiliser une méthode alternative
                if "unknown command" in e.details() and "embeddings" in e.details():
                    logger.warning(f"Server doesn't support Embeddings API, falling back to RunModel. Error: {e.details()}")
                    return self._embeddings_with_runmodel(model, prompt, options)
                else:
                    # Autre erreur gRPC, la propager
                    logger.error(f"gRPC error: {e.code()}: {e.details()}")
                    raise
        except Exception as e:
            logger.error(f"Embeddings error: {str(e)}")
            raise
            
    def _embeddings_with_runmodel(self, model: str, prompt: str, 
                               options: Optional[Dict[str, str]] = None) -> ollama_pb2.EmbeddingsResponse:
        """Implementation fallback for embeddings.
        
        Args:
            model: Name of the model to use
            prompt: Text to embed
            options: Optional parameters
            
        Returns:
            EmbeddingsResponse with empty embeddings
            
        Note:
            This is a placeholder fallback since RunModel doesn't provide embeddings.
            In a real implementation, you might want to use a different approach.
        """
        # Pour la compatibilité, on renvoie une réponse vide
        logger.warning("Embeddings not available in this version of Ollama server")
        return ollama_pb2.EmbeddingsResponse(
            embeddings=[]
        )
            
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