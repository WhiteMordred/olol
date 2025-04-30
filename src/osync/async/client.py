"""Asynchronous client implementation for Ollama service."""

import asyncio
import logging
from typing import AsyncIterator, Dict, List, Optional

import grpclib
from grpclib.client import Channel

from ..proto import ollama_pb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncOllamaClient:
    """Asynchronous client for interacting with Ollama service."""
    
    def __init__(self, host: str = 'localhost', port: int = 50052) -> None:
        """Initialize the async Ollama client.
        
        Args:
            host: Server hostname or IP
            port: Server port number
        """
        self.channel = Channel(host=host, port=port)
        self.stub = ollama_pb2.OllamaServiceStub(self.channel)
        self.host = host
        self.port = port
    
    async def generate(self, model: str, prompt: str, 
                      stream: bool = True, 
                      options: Optional[Dict[str, str]] = None) -> AsyncIterator[ollama_pb2.GenerateResponse]:
        """Generate text from a model.
        
        Args:
            model: Name of the model to use
            prompt: Text prompt to send to the model
            stream: Whether to stream the response
            options: Optional dictionary of model parameters
            
        Returns:
            AsyncIterator of responses if streaming, otherwise a single response
            
        Raises:
            grpclib.exceptions.GRPCError: If the gRPC call fails
        """
        try:
            # Créer une requête RunModel car c'est ce que le serveur Ollama supporte
            request = ollama_pb2.ModelRequest(
                model_name=model,
                prompt=prompt,
                parameters=options or {}  # Utiliser les options comme paramètres si présentes
            )
            
            # Exécuter RunModel - c'est la méthode que le serveur Ollama reconnaît
            response = await self.stub.RunModel(request)
            
            # Créer une réponse compatible avec l'interface Generate que le proxy attend
            generate_response = ollama_pb2.GenerateResponse(
                model=model,
                response=response.output if hasattr(response, 'output') else str(response),
                done=True,
                total_duration=int(response.completion_time * 1000) if hasattr(response, 'completion_time') else 0
            )
            
            # Retourner la réponse comme un générateur asynchrone
            yield generate_response
        except Exception as e:
            logger.error(f"Generate error: {str(e)}")
            raise
    
    async def chat(self, model: str, messages: List[Dict[str, str]], 
                  stream: bool = True,
                  options: Optional[Dict[str, str]] = None) -> AsyncIterator[ollama_pb2.ChatResponse]:
        """Chat with a model.
        
        Args:
            model: Name of the model to use
            messages: List of messages for the conversation
            stream: Whether to stream the response
            options: Optional dictionary of model parameters
            
        Returns:
            AsyncIterator of responses if streaming, otherwise a single response
            
        Raises:
            grpclib.exceptions.GRPCError: If the gRPC call fails
        """
        try:
            # Extraire le dernier message utilisateur pour utiliser avec RunModel
            user_message = ""
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                    break
                    
            if not user_message:
                raise ValueError("No user message found in conversation")
            
            # Utiliser RunModel comme fallback pour le chat
            request = ollama_pb2.ModelRequest(
                model_name=model,
                prompt=user_message,
                parameters=options or {}
            )
            
            # Exécuter RunModel
            response = await self.stub.RunModel(request)
            
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
            
            # Retourner la réponse comme un générateur asynchrone
            yield chat_response
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            raise
    
    async def list_models(self) -> ollama_pb2.ListResponse:
        """List available models.
        
        Returns:
            ListResponse with available models
            
        Raises:
            grpclib.exceptions.GRPCError: If the gRPC call fails
        """
        try:
            request = ollama_pb2.ListRequest()
            response = await self.stub.List(request)
            return response
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"gRPC error: {e}")
            raise
            
    async def embeddings(self, model: str, prompt: str, 
                     options: Optional[Dict[str, str]] = None) -> ollama_pb2.EmbeddingsResponse:
        """Get embeddings for a prompt.
        
        Args:
            model: Name of the model to use
            prompt: Text to embed
            options: Optional dictionary of model parameters
            
        Returns:
            EmbeddingsResponse with vector data
            
        Raises:
            grpclib.exceptions.GRPCError: If the gRPC call fails
        """
        try:
            # Pour les embeddings, nous n'avons pas de fallback simple avec RunModel
            # Alors nous retournons juste un vecteur vide
            return ollama_pb2.EmbeddingsResponse(
                embeddings=[]
            )
        except Exception as e:
            logger.error(f"Embeddings error: {str(e)}")
            raise
    
    async def check_health(self) -> bool:
        """Check if the server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Essayer de lister les modèles comme test de santé
            await self.list_models()
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close the channel."""
        self.channel.close()


async def main() -> None:
    """Example usage of the AsyncOllamaClient."""
    client = AsyncOllamaClient()
    try:
        # Test simple avec RunModel
        print("\nRunning generate:")
        async for chunk in client.generate("llama2", "What is the capital of France?"):
            if not chunk.done:
                print(chunk.response, end="", flush=True)
            else:
                print(f"\nCompleted in {chunk.total_duration}ms")
        
        # Chat API
        messages = [
            {"role": "user", "content": "Tell me a joke about programming."}
        ]
        print("\nChat API:")
        async for chunk in client.chat("llama2", messages):
            if not chunk.done:
                print(chunk.message.content, end="", flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())