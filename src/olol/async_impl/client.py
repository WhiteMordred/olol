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
    
    # Legacy methods for backward compatibility
    async def run_model(self, model_name: str, prompt: str):
        """Run a one-off model query"""
        try:
            request = ollama_pb2.ModelRequest(
                model_name=model_name,
                prompt=prompt
            )
            response = await self.stub.RunModel(request)
            return response.output
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"gRPC error: {e}")
            raise
    
    async def create_session(self, session_id: str, model_name: str):
        """Create a new chat session"""
        try:
            request = ollama_pb2.SessionRequest(
                session_id=session_id,
                model_name=model_name
            )
            response = await self.stub.CreateSession(request)
            return response.success
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"gRPC error: {e}")
            raise

    async def chat_message(self, session_id: str, message: str):
        """Send a message in an existing chat session"""
        try:
            request = ollama_pb2.ChatRequest(
                session_id=session_id,
                message=message
            )
            response = await self.stub.ChatMessage(request)
            return response.output
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"gRPC error: {e}")
            raise
        
    # New methods matching the latest Ollama API
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
            request = ollama_pb2.GenerateRequest(
                model=model,
                prompt=prompt,
                stream=stream,
                options=options or {}
            )
            async for response in self.stub.Generate(request):
                yield response
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"gRPC error: {e}")
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
            
            async for response in self.stub.Chat(request):
                yield response
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"gRPC error: {e}")
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
            request = ollama_pb2.EmbeddingsRequest(
                model=model,
                prompt=prompt,
                options=options or {}
            )
            response = await self.stub.Embeddings(request)
            return response
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"gRPC error: {e}")
            raise
    
    async def close(self) -> None:
        """Close the channel."""
        self.channel.close()


async def main() -> None:
    """Example usage of the AsyncOllamaClient."""
    client = AsyncOllamaClient()
    try:
        # Simple model run (legacy API)
        response = await client.run_model("llama2", "What is the capital of France?")
        print(f"Legacy API response: {response}")
        
        # New API with streaming
        print("\nNew API with streaming:")
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