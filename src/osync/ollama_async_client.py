import asyncio
import logging
from typing import AsyncIterator, Dict, List, Optional

import grpclib
import ollama_pb2
from grpclib.client import Channel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncOllamaClient:
    def __init__(self, host='localhost', port=50052):
        self.channel = Channel(host=host, port=port)
        self.stub = ollama_pb2.OllamaServiceStub(self.channel)

    async def generate(self, 
                      model: str, 
                      prompt: str, 
                      options: Optional[Dict] = None, 
                      stream: bool = True) -> AsyncIterator[ollama_pb2.GenerateResponse]:
        """Stream generate responses"""
        try:
            request = ollama_pb2.GenerateRequest(
                model=model,
                prompt=prompt,
                options=options or {},
                stream=stream
            )
            async for response in self.stub.Generate(request):
                yield response
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Generate error: {e}")
            raise

    async def chat(self, 
                  model: str, 
                  messages: List[Dict], 
                  stream: bool = True) -> AsyncIterator[ollama_pb2.ChatResponse]:
        """Stream chat responses"""
        try:
            request = ollama_pb2.ChatRequest(
                model=model,
                messages=[ollama_pb2.Message(**m) for m in messages],
                stream=stream
            )
            async for response in self.stub.Chat(request):
                yield response
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Chat error: {e}")
            raise

    async def embeddings(self, model: str, prompt: str) -> List[float]:
        """Get embeddings for text"""
        try:
            request = ollama_pb2.EmbeddingsRequest(
                model=model,
                prompt=prompt
            )
            response = await self.stub.Embeddings(request)
            return response.embeddings
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Embeddings error: {e}")
            raise

    async def create_model(self, name: str, path: str, template: str) -> bool:
        """Create a new model"""
        try:
            request = ollama_pb2.CreateModelRequest(
                name=name,
                path=path,
                template=template
            )
            response = await self.stub.CreateModel(request)
            return response.success
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Create model error: {e}")
            raise

    async def list_models(self) -> List[ollama_pb2.Model]:
        """List available models"""
        try:
            request = ollama_pb2.ListModelsRequest()
            response = await self.stub.ListModels(request)
            return response.models
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"List models error: {e}")
            raise

    async def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model"""
        try:
            request = ollama_pb2.CopyModelRequest(
                source=source,
                destination=destination
            )
            response = await self.stub.CopyModel(request)
            return response.success
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Copy model error: {e}")
            raise

    async def delete_model(self, name: str) -> bool:
        """Delete a model"""
        try:
            request = ollama_pb2.DeleteModelRequest(name=name)
            response = await self.stub.DeleteModel(request)
            return response.success
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Delete model error: {e}")
            raise

    async def pull_model(self, name: str) -> AsyncIterator[ollama_pb2.PullResponse]:
        """Pull a model with progress updates"""
        try:
            request = ollama_pb2.PullModelRequest(name=name)
            async for response in self.stub.PullModel(request):
                yield response
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Pull model error: {e}")
            raise

    async def push_model(self, name: str) -> AsyncIterator[ollama_pb2.PushResponse]:
        """Push a model with progress updates"""
        try:
            request = ollama_pb2.PushModelRequest(name=name)
            async for response in self.stub.PushModel(request):
                yield response
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Push model error: {e}")
            raise

    async def show_model(self, name: str) -> ollama_pb2.Model:
        """Get model details"""
        try:
            request = ollama_pb2.ShowModelRequest(name=name)
            response = await self.stub.ShowModel(request)
            return response.model
        except grpclib.exceptions.GRPCError as e:
            logger.error(f"Show model error: {e}")
            raise

    async def close(self):
        """Close the channel"""
        self.channel.close()

# Example usage
async def main():
    client = AsyncOllamaClient()
    try:
        # List models
        models = await client.list_models()
        print("Available models:", [m.name for m in models])

        # Generate example
        async for response in client.generate("llama2", "What is Python?"):
            print(response.response, end="", flush=True)
        print()

        # Chat example
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        async for response in client.chat("llama2", messages):
            print(response.message.content, end="", flush=True)
        print()

        # Embeddings example
        embeddings = await client.embeddings("llama2", "Hello world")
        print(f"Got {len(embeddings)} dimensional embedding")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())