"""Asynchronous server implementation for Ollama service."""

import asyncio
import json
import logging
from typing import AsyncIterator

import aiohttp
import grpclib
from grpclib.server import Server

from ..proto import ollama_pb2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    from ..ollama_pb2_grpc import AsyncOllamaServiceServicer as AsyncOllamaServiceBase
except (ImportError, AttributeError):
    # Fallback if not generated yet
    class AsyncOllamaServiceBase:
        pass

class AsyncOllamaService(AsyncOllamaServiceBase):
    """Asynchronous implementation of the Ollama gRPC service."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434") -> None:
        """Initialize the async Ollama service.
        
        Args:
            ollama_host: URL to the Ollama HTTP API
        """
        self.ollama_host = ollama_host
        self._session = None  # Will be initialized on demand
        
    def _get_session(self):
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            self._session = aiohttp.ClientSession(loop=loop)
        return self._session
        
    async def cleanup(self):
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        
    def __del__(self):
        """Clean up resources on deletion."""
        if self._session is not None and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except RuntimeError:
                # If there's no event loop, we can't clean up properly
                logger.warning("Could not close aiohttp session: no running event loop")
                pass
    
    async def Generate(self, stream: AsyncIterator[ollama_pb2.GenerateRequest]) -> AsyncIterator[ollama_pb2.GenerateResponse]:
        """Generate text from a model.
        
        Args:
            stream: Stream of GenerateRequest messages
            
        Yields:
            Stream of GenerateResponse messages
        """
        # Get session
        session = self._get_session()
        
        request = await stream.__anext__()
        url = f"{self.ollama_host}/api/generate"
        
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": True,
            "options": dict(request.options),
            "context": list(request.context),
            "template": request.template,
            "format": request.format
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str:
                            try:
                                data = json.loads(line_str)
                                yield ollama_pb2.GenerateResponse(
                                    model=data.get('model', ''),
                                    created_at=data.get('created_at', ''),
                                    response=data.get('response', ''),
                                    done=data.get('done', False),
                                    context=data.get('context', []),
                                    total_duration=data.get('total_duration', 0),
                                    eval_count=str(data.get('eval_count', '')),
                                    eval_duration=str(data.get('eval_duration', ''))
                                )
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse Ollama response: {line_str}")
        except Exception as e:
            logger.error(f"Error in Generate: {str(e)}")
            raise grpclib.GRPCError(grpclib.Status.INTERNAL, str(e))
    
    async def Chat(self, stream: AsyncIterator[ollama_pb2.ChatRequest]) -> AsyncIterator[ollama_pb2.ChatResponse]:
        """Chat with a model.
        
        Args:
            stream: Stream of ChatRequest messages
            
        Yields:
            Stream of ChatResponse messages
        """
        # Get session
        session = self._get_session()
        
        request = await stream.__anext__()
        url = f"{self.ollama_host}/api/chat"
        
        # Convert gRPC Message objects to dict format expected by Ollama API
        messages = []
        for msg in request.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
                "images": [img.decode('utf-8') for img in msg.images] if msg.images else []
            })
        
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "options": dict(request.options),
            "format": request.format
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str:
                            try:
                                data = json.loads(line_str)
                                msg_data = data.get('message', {})
                                message = ollama_pb2.Message(
                                    role=msg_data.get('role', ''),
                                    content=msg_data.get('content', '')
                                )
                                yield ollama_pb2.ChatResponse(
                                    message=message,
                                    model=data.get('model', ''),
                                    created_at=data.get('created_at', ''),
                                    done=data.get('done', False),
                                    total_duration=data.get('total_duration', 0),
                                    load_duration=data.get('load_duration', ''),
                                    prompt_eval_duration=data.get('prompt_eval_duration', ''),
                                    eval_count=data.get('eval_count', ''),
                                    eval_duration=data.get('eval_duration', '')
                                )
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse Ollama response: {line_str}")
        except Exception as e:
            logger.error(f"Error in Chat: {str(e)}")
            raise grpclib.GRPCError(grpclib.Status.INTERNAL, str(e))
    
    async def Embeddings(self, request: ollama_pb2.EmbeddingsRequest) -> ollama_pb2.EmbeddingsResponse:
        """Get embeddings for a prompt.
        
        Args:
            request: EmbeddingsRequest message
            
        Returns:
            EmbeddingsResponse message with vector data
        """
        # Get session
        session = self._get_session()
        
        url = f"{self.ollama_host}/api/embeddings"
        
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "options": dict(request.options)
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return ollama_pb2.EmbeddingsResponse(
                    embeddings=data.get('embedding', [])
                )
        except Exception as e:
            logger.error(f"Error in Embeddings: {str(e)}")
            raise grpclib.GRPCError(grpclib.Status.INTERNAL, str(e))
    
    async def List(self, request: ollama_pb2.ListRequest) -> ollama_pb2.ListResponse:
        """List available models.
        
        Args:
            request: ListRequest message
            
        Returns:
            ListResponse message with model list
        """
        # Get session
        session = self._get_session()
        
        url = f"{self.ollama_host}/api/tags"
        
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                models = []
                
                for model_data in data.get('models', []):
                    model_files = []
                    for file_data in model_data.get('files', []):
                        model_file = ollama_pb2.ModelFile(
                            name=file_data.get('name', ''),
                            type=file_data.get('type', ''),
                            size=file_data.get('size', ''),
                            digest=file_data.get('digest', '')
                        )
                        model_files.append(model_file)
                    
                    model = ollama_pb2.Model(
                        name=model_data.get('name', ''),
                        model_file=model_data.get('model_file', ''),
                        parameter_size=model_data.get('parameter_size', ''),
                        quantization_level=model_data.get('quantization_level', 0),
                        files=model_files
                    )
                    models.append(model)
                
                return ollama_pb2.ListResponse(models=models)
        except Exception as e:
            logger.error(f"Error in List: {str(e)}")
            raise grpclib.GRPCError(grpclib.Status.INTERNAL, str(e))
    
    async def close(self) -> None:
        """Close resources when the server is shutting down."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


async def start_server(host: str = "0.0.0.0", port: int = 50052) -> Server:
    """Start the gRPC server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        
    Returns:
        Server instance
    """
    service = AsyncOllamaService()
    server = Server([service])
    await server.start(host, port)
    logger.info(f"Async Ollama gRPC server started on {host}:{port}")
    return server


async def serve() -> None:
    """Main entry point for the async server."""
    service = AsyncOllamaService()
    server = await start_server()
    
    # Listen for keyboard interrupt to stop the server
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Stopping async server...")
        # Cleanup service resources first
        await service.cleanup()
        # Then close the server
        server.close()
        await server.wait_closed()
        logger.info("Server stopped.")


if __name__ == "__main__":
    asyncio.run(serve())