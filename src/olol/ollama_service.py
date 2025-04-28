import asyncio
import json
import logging
import subprocess
from concurrent import futures

import aiohttp
import grpc
import grpc.aio
import ollama_pb2
import ollama_pb2_grpc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaService(ollama_pb2_grpc.OllamaServiceServicer):
    def __init__(self, ollama_host="http://localhost:11434"):
        self.ollama_host = ollama_host
        self.session = aiohttp.ClientSession()
        self.loaded_models = set()
        self.active_sessions = {}
        self._loop = asyncio.get_event_loop()
    
    def RunModel(self, request, context):
        """Simple model run without persistent state"""
        logger.info(f"Running model: {request.model_name}")
        try:
            result = subprocess.run(
                ["ollama", "run", request.model_name, request.prompt],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                error_msg = f"Ollama execution failed: {result.stderr}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return ollama_pb2.ModelResponse()
            
            return ollama_pb2.ModelResponse(output=result.stdout)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.ModelResponse()
    
    def CreateSession(self, request, context):
        """Create a new chat session"""
        session_id = request.session_id
        model_name = request.model_name
        
        logger.info(f"Creating session {session_id} with model {model_name}")
        
        if session_id in self.active_sessions:
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            context.set_details(f"Session {session_id} already exists")
            return ollama_pb2.SessionResponse(success=False)
        
        # Ensure model is downloaded
        try:
            if model_name not in self.loaded_models:
                logger.info(f"Pulling model {model_name}")
                subprocess.run(["ollama", "pull", model_name], check=True)
                self.loaded_models.add(model_name)
            
            # Initialize session
            self.active_sessions[session_id] = {
                "model": model_name,
                "messages": []
            }
            
            return ollama_pb2.SessionResponse(success=True)
        except Exception as e:
            error_msg = f"Failed to create session: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.SessionResponse(success=False)
    
    def ChatMessage(self, request, context):
        """Send a message in an existing chat session"""
        session_id = request.session_id
        message = request.message
        
        logger.info(f"Processing message for session {session_id}")
        
        if session_id not in self.active_sessions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {session_id} not found")
            return ollama_pb2.ModelResponse()
        
        session = self.active_sessions[session_id]
        model_name = session["model"]
        
        # Add message to session history
        session["messages"].append({"role": "user", "content": message})
        
        try:
            # Format the chat history for ollama
            history_arg = json.dumps(session["messages"])
            
            # Call ollama with the chat history
            result = subprocess.run(
                ["ollama", "run", model_name, "--format", "json", "--options", 
                 f'{{"messages": {history_arg}}}'],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                error_msg = f"Ollama chat failed: {result.stderr}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return ollama_pb2.ModelResponse()
            
            # Parse the response
            response_text = result.stdout.strip()
            
            # Add assistant response to session history
            session["messages"].append({"role": "assistant", "content": response_text})
            
            return ollama_pb2.ModelResponse(output=response_text)
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.ModelResponse()
    
    def ListModels(self, request, context):
        """List available models"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                error_msg = f"Failed to list models: {result.stderr}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return ollama_pb2.ModelsResponse(models="[]")
            
            # Essayer de parser le résultat comme JSON (nouvelles versions d'Ollama)
            try:
                json.loads(result.stdout)
                # Si c'est un JSON valide, retourner directement
                return ollama_pb2.ModelsResponse(models=result.stdout)
            except json.JSONDecodeError:
                # Format texte, le convertir en JSON
                lines = result.stdout.strip().split('\n')
                models = []
                
                # Ignorer la première ligne (en-tête) si elle existe
                if lines and len(lines) > 1:
                    for line in lines[1:]:  # Sauter l'en-tête
                        if not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            model_name = parts[0]
                            tag = parts[1]
                            model_info = {
                                "name": model_name,
                                "tag": tag,
                                "size": parts[3] if len(parts) > 3 else "",
                                "modified": " ".join(parts[4:]) if len(parts) > 4 else ""
                            }
                            models.append(model_info)
                
                # Convertir en JSON
                models_json = json.dumps({"models": models})
                return ollama_pb2.ModelsResponse(models=models_json)
        except Exception as e:
            error_msg = f"Error listing models: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.ModelsResponse(models="[]")
    
    def PullModel(self, request, context):
        """Pull a model from Ollama library"""
        model_name = request.model_name
        logger.info(f"Pulling model {model_name}")
        
        try:
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream the progress back to the client
            for line in iter(process.stdout.readline, ''):
                if context.is_active():
                    yield ollama_pb2.PullResponse(progress=line.strip())
                else:
                    process.terminate()
                    break
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                self.loaded_models.add(model_name)
                yield ollama_pb2.PullResponse(progress="Pull completed successfully")
            else:
                error_msg = f"Pull failed with code {return_code}"
                logger.error(error_msg)
                yield ollama_pb2.PullResponse(progress=error_msg)
        except Exception as e:
            error_msg = f"Error pulling model: {str(e)}"
            logger.error(error_msg)
            yield ollama_pb2.PullResponse(progress=error_msg)
    
    def DeleteSession(self, request, context):
        """Delete an existing chat session"""
        session_id = request.session_id
        logger.info(f"Deleting session {session_id}")
        
        if session_id not in self.active_sessions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {session_id} not found")
            return ollama_pb2.SessionResponse(success=False)
        
        try:
            del self.active_sessions[session_id]
            return ollama_pb2.SessionResponse(success=True)
        except Exception as e:
            error_msg = f"Error deleting session: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.SessionResponse(success=False)

    async def Push(self, request, context):
        """Push a model to registry"""
        cmd = ["ollama", "push", request.model]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield ollama_pb2.PushResponse(status=line.decode().strip())
                
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return

    async def Create(self, request, context):
        """Create a model"""
        try:
            with open(request.modelfile, 'w') as f:
                f.write(request.modelfile_content)
            
            process = await asyncio.create_subprocess_exec(
                "ollama", "create", request.model, "-f", request.modelfile,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(stderr.decode())
                
            return ollama_pb2.CreateResponse(success=True)
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ollama_pb2.CreateResponse(success=False)

    async def Copy(self, request, context):
        """Copy a model"""
        cmd = ["ollama", "cp", request.source, request.destination]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(stderr.decode())
                
            return ollama_pb2.CopyResponse(success=True)
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ollama_pb2.CopyResponse(success=False)

    async def Show(self, request, context):
        """Show model details"""
        # Suppression du flag --format qui n'est plus supporté
        cmd = ["ollama", "show", request.model]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(stderr.decode())
            
            # Tenter de décoder la sortie comme JSON
            try:
                model_info = json.loads(stdout)
                return ollama_pb2.ShowResponse(**model_info)
            except json.JSONDecodeError:
                # Si la sortie n'est pas un JSON, la convertir en format structuré
                output = stdout.decode().strip()
                model_info = {}
                
                # Parsing simple du format texte
                current_section = "general"
                model_info[current_section] = {}
                
                for line in output.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Vérifier s'il s'agit d'un en-tête de section
                    if line.endswith(':') and not ': ' in line:
                        current_section = line[:-1].lower().replace(' ', '_')
                        model_info[current_section] = {}
                        continue
                        
                    if ':' in line:
                        key, value = line.split(':', 1)
                        model_info[current_section][key.strip().lower().replace(' ', '_')] = value.strip()
                
                # Créer une structure aplatie pour ShowResponse
                flat_info = {}
                for section, items in model_info.items():
                    if isinstance(items, dict):
                        for k, v in items.items():
                            flat_info[k] = v
                    else:
                        flat_info[section] = items
                        
                return ollama_pb2.ShowResponse(**flat_info)
                
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ollama_pb2.ShowResponse()

    async def Delete(self, request, context):
        """Delete a model"""
        cmd = ["ollama", "rm", request.model]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(stderr.decode())
                
            return ollama_pb2.DeleteResponse(success=True)
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ollama_pb2.DeleteResponse(success=False)

    async def Generate(self, request, context):
        """Async streaming generate"""
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": True,
            "options": dict(request.options),
            "context": request.context,
            "template": request.template,
            "format": request.format
        }

        try:
            async with self.session.post(url, json=payload) as response:
                async for line in response.content:
                    if line:
                        response_data = json.loads(line)
                        yield ollama_pb2.GenerateResponse(**response_data)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return

    async def Chat(self, request, context):
        """Async streaming chat"""
        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": True,
            "options": dict(request.options),
            "format": request.format
        }

        try:
            async with self.session.post(url, json=payload) as response:
                async for line in response.content:
                    if line:
                        response_data = json.loads(line)
                        yield ollama_pb2.ChatResponse(**response_data)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return

    async def GetModels(self, request, context):
        """Get list of available models"""
        url = f"{self.ollama_host}/api/tags"
        
        try:
            async with self.session.get(url) as response:
                data = await response.json()
                return ollama_pb2.GetModelsResponse(models=data)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ollama_pb2.GetModelsResponse()

    def LegacyChat(self, request, context):
        try:
            # Validate request
            model_name = request.model if hasattr(request, "model") else ""
            session_id = request.session_id if hasattr(request, "session_id") else ""
            
            logger.info(f"Legacy chat request for model {model_name}, session {session_id}")
            
            # Format chat history for ollama if needed
            history_arg = json.dumps(request.messages) if hasattr(request, "messages") else "{}"
            
            # Supprimer le flag --format json qui n'est plus supporté
            result = subprocess.run(
                ["ollama", "run", model_name, "--options", 
                 f'{{"messages": {history_arg}}}'],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                error_msg = f"Ollama chat failed: {result.stderr}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return ollama_pb2.ModelResponse()
            
            # Parse the response
            response_text = result.stdout.strip()
            
            # Try to parse as JSON if it looks like JSON
            if response_text.startswith("{") and response_text.endswith("}"):
                try:
                    data = json.loads(response_text)
                    response_text = data.get("response", response_text)
                except json.JSONDecodeError:
                    # Not valid JSON, use as is
                    pass
                    
            return ollama_pb2.ModelResponse(output=response_text)
        except Exception as e:
            error_msg = f"Error in legacy chat: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.ModelResponse()

    # Add cleanup method
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()

    # Add wrapper for sync methods to work with async
    def _run_sync(self, func, *args, **kwargs):
        return self._loop.run_in_executor(None, func, *args, **kwargs)

    # Method to handle both sync and async calls
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if asyncio.iscoroutinefunction(attr):
            # If it's an async method, wrap it to handle sync calls
            def wrapper(*args, **kwargs):
                if asyncio.get_event_loop().is_running():
                    return attr(*args, **kwargs)
                return self._run_sync(attr, *args, **kwargs)
            return wrapper
        return attr

# Update server to handle both sync and async
def serve():
    # Create both sync and async servers
    sync_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )
    
    async_server = grpc.aio.server(
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )

    # Add service to both servers
    service = OllamaService()
    ollama_pb2_grpc.add_OllamaServiceServicer_to_server(service, sync_server)
    ollama_pb2_grpc.add_OllamaServiceServicer_to_server(service, async_server)

    # Start servers on different ports
    sync_server.add_insecure_port("[::]:50051")
    async_server.add_insecure_port("[::]:50052")
    
    sync_server.start()
    logger.info("Sync Ollama gRPC server started on port 50051")

    async def run_async_server():
        await async_server.start()
        logger.info("Async Ollama gRPC server started on port 50052")
        await async_server.wait_for_termination()

    # Run both servers
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(run_async_server())
        sync_server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Servers shutting down...")
        sync_server.stop(0)
        loop.run_until_complete(async_server.stop(0))

if __name__ == "__main__":
    serve()
