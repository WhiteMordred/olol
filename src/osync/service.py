import asyncio
import json
import logging
import subprocess
import platform
import psutil
from concurrent import futures
import os
import time

import aiohttp
import grpc
import grpc.aio

# Import protobuf modules safely with fallback
try:
    from . import ollama_pb2, ollama_pb2_grpc
except ImportError:
    # Try relative import - might happen during development
    try:
        import ollama_pb2
        import ollama_pb2_grpc
    except ImportError:
        # On import failure, these will be built dynamically at runtime
        # through the __init__.py mechanism
        pass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaService(ollama_pb2_grpc.OllamaServiceServicer):
    def __init__(self, ollama_host="http://localhost:11434"):
        self.ollama_host = ollama_host
        self.session = None  # Initialize session lazily when needed
        self.loaded_models = set()
        self.active_sessions = {}
        # Get event loop only when in async context - prevents "no running event loop" error
        self._loop = None  # Will be initialized when needed
        
    def _get_session(self):
        """Get or create the aiohttp session."""
        if self.session is None or self.session.closed:
            # Get the loop first to ensure we have one
            loop = self._get_loop()
            self.session = aiohttp.ClientSession(loop=loop)
        return self.session
        
    def _get_loop(self):
        """Get the event loop safely."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop, create a new one
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def RunModel(self, request, context):
        """Simple model run without persistent state - Legacy method"""
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
    
    def Chat(self, request, context):
        """Synchronous implementation of chat"""
        logger.info(f"Chat request for model: {request.model}")
        
        # For sync implementation, use subprocess to call Ollama CLI
        try:
            # Convert messages to format ollama expects
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
                
            # Build the command - supprimer le flag --format
            cmd = ["ollama", "chat", request.model]
            
            # Add options if provided
            if request.options:
                options_str = json.dumps(dict(request.options))
                cmd.extend(["--options", options_str])
                
            # Prepare messages input
            messages_json = json.dumps({"messages": messages})
            
            # Run the command
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Write messages to stdin
            process.stdin.write(messages_json)
            process.stdin.close()
            
            # Stream the responses back to the client
            for line in iter(process.stdout.readline, ''):
                if context.is_active():
                    try:
                        # Essayer de parser comme JSON, mais gérer aussi le texte brut
                        try:
                            response_data = json.loads(line)
                            message = ollama_pb2.Message(
                                role="assistant",
                                content=response_data.get("message", {}).get("content", "")
                            )
                            yield ollama_pb2.ChatResponse(
                                message=message,
                                model=request.model,
                                done=response_data.get("done", False)
                            )
                        except json.JSONDecodeError:
                            # Traiter comme texte brut
                            message = ollama_pb2.Message(
                                role="assistant",
                                content=line.strip()
                            )
                            yield ollama_pb2.ChatResponse(
                                message=message,
                                model=request.model,
                                done=False
                            )
                    except Exception:
                        # Skip problematic lines
                        continue
                else:
                    process.terminate()
                    break
                    
            # Send done message at the end if needed
            if context.is_active():
                yield ollama_pb2.ChatResponse(
                    message=ollama_pb2.Message(role="assistant", content=""),
                    model=request.model,
                    done=True
                )
                    
            # Check for errors
            return_code = process.wait()
            if return_code != 0:
                error_output = process.stderr.read()
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Chat failed: {error_output}")
                
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
    def LegacyChat(self, request, context):
        """Send a message in an existing chat session (Legacy method)"""
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
    
    def List(self, request, context):
        """List available models"""
        try:
            # Exécuter "ollama list" sans l'option --format, qui cause l'erreur
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                error_msg = f"Failed to list models: {result.stderr}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return ollama_pb2.ListResponse(models=[])
            
            # Parser la sortie sous forme de texte (format tabulaire)
            lines = result.stdout.strip().split('\n')
            model_objects = []
            
            # Ignorer la première ligne (en-tête) si elle existe
            if lines and len(lines) > 1:
                for i, line in enumerate(lines):
                    # Sauter l'en-tête (première ligne)
                    if i == 0 and "NAME" in line.upper():
                        continue
                        
                    if not line.strip():
                        continue
                        
                    # Format typique: NAME ID SIZE MODIFIED
                    parts = line.split()
                    if len(parts) >= 1:
                        # Le nom du modèle est la première colonne
                        model_name = parts[0]
                        
                        # Récupérer les autres informations si disponibles
                        model_id = parts[1] if len(parts) > 1 else ""
                        model_size = parts[2] if len(parts) > 2 else ""
                        
                        # Créer un objet Model
                        model_obj = ollama_pb2.Model(
                            name=model_name,
                            model_file=model_id,  # Utiliser ID comme model_file
                            parameter_size=model_size,
                            quantization_level=0,  # Information non disponible dans la sortie
                            template=""  # Information non disponible dans la sortie
                        )
                        model_objects.append(model_obj)
            
            return ollama_pb2.ListResponse(models=model_objects)
                
        except Exception as e:
            error_msg = f"Error listing models: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.ListResponse(models=[])
    
    def Pull(self, request, context):
        """Pull a model from Ollama library"""
        model_name = request.model
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
                    yield ollama_pb2.PullResponse(status=line.strip())
                else:
                    process.terminate()
                    break
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                self.loaded_models.add(model_name)
                yield ollama_pb2.PullResponse(status="Pull completed successfully")
            else:
                error_msg = f"Pull failed with code {return_code}"
                logger.error(error_msg)
                yield ollama_pb2.PullResponse(status=error_msg)
        except Exception as e:
            error_msg = f"Error pulling model: {str(e)}"
            logger.error(error_msg)
            yield ollama_pb2.PullResponse(status=error_msg)
    
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
        # Ensure we have an event loop
        loop = self._get_loop()
        
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
                
            # Essayer de parser le résultat comme JSON
            try:
                model_info = json.loads(stdout)
                return ollama_pb2.ShowResponse(**model_info)
            except json.JSONDecodeError:
                # Si ce n'est pas du JSON, on essaie de convertir le format texte
                output = stdout.decode().strip()
                model_info = {}
                
                # Parsing simple du format texte
                for line in output.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        model_info[key.strip().lower().replace(' ', '_')] = value.strip()
                
                return ollama_pb2.ShowResponse(**model_info)
            
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

    def Generate(self, request, context):
        """Synchronous streaming generate implementation"""
        logger.info(f"Generate request for model: {request.model}")
        
        # For sync implementation, use subprocess to call Ollama CLI
        try:
            cmd = ["ollama", "generate", request.model, request.prompt]
            
            # Add options if provided
            if request.options:
                options_str = json.dumps(dict(request.options))
                cmd.extend(["--options", options_str])
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Stream the responses back to the client
            for line in iter(process.stdout.readline, ''):
                if context.is_active():
                    try:
                        # Essayer de parser comme JSON
                        try:
                            response_data = json.loads(line)
                            yield ollama_pb2.GenerateResponse(
                                model=request.model,
                                response=response_data.get("response", ""),
                                done=response_data.get("done", False),
                                total_duration=response_data.get("total_duration", 0)
                            )
                        except json.JSONDecodeError:
                            # Traiter comme texte brut en cas d'échec
                            yield ollama_pb2.GenerateResponse(
                                model=request.model,
                                response=line.strip(),
                                done=False
                            )
                    except Exception:
                        # Skip problematic lines
                        continue
                else:
                    process.terminate()
                    break
            
            # Envoyer un message final "done" si nécessaire
            if context.is_active():
                yield ollama_pb2.GenerateResponse(
                    model=request.model,
                    response="",
                    done=True
                )
                
            # Check for errors
            return_code = process.wait()
            if return_code != 0:
                error_output = process.stderr.read()
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Generate failed: {error_output}")
                
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
    async def Generate_async(self, request, context):
        """Async streaming generate"""
        # Ensure we have a session and event loop
        session = self._get_session()
        loop = self._get_loop()
        
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
            async with session.post(url, json=payload) as response:
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
        # Ensure we have a session and event loop
        session = self._get_session()
        loop = self._get_loop()
        
        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": True,
            "options": dict(request.options),
            "format": request.format
        }

        try:
            async with session.post(url, json=payload) as response:
                async for line in response.content:
                    if line:
                        response_data = json.loads(line)
                        yield ollama_pb2.ChatResponse(**response_data)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return

    # Already implemented as List(), removing redundant method
    def Embeddings(self, request, context):
        """Get embeddings for text"""
        logger.info(f"Embeddings request for model: {request.model}")
        
        try:
            cmd = ["ollama", "embeddings", request.model, request.prompt]
            
            # Add options if provided
            if request.options:
                options_str = json.dumps(dict(request.options))
                cmd.extend(["--options", options_str])
                
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                error_msg = f"Embeddings failed: {result.stderr}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return ollama_pb2.EmbeddingsResponse()
                
            # Parse the response
            try:
                data = json.loads(result.stdout)
                embeddings = data.get("embedding", [])
                return ollama_pb2.EmbeddingsResponse(embeddings=embeddings)
            except json.JSONDecodeError:
                error_msg = f"Failed to parse embeddings JSON: {result.stdout}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return ollama_pb2.EmbeddingsResponse()
                
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ollama_pb2.EmbeddingsResponse()

    # Add cleanup method
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
        
    def __del__(self):
        """Clean up resources on deletion."""
        # Check if there's a session to close and we have a loop
        if self.session is not None and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
                else:
                    loop.run_until_complete(self.session.close())
            except RuntimeError:
                # If there's no event loop, we can't clean up properly
                # This is a best-effort cleanup
                logger.warning("Could not close aiohttp session: no running event loop")
                pass

    # Add wrapper for sync methods to work with async
    def _run_sync(self, func, *args, **kwargs):
        loop = self._get_loop()
        return loop.run_in_executor(None, func, *args, **kwargs)

    # Method to handle both sync and async calls
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        # Special attributes we don't want to wrap
        if name in ('_get_loop', '_get_session', '__init__', '__dict__', '__class__'):
            return attr
            
        if asyncio.iscoroutinefunction(attr):
            # If it's an async method, wrap it to handle sync calls
            def wrapper(*args, **kwargs):
                try:
                    loop = asyncio.get_running_loop()
                    return attr(*args, **kwargs)
                except RuntimeError:
                    # No running event loop
                    loop = self._get_loop()
                    return loop.run_until_complete(attr(*args, **kwargs))
            return wrapper
        return attr

    def GetCompleteNodeStatus(self, request, context):
        """
        Collecte l'état complet du nœud, incluant les métriques système.
        """
        logger.info("Collecting complete node status")
        
        try:
            # Collecter les informations sur le système
            system_info = self._collect_system_info()
            
            # Collecter les informations sur Ollama
            ollama_info = self._collect_ollama_info()
            
            # Collecter les informations sur les modèles chargés
            models_info = self._collect_models_info()
            
            # Assembler toutes les informations
            status = {
                "system": system_info,
                "ollama": ollama_info,
                "models": models_info,
                "timestamp": time.time()
            }
            
            # Créer et retourner la réponse
            return ollama_pb2.NodeStatusResponse(
                status_json=json.dumps(status),
                healthy=ollama_info.get("healthy", False),
                load=system_info.get("cpu", {}).get("percent", 0)
            )
            
        except Exception as e:
            error_msg = f"Error collecting node status: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.NodeStatusResponse(
                status_json="{}",
                healthy=False,
                load=0
            )
    
    def _collect_system_info(self):
        """Collecte les informations sur le système."""
        # Informations sur le CPU
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "percent": psutil.cpu_percent(interval=0.1),
            "frequency": {
                "current": psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), "current") else None,
                "min": psutil.cpu_freq().min if hasattr(psutil.cpu_freq(), "min") else None,
                "max": psutil.cpu_freq().max if hasattr(psutil.cpu_freq(), "max") else None
            }
        }
        
        # Informations sur la mémoire
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        
        # Informations sur le stockage
        disk = psutil.disk_usage('/')
        disk_info = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
        
        # Informations sur le GPU
        gpu_info = self._collect_gpu_info()
        
        # Assembler toutes les informations système
        return {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info,
            "gpu": gpu_info
        }
    
    def _collect_gpu_info(self):
        """Collecte les informations sur les GPU disponibles."""
        gpu_info = {"detected": False, "devices": []}
        
        try:
            # Essayer de détecter NVIDIA GPU avec nvidia-smi
            nvidia_smi_output = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            
            if nvidia_smi_output.returncode == 0:
                gpu_info["detected"] = True
                gpu_info["type"] = "NVIDIA"
                
                # Parser la sortie de nvidia-smi
                lines = nvidia_smi_output.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.strip().split(', ')
                        if len(parts) >= 6:
                            device = {
                                "name": parts[0],
                                "memory_total_mb": float(parts[1]),
                                "memory_used_mb": float(parts[2]),
                                "memory_free_mb": float(parts[3]),
                                "temperature_c": float(parts[4]),
                                "utilization_percent": float(parts[5])
                            }
                            gpu_info["devices"].append(device)
        except:
            # Si nvidia-smi échoue, essayer avec AMD ROCm
            try:
                rocm_smi_output = subprocess.run(
                    ["rocm-smi", "--showuse", "--json"],
                    capture_output=True, text=True, timeout=5
                )
                
                if rocm_smi_output.returncode == 0:
                    data = json.loads(rocm_smi_output.stdout)
                    gpu_info["detected"] = True
                    gpu_info["type"] = "AMD"
                    
                    for gpu_id, gpu_data in data.items():
                        if isinstance(gpu_data, dict):
                            device = {
                                "name": gpu_data.get("Card name", "Unknown AMD GPU"),
                                "memory_total_mb": gpu_data.get("VRAM Total Memory", 0),
                                "memory_used_mb": gpu_data.get("VRAM Memory Used", 0),
                                "temperature_c": gpu_data.get("Temperature (Sensor edge)", 0),
                                "utilization_percent": gpu_data.get("GPU use (%)", 0)
                            }
                            gpu_info["devices"].append(device)
            except:
                # Si aucun outil GPU n'est disponible
                pass
        
        # Essayer de détecter Metal sur macOS
        if platform.system() == "Darwin" and not gpu_info["detected"]:
            try:
                system_profiler_output = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5
                )
                
                if system_profiler_output.returncode == 0 and "Metal" in system_profiler_output.stdout:
                    gpu_info["detected"] = True
                    gpu_info["type"] = "Apple Silicon"
                    
                    # Extraction des informations basiques sur le GPU
                    lines = system_profiler_output.stdout.strip().split('\n')
                    current_device = {}
                    
                    for line in lines:
                        line = line.strip()
                        if "Chipset Model" in line:
                            if current_device:
                                gpu_info["devices"].append(current_device)
                            current_device = {"name": line.split(": ")[1]}
                        elif "Metal" in line and "supported" in line.lower():
                            current_device["metal_support"] = True
                    
                    if current_device:
                        gpu_info["devices"].append(current_device)
            except:
                # En cas d'échec de détection Metal
                pass
        
        return gpu_info
    
    def _collect_ollama_info(self):
        """Collecte les informations sur l'instance Ollama."""
        ollama_info = {
            "healthy": False,
            "version": "unknown",
            "api_url": self.ollama_host
        }
        
        try:
            # Vérifier si Ollama est en cours d'exécution en appelant l'API
            result = subprocess.run(
                ["curl", "-s", f"{self.ollama_host}/api/version"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0 and result.stdout:
                try:
                    version_data = json.loads(result.stdout)
                    ollama_info["healthy"] = True
                    ollama_info["version"] = version_data.get("version", "unknown")
                except json.JSONDecodeError:
                    pass
            
            # Vérifier le processus Ollama pour des informations supplémentaires
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline']):
                if 'ollama' in proc.info['name'].lower() or any('ollama' in cmd.lower() for cmd in proc.info['cmdline'] if cmd):
                    try:
                        process = psutil.Process(proc.info['pid'])
                        ollama_info["process"] = {
                            "pid": proc.info['pid'],
                            "cpu_percent": process.cpu_percent(interval=0.1),
                            "memory_percent": process.memory_percent(),
                            "memory_info": {
                                "rss": process.memory_info().rss,
                                "vms": process.memory_info().vms
                            },
                            "create_time": process.create_time(),
                            "status": process.status()
                        }
                        break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            logger.error(f"Error collecting Ollama info: {str(e)}")
        
        return ollama_info
    
    def _collect_models_info(self):
        """Collecte des informations détaillées sur les modèles."""
        models_info = []
        
        try:
            # Obtenir la liste des modèles
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                
                # Ignorer l'en-tête
                if len(lines) > 1 and "NAME" in lines[0].upper():
                    lines = lines[1:]
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 1:
                            model_name = parts[0]
                            
                            # Collecter des informations détaillées pour ce modèle
                            try:
                                show_result = subprocess.run(
                                    ["ollama", "show", model_name],
                                    capture_output=True, text=True, timeout=10
                                )
                                
                                model_info = {
                                    "name": model_name,
                                    "id": parts[1] if len(parts) > 1 else "",
                                    "size": parts[2] if len(parts) > 2 else "",
                                }
                                
                                if show_result.returncode == 0:
                                    try:
                                        # Essayer de parser le résultat comme JSON
                                        model_details = json.loads(show_result.stdout)
                                        model_info.update(model_details)
                                    except json.JSONDecodeError:
                                        # Parser le format texte si ce n'est pas du JSON
                                        for detail_line in show_result.stdout.split('\n'):
                                            if ':' in detail_line:
                                                key, value = detail_line.split(':', 1)
                                                model_info[key.strip().lower().replace(' ', '_')] = value.strip()
                                
                                models_info.append(model_info)
                            except Exception as e:
                                logger.error(f"Error collecting details for model {model_name}: {str(e)}")
                                models_info.append({
                                    "name": model_name,
                                    "error": str(e)
                                })
        except Exception as e:
            logger.error(f"Error collecting models information: {str(e)}")
        
        return models_info
    
    def RemoteModelCommand(self, request, context):
        """
        Exécute une commande sur un modèle à distance (pull, push, delete).
        """
        command = request.command
        model_name = request.model_name
        options = json.loads(request.options_json) if request.options_json else {}
        
        logger.info(f"Remote model command: {command} on model {model_name}")
        
        try:
            result = {
                "success": False,
                "message": "",
                "command": command,
                "model": model_name
            }
            
            if command == "pull":
                # Exécuter la commande pull
                process = subprocess.Popen(
                    ["ollama", "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output = []
                for line in iter(process.stdout.readline, ''):
                    output.append(line.strip())
                
                return_code = process.wait()
                result["success"] = return_code == 0
                result["output"] = output
                
                if return_code == 0:
                    self.loaded_models.add(model_name)
                    result["message"] = "Model pulled successfully"
                else:
                    result["message"] = "Failed to pull model"
                
            elif command == "push":
                # Exécuter la commande push
                process = subprocess.Popen(
                    ["ollama", "push", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output = []
                for line in iter(process.stdout.readline, ''):
                    output.append(line.strip())
                
                return_code = process.wait()
                result["success"] = return_code == 0
                result["output"] = output
                
                if return_code == 0:
                    result["message"] = "Model pushed successfully"
                else:
                    result["message"] = "Failed to push model"
                
            elif command == "delete":
                # Exécuter la commande delete
                process = subprocess.Popen(
                    ["ollama", "rm", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output = []
                for line in iter(process.stdout.readline, ''):
                    output.append(line.strip())
                
                return_code = process.wait()
                result["success"] = return_code == 0
                result["output"] = output
                
                if return_code == 0:
                    if model_name in self.loaded_models:
                        self.loaded_models.remove(model_name)
                    result["message"] = "Model deleted successfully"
                else:
                    result["message"] = "Failed to delete model"
                
            elif command == "create":
                # Exécuter la commande create
                modelfile_content = options.get("modelfile_content", "")
                modelfile_path = f"/tmp/modelfile_{int(time.time())}"
                
                # Écrire le contenu du Modelfile dans un fichier temporaire
                with open(modelfile_path, 'w') as f:
                    f.write(modelfile_content)
                
                # Créer le modèle avec le fichier temporaire
                process = subprocess.Popen(
                    ["ollama", "create", model_name, "-f", modelfile_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output = []
                for line in iter(process.stdout.readline, ''):
                    output.append(line.strip())
                
                return_code = process.wait()
                
                # Supprimer le fichier temporaire
                try:
                    os.remove(modelfile_path)
                except:
                    pass
                
                result["success"] = return_code == 0
                result["output"] = output
                
                if return_code == 0:
                    self.loaded_models.add(model_name)
                    result["message"] = "Model created successfully"
                else:
                    result["message"] = "Failed to create model"
                
            else:
                result["message"] = f"Unsupported command: {command}"
            
            return ollama_pb2.RemoteCommandResponse(result_json=json.dumps(result))
            
        except Exception as e:
            error_msg = f"Error executing remote command: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return ollama_pb2.RemoteCommandResponse(
                result_json=json.dumps({
                    "success": False,
                    "message": error_msg,
                    "command": command,
                    "model": model_name
                })
            )

# Update server to handle both sync and async
def serve():
    # Create sync server only (simplify for now to avoid event loop issues)
    sync_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )

    # Add service to server
    service = OllamaService()
    ollama_pb2_grpc.add_OllamaServiceServicer_to_server(service, sync_server)
    
    # Add health service for monitoring
    # Note: This is a simple implementation without full gRPC health service
    
    # Start server
    sync_server.add_insecure_port("[::]:50051")
    sync_server.start()
    logger.info("Ollama gRPC server started on port 50051")

    # Run server
    try:
        sync_server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        sync_server.stop(0)

if __name__ == "__main__":
    serve()
