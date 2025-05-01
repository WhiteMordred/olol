"""Synchronous client implementation for Ollama service."""

import logging
from typing import Dict, Iterator, List, Optional

import grpc

from ..proto import ollama_pb2, ollama_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Synchronous client for interacting with Ollama service."""
    
    def __init__(self, host: str = 'localhost', port: int = 50051, timeout: float = 5.0) -> None:
        """Initialize the Ollama client.
        
        Args:
            host: Server hostname or IP
            port: Server port number
            timeout: Connection timeout in seconds
        """
        # Si host contient déjà le port (format: "host:port"), on l'utilise directement
        if ":" in host and not host.startswith("["):  # IPv4 with port or hostname with port
            self.channel = grpc.insecure_channel(host, options=[
                ('grpc.enable_http_proxy', 0),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000)
            ])
            # Extraire le host et le port pour les stocker
            self.host, port_str = host.rsplit(":", 1)
            self.port = int(port_str)
        else:
            # Autrement on construit l'adresse avec les paramètres individuels
            self.channel = grpc.insecure_channel(f"{host}:{port}", options=[
                ('grpc.enable_http_proxy', 0),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000)
            ])
            self.host = host
            self.port = port
            
        # Configurer le timeout pour les appels RPC
        self.timeout = timeout
        self.stub = ollama_pb2_grpc.OllamaServiceStub(self.channel)
        
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
            Iterator of responses
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        try:
            # Utiliser directement RunModel car c'est ce que le serveur Ollama supporte
            request = ollama_pb2.ModelRequest(
                model_name=model,
                prompt=prompt,
                parameters=options or {},
                stream=stream
            )
            
            if stream:
                # Utiliser StreamingRunModel si disponible et streaming demandé
                try:
                    responses = self.stub.StreamingRunModel(request)
                    accumulated_response = ""
                    
                    # Traiter chaque réponse du stream et accumuler la réponse totale
                    for response in responses:
                        chunk = response.output if hasattr(response, 'output') else ""
                        accumulated_response += chunk
                        
                        yield ollama_pb2.GenerateResponse(
                            model=model,
                            response=chunk,  # Envoyer seulement le nouveau morceau
                            done=response.done if hasattr(response, 'done') else False,
                            total_duration=int(response.completion_time * 1000) if hasattr(response, 'completion_time') else 0
                        )
                        
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                        # Fallback à RunModel avec simulation de streaming
                        logger.warning("StreamingRunModel non disponible, simulation du streaming avec RunModel")
                        response = self.stub.RunModel(request)
                        
                        # Extraire la réponse complète
                        output = response.output if hasattr(response, 'output') else str(response)
                        
                        # Simuler le streaming en fragmentant la réponse en plus petits morceaux
                        # pour une meilleure expérience utilisateur
                        chunks = [output[i:i+5] for i in range(0, len(output), 5)]
                        
                        for i, chunk in enumerate(chunks):
                            is_last = i == len(chunks) - 1
                            yield ollama_pb2.GenerateResponse(
                                model=model,
                                response=chunk,
                                done=is_last,
                                total_duration=int(response.completion_time * 1000) if hasattr(response, 'completion_time') and is_last else 0
                            )
                    else:
                        raise
            else:
                # Mode non-streaming
                response = self.stub.RunModel(request)
                
                # Extraire la réponse
                output = response.output if hasattr(response, 'output') else str(response)
                
                yield ollama_pb2.GenerateResponse(
                    model=model,
                    response=output,
                    done=True,
                    total_duration=int(response.completion_time * 1000) if hasattr(response, 'completion_time') else 0
                )
                
        except Exception as e:
            logger.error(f"Generate error: {str(e)}")
            raise
    
    def list_models(self) -> List[Dict[str, any]]:
        """Liste tous les modèles disponibles sur ce serveur Ollama.
        
        Returns:
            Liste de dictionnaires contenant les informations des modèles
        
        Raises:
            grpc.RpcError: Si l'appel gRPC échoue
        """
        try:
            # Essayer d'abord avec la nouvelle API List
            try:
                request = ollama_pb2.ListRequest()
                response = self.stub.List(request, timeout=self.timeout)
                
                # Vérifier si la réponse est itérable comme prévu
                if hasattr(response, 'models') and hasattr(response.models, '__iter__'):
                    models_list = []
                    for model in response.models:
                        model_info = {
                            "name": model.name,
                            "size": getattr(model, 'size', 0),
                            "modified_at": getattr(model, 'modified_at', ""),
                            "digest": getattr(model, 'digest', None)
                        }
                        models_list.append(model_info)
                    return models_list
                else:
                    # Si la réponse a un format différent mais est une ListResponse
                    # (API évoluée mais pas standard)
                    logger.debug(f"Format de réponse non standard pour List: {type(response)}")
                    models_list = []
                    if hasattr(response, 'model'):
                        # Format potentiel avec un seul modèle
                        models_list.append({"name": response.model})
                    return models_list
            except (AttributeError, grpc.RpcError) as e:
                # Si le stub n'a pas la méthode List, on passe à l'alternative
                logger.debug(f"Server does not support List API, using fallback: {str(e)}")
                
            # Si List n'existe pas ou échoue, essayer avec l'API Legacy GetModels
            try:
                request = ollama_pb2.GetModelsRequest()
                response = self.stub.GetModels(request, timeout=self.timeout)
                
                if hasattr(response, 'models'):
                    models_list = []
                    for model in response.models:
                        model_info = {
                            "name": model.name,
                            "size": getattr(model, 'size', 0),
                            "modified_at": getattr(model, 'modified_at', ""),
                            "digest": getattr(model, 'digest', None)
                        }
                        models_list.append(model_info)
                    return models_list
                else:
                    return []
            except (AttributeError, grpc.RpcError) as e:
                logger.debug(f"Server does not support GetModels API: {str(e)}")
                
            # En dernier recours, essayer de demander la liste des modèles avec info
            try:
                # Cette approche utilise l'API générique pour obtenir des informations
                request = ollama_pb2.InfoRequest()
                response = self.stub.Info(request, timeout=self.timeout)
                
                if hasattr(response, 'models'):
                    if isinstance(response.models, list) or hasattr(response.models, '__iter__'):
                        return [{"name": model_name} for model_name in response.models]
                    elif isinstance(response.models, str):
                        # Si models est une chaîne unique, la convertir en liste d'un élément
                        return [{"name": response.models}]
                    else:
                        logger.warning(f"Unexpected models type: {type(response.models)}")
                        return []
                else:
                    return []
            except (AttributeError, grpc.RpcError):
                logger.debug("Server does not support Info API for model listing")
                return []
                
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            # En cas d'erreur, retourner une liste vide plutôt que de planter
            return []
    
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
            Iterator of responses
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        try:
            # Formatage du prompt à partir de l'historique des messages
            system_message = ""
            conversation_history = ""
            latest_user_message = ""
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'system':
                    system_message = content
                elif role == 'user':
                    latest_user_message = content
                    conversation_history += f"User: {content}\n"
                elif role == 'assistant':
                    conversation_history += f"Assistant: {content}\n"
            
            # Construire le prompt complet
            full_prompt = ""
            if system_message:
                full_prompt = f"{system_message}\n\n{conversation_history}"
            else:
                full_prompt = conversation_history
                
            if not latest_user_message:
                full_prompt += "User: "  # Inciter une réponse si pas de message utilisateur
            
            # Créer une requête RunModel
            request = ollama_pb2.ModelRequest(
                model_name=model,
                prompt=full_prompt,
                parameters=options or {},
                stream=stream
            )
            
            if stream:
                # Utiliser StreamingRunModel si disponible et streaming demandé
                try:
                    # Tenter d'utiliser l'API de streaming
                    responses = self.stub.StreamingRunModel(request)
                    
                    # Créer un itérateur qui transforme les réponses en ChatResponse
                    for response in responses:
                        response_text = response.output if hasattr(response, 'output') else ""
                        
                        # Créer un message assistant pour chaque morceau
                        assistant_message = {
                            "role": "assistant",
                            "content": response_text
                        }
                        
                        chat_response = ollama_pb2.ChatResponse()
                        chat_response.model = model
                        chat_response.done = response.done if hasattr(response, 'done') else False
                        
                        # Ajouter la durée si disponible
                        if hasattr(response, 'completion_time'):
                            chat_response.total_duration = int(response.completion_time * 1000)
                            
                        # Utiliser setattr pour le message car la structure peut varier
                        # selon la version de protobuf
                        try:
                            chat_response.message.role = "assistant"
                            chat_response.message.content = response_text
                        except AttributeError:
                            # Alternative en cas d'erreur d'attribut
                            setattr(chat_response, 'message', assistant_message)
                            
                        yield chat_response
                
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                        # Fallback à RunModel avec simulation de streaming
                        logger.warning("StreamingRunModel non disponible, simulation du streaming avec RunModel")
                        response = self.stub.RunModel(request)
                        
                        # Extraire la réponse
                        output = response.output if hasattr(response, 'output') else str(response)
                        
                        # Simuler le streaming en fragmentant la réponse en plus petits morceaux
                        # pour une meilleure expérience utilisateur
                        chunks = [output[i:i+5] for i in range(0, len(output), 5)]
                        
                        for i, chunk in enumerate(chunks):
                            is_last = i == len(chunks) - 1
                            
                            chat_response = ollama_pb2.ChatResponse()
                            chat_response.model = model
                            chat_response.done = is_last
                            
                            # Ajouter la durée si disponible et si c'est le dernier message
                            if hasattr(response, 'completion_time') and is_last:
                                chat_response.total_duration = int(response.completion_time * 1000)
                                
                            # Utiliser setattr pour le message
                            try:
                                chat_response.message.role = "assistant"
                                chat_response.message.content = chunk
                            except AttributeError:
                                # Alternative
                                setattr(chat_response, 'message', {
                                    "role": "assistant",
                                    "content": chunk
                                })
                            
                            yield chat_response
                    else:
                        raise
            
            else:
                # Mode non-streaming
                response = self.stub.RunModel(request)
                
                # Extraire la réponse
                output = response.output if hasattr(response, 'output') else str(response)
                
                chat_response = ollama_pb2.ChatResponse()
                chat_response.model = model
                chat_response.done = True
                
                # Ajouter la durée si disponible
                if hasattr(response, 'completion_time'):
                    chat_response.total_duration = int(response.completion_time * 1000)
                    
                # Utiliser setattr pour le message
                try:
                    chat_response.message.role = "assistant"
                    chat_response.message.content = output
                except AttributeError:
                    # Alternative
                    setattr(chat_response, 'message', {
                        "role": "assistant",
                        "content": output
                    })
                
                yield chat_response
                
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
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
            # Pour les embeddings, nous n'avons pas de fallback simple avec RunModel
            # Alors nous retournons juste un vecteur vide
            logger.warning("Embeddings not available in this version of Ollama server")
            return ollama_pb2.EmbeddingsResponse(
                embeddings=[]
            )
        except Exception as e:
            logger.error(f"Embeddings error: {str(e)}")
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
    
    def check_health(self, timeout: float = None) -> bool:
        """Check if the server is healthy.
        
        Uses dedicated health check API if available, with fallback to listing models.
        
        Args:
            timeout: Optional timeout for the health check call (in seconds)
            
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Use the provided timeout if any, otherwise use the client's default
            opts = ()
            if timeout is not None:
                opts = [('grpc.timeout', int(timeout * 1000))]
            
            # First try dedicated health check API
            request = ollama_pb2.HealthCheckRequest(detail=True)
            try:
                response = self.stub.HealthCheck(request, timeout=timeout)
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
        for model in models:
            print(f"- {model['name']} ({model.get('parameter_size', 'unknown')}, {model.get('quantization_level', 'unknown')})")
        
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