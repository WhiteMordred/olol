"""
Services pour l'API du proxy Ollama.

Ce module contient les services métier pour interagir avec les serveurs Ollama
à travers le gestionnaire de cluster. Il sépare la logique métier des routes HTTP.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Generator, Tuple

from flask import Response, stream_with_context
from olol.sync.client import OllamaClient

from .models import (
    GenerateRequest, GenerateResponse, 
    ChatRequest, ChatResponse, ChatMessage,
    EmbeddingsRequest, EmbeddingsResponse,
    ModelInfo, ServerInfo, StatusResponse, ModelsResponse, ServersResponse
)

# Configuration du logging
logger = logging.getLogger(__name__)


class OllamaProxyService:
    """Service pour interagir avec les serveurs Ollama via le gestionnaire de cluster."""
    
    def __init__(self, cluster_manager):
        """
        Initialise le service avec un gestionnaire de cluster.
        
        Args:
            cluster_manager: Le gestionnaire de cluster Ollama
        """
        self.cluster_manager = cluster_manager
    
    def get_client_for_model(self, model_name: str) -> Tuple[Optional[OllamaClient], str, int]:
        """
        Obtient un client pour un modèle spécifique.
        
        Args:
            model_name: Le nom du modèle
            
        Returns:
            Un tuple (client, host, port) ou (None, "", 0) si aucun serveur n'est disponible
        """
        server_address = self.cluster_manager.get_best_server_for_model(model_name)
        if not server_address:
            return None, "", 0
            
        try:
            if server_address.count(':') == 1:
                host, port_str = server_address.split(':')
                port = int(port_str)
                client = OllamaClient(host=host, port=port)
                return client, host, port
        except Exception as e:
            logger.error(f"Erreur lors de la création du client pour {server_address}: {e}")
            
        return None, "", 0
    
    def generate(self, request: GenerateRequest) -> Dict[str, Any]:
        """
        Génère du texte en utilisant un modèle.
        
        Args:
            request: La requête de génération
            
        Returns:
            Un dictionnaire contenant la réponse
        """
        client, host, port = self.get_client_for_model(request.model)
        if not client:
            return {
                "error": "Aucun serveur disponible pour ce modèle",
                "model": request.model,
                "done": True
            }
            
        try:
            if not request.stream:
                # Génération non-streaming
                response_text = ""
                final_response = None
                
                for resp in client.generate(request.model, request.prompt, False, request.options):
                    final_response = resp
                    if hasattr(resp, 'response'):
                        response_text += resp.response
                
                # Créer la réponse finale
                response = {
                    "model": request.model,
                    "response": response_text,
                    "done": True
                }
                
                # Ajouter les métriques si disponibles
                if final_response:
                    if hasattr(final_response, 'total_duration'):
                        response["total_duration"] = final_response.total_duration
                    if hasattr(final_response, 'load_duration'):
                        response["load_duration"] = final_response.load_duration
                    if hasattr(final_response, 'prompt_eval_count'):
                        response["prompt_eval_count"] = final_response.prompt_eval_count
                    if hasattr(final_response, 'eval_count'):
                        response["eval_count"] = final_response.eval_count
                    if hasattr(final_response, 'eval_duration'):
                        response["eval_duration"] = final_response.eval_duration
                
                return response
            else:
                # La génération en streaming sera gérée directement dans la route
                return {"streaming": True}
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            return {
                "error": str(e),
                "model": request.model,
                "done": True
            }
        finally:
            if client:
                client.close()
    
    def generate_stream(self, request: GenerateRequest) -> Generator[Dict[str, Any], None, None]:
        """
        Génère du texte en streaming en utilisant un modèle.
        
        Args:
            request: La requête de génération
            
        Returns:
            Un générateur de dictionnaires contenant les parties de la réponse
        """
        client, host, port = self.get_client_for_model(request.model)
        if not client:
            yield {
                "error": "Aucun serveur disponible pour ce modèle",
                "model": request.model,
                "done": True
            }
            return
            
        try:
            # Utiliser directement la méthode generate avec streaming
            for resp in client.generate(request.model, request.prompt, True, request.options):
                response = {
                    "model": request.model,
                    "done": resp.done if hasattr(resp, 'done') else False
                }
                
                if hasattr(resp, 'response'):
                    response["response"] = resp.response
                    
                # Ajouter les métriques si disponibles
                if hasattr(resp, 'total_duration'):
                    response["total_duration"] = resp.total_duration
                if hasattr(resp, 'load_duration'):
                    response["load_duration"] = resp.load_duration
                if hasattr(resp, 'prompt_eval_count'):
                    response["prompt_eval_count"] = resp.prompt_eval_count
                if hasattr(resp, 'eval_count'):
                    response["eval_count"] = resp.eval_count
                if hasattr(resp, 'eval_duration'):
                    response["eval_duration"] = resp.eval_duration
                
                yield response
                
                if hasattr(resp, 'done') and resp.done:
                    break
                    
            # Message de fin si nécessaire
            yield {
                "model": request.model,
                "response": "",
                "done": True
            }
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération en streaming: {e}")
            yield {
                "error": str(e),
                "model": request.model,
                "done": True
            }
        finally:
            if client:
                client.close()
    
    def chat(self, request: ChatRequest) -> Dict[str, Any]:
        """
        Génère une réponse de chat en utilisant un modèle.
        
        Args:
            request: La requête de chat
            
        Returns:
            Un dictionnaire contenant la réponse
        """
        client, host, port = self.get_client_for_model(request.model)
        if not client:
            return {
                "error": "Aucun serveur disponible pour ce modèle",
                "model": request.model,
                "done": True
            }
            
        try:
            if not request.stream:
                # Chat non-streaming
                # Convertir les messages au format attendu par le client
                messages_dict = []
                for msg in request.messages:
                    msg_dict = {
                        "role": msg.role,
                        "content": msg.content
                    }
                    if msg.images:
                        msg_dict["images"] = msg.images
                    messages_dict.append(msg_dict)
                
                response = None
                for resp in client.chat(request.model, messages_dict, False, request.options):
                    response = resp
                
                if response:
                    # Convertir la réponse au format attendu par l'API
                    return {
                        "model": request.model,
                        "message": {
                            "role": "assistant",
                            "content": response.message.content if hasattr(response, 'message') and hasattr(response.message, 'content') else ""
                        },
                        "done": True
                    }
                else:
                    return {
                        "model": request.model,
                        "message": {
                            "role": "assistant",
                            "content": "Pas de réponse du serveur"
                        },
                        "done": True
                    }
            else:
                # Le chat en streaming sera géré directement dans la route
                return {"streaming": True}
                
        except Exception as e:
            logger.error(f"Erreur lors du chat: {e}")
            return {
                "error": str(e),
                "model": request.model,
                "done": True
            }
        finally:
            if client:
                client.close()
    
    def chat_stream(self, request: ChatRequest) -> Generator[Dict[str, Any], None, None]:
        """
        Génère une réponse de chat en streaming en utilisant un modèle.
        
        Args:
            request: La requête de chat
            
        Returns:
            Un générateur de dictionnaires contenant les parties de la réponse
        """
        client, host, port = self.get_client_for_model(request.model)
        if not client:
            yield {
                "error": "Aucun serveur disponible pour ce modèle",
                "model": request.model,
                "done": True
            }
            return
            
        try:
            # Convertir les messages au format attendu par le client
            messages_dict = []
            for msg in request.messages:
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content
                }
                if msg.images:
                    msg_dict["images"] = msg.images
                messages_dict.append(msg_dict)
            
            # Utiliser directement la méthode chat avec streaming
            for resp in client.chat(request.model, messages_dict, True, request.options):
                if hasattr(resp, 'message'):
                    yield {
                        "model": request.model,
                        "message": {
                            "role": "assistant",
                            "content": resp.message.content if hasattr(resp.message, 'content') else ""
                        },
                        "done": resp.done if hasattr(resp, 'done') else False
                    }
                
                if hasattr(resp, 'done') and resp.done:
                    break
                    
            # Message de fin si nécessaire
            yield {
                "model": request.model,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True
            }
                
        except Exception as e:
            logger.error(f"Erreur lors du chat en streaming: {e}")
            yield {
                "error": str(e),
                "model": request.model,
                "done": True
            }
        finally:
            if client:
                client.close()
    
    def embeddings(self, request: EmbeddingsRequest) -> Dict[str, Any]:
        """
        Génère des embeddings en utilisant un modèle.
        
        Args:
            request: La requête d'embeddings
            
        Returns:
            Un dictionnaire contenant la réponse
        """
        client, host, port = self.get_client_for_model(request.model)
        if not client:
            return {
                "error": "Aucun serveur disponible pour ce modèle",
                "model": request.model,
                "embedding": []
            }
            
        try:
            # Définir un timeout court pour éviter le blocage
            start_time = time.time()
            max_time = 10  # 10 secondes maximum
            
            # Appel à embeddings
            response = client.embeddings(request.model, request.prompt, request.options)
            
            # Vérifier si on a dépassé le temps maximal
            if time.time() - start_time > max_time:
                return {
                    "model": request.model,
                    "embedding": [],
                    "error": "Timeout lors de la génération des embeddings"
                }
                
            # Réponse standard
            if hasattr(response, 'embeddings'):
                return {
                    "model": request.model,
                    "embedding": list(response.embeddings),
                }
            else:
                # En cas de réponse incorrecte
                return {
                    "model": request.model,
                    "embedding": [],
                    "error": "Format de réponse invalide du serveur"
                }
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings: {e}")
            return {
                "error": str(e),
                "model": request.model,
                "embedding": []
            }
        finally:
            if client:
                client.close()
    
    def get_status(self) -> StatusResponse:
        """
        Obtient le statut actuel du proxy.
        
        Returns:
            Un objet StatusResponse contenant le statut
        """
        from ..stats import request_stats
        
        distributed_available = False
        try:
            from olol.rpc.coordinator import InferenceCoordinator
            distributed_available = True
        except ImportError:
            pass
            
        timestamp = time.time()
        server_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        status = StatusResponse(
            timestamp=timestamp,
            server_time=server_time,
            proxy_uptime=int(timestamp - request_stats["start_time"]),
            active_requests=request_stats["active_requests"],
            total_requests=request_stats["total_requests"],
            distributed_available=distributed_available,
            distributed_enabled=self.cluster_manager.use_distributed_inference
        )
        
        # Ajouter les informations sur les serveurs
        server_addresses = self.cluster_manager.get_server_addresses()
        status.server_count = len(server_addresses)
        status.server_addresses = server_addresses
        
        return status
    
    def list_models(self) -> ModelsResponse:
        """
        Liste tous les modèles disponibles sur les serveurs.
        
        Returns:
            Un objet ModelsResponse contenant la liste des modèles
        """
        timestamp = time.time()
        models_dict = {}
        
        # Collecter les modèles de tous les serveurs
        for server_address in self.cluster_manager.get_server_addresses():
            try:
                # Créer un client temporaire
                if server_address.count(':') == 1:
                    host, port_str = server_address.split(':')
                    port = int(port_str)
                    client = OllamaClient(host=host, port=port)
                    
                    try:
                        # Récupérer les modèles
                        models_response = client.list_models()
                        
                        # Traiter les modèles
                        if hasattr(models_response, 'models'):
                            for model in models_response.models:
                                model_name = model.name
                                if model_name not in models_dict:
                                    # Créer une nouvelle entrée pour ce modèle
                                    models_dict[model_name] = ModelInfo(
                                        name=model_name,
                                        size=model.size if hasattr(model, 'size') else None,
                                        modified_at=model.modified_at if hasattr(model, 'modified_at') else None,
                                        servers=[server_address]
                                    )
                                else:
                                    # Ajouter ce serveur à la liste des serveurs pour ce modèle
                                    models_dict[model_name].servers.append(server_address)
                    finally:
                        client.close()
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des modèles de {server_address}: {e}")
                
        # Créer la réponse
        response = ModelsResponse(
            timestamp=timestamp,
            models=models_dict,
            model_count=len(models_dict)
        )
        
        return response
    
    def list_servers(self) -> ServersResponse:
        """
        Liste tous les serveurs du cluster.
        
        Returns:
            Un objet ServersResponse contenant la liste des serveurs
        """
        servers_dict = {}
        
        # Récupérer les informations sur les serveurs
        for server_address in self.cluster_manager.get_server_addresses():
            # Récupérer l'état de santé
            is_healthy = self.cluster_manager.is_server_healthy(server_address)
            
            # Récupérer la charge
            server_load = self.cluster_manager.get_server_load(server_address)
            
            # Récupérer les modèles
            server_models = self.cluster_manager.get_models_for_server(server_address)
            
            # Créer l'objet ServerInfo
            servers_dict[server_address] = ServerInfo(
                address=server_address,
                healthy=is_healthy,
                load=server_load,
                models=server_models
            )
        
        # Créer la réponse
        response = ServersResponse(
            servers=servers_dict,
            count=len(servers_dict),
            timestamp=time.time()
        )
        
        return response