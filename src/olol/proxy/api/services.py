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
        server_address = self.cluster_manager.get_optimal_server(model_name)
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
        distributed_enabled = False
        
        try:
            from olol.rpc.coordinator import InferenceCoordinator
            distributed_available = True
            # Vérifier si la fonctionnalité distributed_inference est configurée
            # sans accéder directement à l'attribut qui pourrait ne pas exister
            distributed_enabled = hasattr(self.cluster_manager, 'use_distributed_inference') and self.cluster_manager.use_distributed_inference
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
            distributed_enabled=distributed_enabled
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
            is_healthy = self.cluster_manager.get_server_health(server_address)
            
            # Récupérer la charge
            server_load = self.cluster_manager.get_server_load(server_address)
            
            # Récupérer les modèles en utilisant les informations du cache global
            # Pour chaque modèle, vérifier si ce serveur est dans la liste des serveurs
            server_models = []
            all_models = self.cluster_manager.get_all_models()
            for model, servers in all_models.items():
                if server_address in servers:
                    server_models.append(model)
            
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
        
    def add_server(self, address: str, verify_health: bool = True) -> Dict[str, Any]:
        """
        Ajoute un nouveau serveur au cluster.
        
        Args:
            address: L'adresse du serveur au format host:port
            verify_health: Si True, vérifie la santé du serveur avant de l'ajouter
            
        Returns:
            Un dictionnaire contenant le résultat de l'opération
        """
        try:
            # Vérifier le format de l'adresse
            if address.count(':') != 1:
                return {
                    "success": False,
                    "error": "Format d'adresse invalide, utilisez host:port"
                }
            
            # Ajouter au cluster via l'attribut _cluster de ClusterManager
            if hasattr(self.cluster_manager, '_cluster') and self.cluster_manager._cluster:
                with self.cluster_manager._cluster.server_lock:
                    # Vérifier si le serveur existe déjà
                    if address in self.cluster_manager._cluster.server_addresses:
                        return {
                            "success": False,
                            "error": "Le serveur existe déjà dans le cluster"
                        }
                    
                    # Vérifier la santé si nécessaire
                    if verify_health:
                        client = self.cluster_manager.get_client_for_server(address)
                        if not client:
                            return {
                                "success": False,
                                "error": "Impossible de créer un client pour ce serveur"
                            }
                        
                        try:
                            is_healthy = client.check_health()
                            if not is_healthy:
                                return {
                                    "success": False,
                                    "error": "Le serveur n'est pas en bonne santé"
                                }
                        finally:
                            client.close()
                    
                    # Ajouter l'adresse à la liste
                    self.cluster_manager._cluster.server_addresses.append(address)
                    
                    # Marquer comme sain dans le cache de santé
                    with self.cluster_manager._cluster.health_lock:
                        self.cluster_manager._cluster.server_health[address] = True
                    
                    # Initialiser la charge à 0
                    with self.cluster_manager._cluster.server_lock:
                        self.cluster_manager._cluster.server_loads[address] = 0.0
                    
                    # Mettre à jour le cache dans le gestionnaire
                    self.cluster_manager.refresh_cache()
                    
                    # Découvrir les modèles disponibles sur ce serveur
                    try:
                        host, port_str = address.split(':')
                        port = int(port_str)
                        client = OllamaClient(host=host, port=port)
                        
                        try:
                            models_response = client.list_models()
                            
                            if hasattr(models_response, 'models'):
                                with self.cluster_manager._cluster.model_lock:
                                    for model in models_response.models:
                                        model_name = model.name
                                        if model_name not in self.cluster_manager._cluster.model_server_map:
                                            self.cluster_manager._cluster.model_server_map[model_name] = set([address])
                                        else:
                                            self.cluster_manager._cluster.model_server_map[model_name].add(address)
                        finally:
                            client.close()
                    except Exception as e:
                        logger.warning(f"Erreur lors de la découverte des modèles pour {address}: {e}")
                        # On continue même en cas d'erreur car le serveur est ajouté
                    
                    return {
                        "success": True,
                        "address": address,
                        "message": "Serveur ajouté avec succès"
                    }
            else:
                return {
                    "success": False,
                    "error": "Gestionnaire de cluster non initialisé"
                }
        
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du serveur {address}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def remove_server(self, address: str) -> Dict[str, Any]:
        """
        Supprime un serveur du cluster.
        
        Args:
            address: L'adresse du serveur à supprimer
            
        Returns:
            Un dictionnaire contenant le résultat de l'opération
        """
        try:
            # Supprimer du cluster via l'attribut _cluster de ClusterManager
            if hasattr(self.cluster_manager, '_cluster') and self.cluster_manager._cluster:
                with self.cluster_manager._cluster.server_lock:
                    # Vérifier si le serveur existe
                    if address not in self.cluster_manager._cluster.server_addresses:
                        return {
                            "success": False,
                            "error": "Le serveur n'existe pas dans le cluster"
                        }
                    
                    # Supprimer de la liste des adresses
                    self.cluster_manager._cluster.server_addresses.remove(address)
                    
                    # Supprimer du cache de santé
                    with self.cluster_manager._cluster.health_lock:
                        if address in self.cluster_manager._cluster.server_health:
                            del self.cluster_manager._cluster.server_health[address]
                    
                    # Supprimer du cache de charge
                    with self.cluster_manager._cluster.server_lock:
                        if address in self.cluster_manager._cluster.server_loads:
                            del self.cluster_manager._cluster.server_loads[address]
                    
                    # Supprimer des mappages de modèles
                    with self.cluster_manager._cluster.model_lock:
                        for model_name, servers in list(self.cluster_manager._cluster.model_server_map.items()):
                            if address in servers:
                                servers.remove(address)
                                # Si plus aucun serveur ne propose ce modèle, supprimer l'entrée
                                if not servers:
                                    del self.cluster_manager._cluster.model_server_map[model_name]
                    
                    # Mettre à jour le cache dans le gestionnaire
                    self.cluster_manager.refresh_cache()
                    
                    return {
                        "success": True,
                        "address": address,
                        "message": "Serveur supprimé avec succès"
                    }
            else:
                return {
                    "success": False,
                    "error": "Gestionnaire de cluster non initialisé"
                }
        
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du serveur {address}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_server_details(self, address: str) -> Dict[str, Any]:
        """
        Obtient les détails d'un serveur spécifique.
        
        Args:
            address: L'adresse du serveur au format host:port
            
        Returns:
            Un dictionnaire contenant les détails du serveur
        """
        try:
            # Vérifier si le serveur existe
            if not self.cluster_manager.has_server(address):
                return {
                    "success": False,
                    "error": "Serveur non trouvé"
                }
                
            # Récupérer l'état de santé
            is_healthy = self.cluster_manager.get_server_health(address)
            
            # Récupérer la charge
            server_load = self.cluster_manager.get_server_load(address)
            
            # Récupérer les modèles disponibles sur ce serveur
            server_models = []
            all_models = self.cluster_manager.get_all_models()
            for model, servers in all_models.items():
                if address in servers:
                    server_models.append(model)
            
            # Obtenir le timestamp de la dernière vérification de santé
            last_health_check = self.cluster_manager.get_last_health_check_time(address)
            
            # Obtenir les informations de système
            system_info = {}
            try:
                host, port_str = address.split(':')
                port = int(port_str)
                client = OllamaClient(host=host, port=port)
                try:
                    # Essayer d'obtenir les informations système si disponibles
                    system_info = client.get_system_info()
                finally:
                    client.close()
            except Exception as e:
                logger.warning(f"Impossible d'obtenir les infos système pour {address}: {e}")
            
            return {
                "success": True,
                "address": address,
                "healthy": is_healthy,
                "load": server_load,
                "models": server_models,
                "last_health_check": last_health_check,
                "system_info": system_info
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'obtention des détails du serveur {address}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def check_server_health(self, address: str) -> Dict[str, Any]:
        """
        Vérifie la santé d'un serveur spécifique.
        
        Args:
            address: L'adresse du serveur à vérifier
            
        Returns:
            Un dictionnaire contenant le résultat de la vérification
        """
        try:
            # Vérifier si le serveur existe
            if not self.cluster_manager.has_server(address):
                return {
                    "success": False,
                    "error": "Serveur non trouvé"
                }
            
            # Créer un client pour ce serveur
            client = self.cluster_manager.get_client_for_server(address)
            if not client:
                return {
                    "success": False,
                    "error": "Impossible de créer un client pour ce serveur"
                }
            
            # Vérifier la santé
            is_healthy = False
            try:
                is_healthy = client.check_health()
            finally:
                client.close()
            
            # Mettre à jour l'état de santé dans le cluster
            self.cluster_manager.update_server_health(address, is_healthy)
            
            return {
                "success": True,
                "address": address,
                "healthy": is_healthy,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de santé du serveur {address}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Obtient un rapport de santé global du cluster.
        
        Returns:
            Un dictionnaire contenant le rapport de santé
        """
        try:
            servers = {}
            total_servers = 0
            healthy_servers = 0
            
            # Récupérer les informations de santé pour chaque serveur
            for address in self.cluster_manager.get_server_addresses():
                total_servers += 1
                
                # Récupérer l'état de santé
                is_healthy = self.cluster_manager.get_server_health(address)
                if is_healthy:
                    healthy_servers += 1
                
                # Récupérer la charge
                server_load = self.cluster_manager.get_server_load(address)
                
                # Ajouter ce serveur au rapport
                servers[address] = {
                    "healthy": is_healthy,
                    "load": server_load
                }
            
            # Calculer le pourcentage de santé global
            if total_servers > 0:
                health_percent = (healthy_servers / total_servers) * 100
            else:
                health_percent = 0
            
            return {
                "success": True,
                "timestamp": time.time(),
                "total_servers": total_servers,
                "healthy_servers": healthy_servers,
                "health_percent": health_percent,
                "servers": servers
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport de santé: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_health_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Obtient des statistiques de santé sur une période donnée.
        
        Args:
            hours: Le nombre d'heures pour lesquelles récupérer les données
            
        Returns:
            Un dictionnaire contenant les statistiques de santé
        """
        try:
            # Vérifier si la fonctionnalité est supportée
            if not hasattr(self.cluster_manager, 'get_health_history'):
                return {
                    "success": False,
                    "error": "Les statistiques de santé ne sont pas disponibles"
                }
                
            # Récupérer l'historique de santé
            history = self.cluster_manager.get_health_history(hours)
            if not history:
                return {
                    "success": True,
                    "message": "Pas de données de santé disponibles",
                    "stats": {}
                }
            
            # Calculer les statistiques à partir de l'historique
            stats = {}
            for address in self.cluster_manager.get_server_addresses():
                server_stats = {
                    "total_checks": 0,
                    "healthy_checks": 0,
                    "health_ratio": 0,
                    "avg_response_time": 0
                }
                
                # Extraire les données pour ce serveur
                server_history = [entry for entry in history if entry.get("server") == address]
                
                if server_history:
                    server_stats["total_checks"] = len(server_history)
                    server_stats["healthy_checks"] = sum(1 for entry in server_history if entry.get("healthy", False))
                    
                    if server_stats["total_checks"] > 0:
                        server_stats["health_ratio"] = server_stats["healthy_checks"] / server_stats["total_checks"]
                        
                    # Calculer le temps de réponse moyen si disponible
                    response_times = [entry.get("response_time") for entry in server_history if "response_time" in entry]
                    if response_times:
                        server_stats["avg_response_time"] = sum(response_times) / len(response_times)
                
                stats[address] = server_stats
            
            return {
                "success": True,
                "period_hours": hours,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques de santé: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_server_health_history(self, server: str, hours: int = 24) -> Dict[str, Any]:
        """
        Obtient l'historique de santé d'un serveur spécifique.
        
        Args:
            server: L'adresse du serveur
            hours: Le nombre d'heures pour lesquelles récupérer l'historique
            
        Returns:
            Un dictionnaire contenant l'historique de santé
        """
        try:
            # Vérifier si le serveur existe
            if not self.cluster_manager.has_server(server):
                return {
                    "success": False,
                    "error": "Serveur non trouvé"
                }
                
            # Vérifier si la fonctionnalité est supportée
            if not hasattr(self.cluster_manager, 'get_server_health_history'):
                return {
                    "success": False,
                    "error": "L'historique de santé n'est pas disponible"
                }
                
            # Récupérer l'historique de santé pour ce serveur
            history = self.cluster_manager.get_server_health_history(server, hours)
            
            return {
                "success": True,
                "server": server,
                "period_hours": hours,
                "history": history
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique du serveur {server}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_load_chart_data(self, hours: int = 1) -> Dict[str, Any]:
        """
        Obtient les données de graphique pour la charge des serveurs.
        
        Args:
            hours: Le nombre d'heures pour lesquelles récupérer les données
            
        Returns:
            Un dictionnaire contenant les données du graphique
        """
        try:
            # Vérifier si la fonctionnalité est supportée
            if not hasattr(self.cluster_manager, 'get_load_history'):
                return {
                    "success": False,
                    "error": "Les données de charge ne sont pas disponibles"
                }
                
            # Récupérer l'historique de charge
            load_history = self.cluster_manager.get_load_history(hours)
            
            # Organiser les données pour le graphique
            servers = set()
            timestamps = set()
            
            # Collecter tous les serveurs et timestamps
            for entry in load_history:
                if "server" in entry and "timestamp" in entry:
                    servers.add(entry["server"])
                    timestamps.add(entry["timestamp"])
            
            # Trier les timestamps
            sorted_timestamps = sorted(timestamps)
            
            # Créer des séries pour chaque serveur
            series = {}
            for server in servers:
                series[server] = []
                
                # Pour chaque timestamp, trouver la valeur pour ce serveur
                for ts in sorted_timestamps:
                    # Trouver l'entrée correspondante dans l'historique
                    matching_entries = [e for e in load_history 
                                        if e.get("server") == server and e.get("timestamp") == ts]
                    
                    if matching_entries:
                        # Prendre la première entrée correspondante
                        value = matching_entries[0].get("load", 0)
                    else:
                        # Pas de données pour ce point
                        value = None
                        
                    series[server].append({
                        "timestamp": ts,
                        "value": value
                    })
            
            return {
                "success": True,
                "period_hours": hours,
                "timestamps": sorted_timestamps,
                "series": series
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de graphique de charge: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_health_chart_data(self, hours: int = 1) -> Dict[str, Any]:
        """
        Obtient les données de graphique pour la santé des serveurs.
        
        Args:
            hours: Le nombre d'heures pour lesquelles récupérer les données
            
        Returns:
            Un dictionnaire contenant les données du graphique
        """
        try:
            # Vérifier si la fonctionnalité est supportée
            if not hasattr(self.cluster_manager, 'get_health_history'):
                return {
                    "success": False,
                    "error": "Les données de santé ne sont pas disponibles"
                }
                
            # Récupérer l'historique de santé
            health_history = self.cluster_manager.get_health_history(hours)
            
            # Organiser les données pour le graphique
            servers = set()
            timestamps = set()
            
            # Collecter tous les serveurs et timestamps
            for entry in health_history:
                if "server" in entry and "timestamp" in entry:
                    servers.add(entry["server"])
                    timestamps.add(entry["timestamp"])
            
            # Trier les timestamps
            sorted_timestamps = sorted(timestamps)
            
            # Créer des séries pour chaque serveur
            series = {}
            for server in servers:
                series[server] = []
                
                # Pour chaque timestamp, trouver la valeur pour ce serveur
                for ts in sorted_timestamps:
                    # Trouver l'entrée correspondante dans l'historique
                    matching_entries = [e for e in health_history 
                                        if e.get("server") == server and e.get("timestamp") == ts]
                    
                    if matching_entries:
                        # Prendre la première entrée correspondante
                        # Convertir booléen en valeur numérique
                        value = 1 if matching_entries[0].get("healthy", False) else 0
                    else:
                        # Pas de données pour ce point
                        value = None
                        
                    series[server].append({
                        "timestamp": ts,
                        "value": value
                    })
            
            return {
                "success": True,
                "period_hours": hours,
                "timestamps": sorted_timestamps,
                "series": series
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de graphique de santé: {e}")
            return {
                "success": False,
                "error": str(e)
            }