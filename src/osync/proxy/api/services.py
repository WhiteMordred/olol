"""
Services pour l'API du proxy Ollama.

Ce module contient les services métier pour interagir avec les serveurs Ollama
à travers le gestionnaire de cluster. Il sépare la logique métier des routes HTTP.
"""

import json
import logging
import time
import datetime
from typing import Dict, List, Any, Optional, Generator, Tuple

from flask import Response, stream_with_context
from osync.sync.client import OllamaClient
from osync.proxy.db.database import DatabaseManager

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
        # Initialiser le gestionnaire de base de données
        self.db_manager = DatabaseManager()
    
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
            from osync.rpc.coordinator import InferenceCoordinator
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
        # Convertir l'ensemble (set) en liste pour permettre la sérialisation JSON
        status.server_addresses = list(server_addresses)
        
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
    
    # Nouvelles méthodes pour la persistance des données
    
    def get_historical_stats(self, time_range: str = "day", limit: int = 100) -> Dict[str, Any]:
        """
        Récupère les statistiques historiques de requêtes.
        
        Args:
            time_range: La plage de temps ('hour', 'day', 'week', 'month', 'all')
            limit: Nombre maximal d'enregistrements à retourner
            
        Returns:
            Un dictionnaire contenant les statistiques historiques
        """
        try:
            # Convertir time_range en secondes pour le filtrage
            now = datetime.datetime.now().timestamp()
            time_filter = 0
            
            if time_range == "hour":
                time_filter = now - 3600  # 1 heure
            elif time_range == "day":
                time_filter = now - 86400  # 24 heures
            elif time_range == "week":
                time_filter = now - 604800  # 7 jours
            elif time_range == "month":
                time_filter = now - 2592000  # 30 jours
            
            # Récupérer les données depuis TinyDB
            if time_range == "all":
                stats_data = self.db_manager.get_collection('stats').all()
            else:
                stats_data = self.db_manager.get_collection('stats').search(
                    lambda doc: doc.get('timestamp', 0) >= time_filter
                )
            
            # Limiter le nombre d'enregistrements
            stats_data = sorted(stats_data, key=lambda x: x.get('timestamp', 0), reverse=True)[:limit]
            
            # Formater la sortie
            formatted_stats = []
            for stat in stats_data:
                formatted_stat = {
                    "timestamp": stat.get('timestamp'),
                    "datetime": datetime.datetime.fromtimestamp(stat.get('timestamp')).strftime("%Y-%m-%d %H:%M:%S"),
                    "active_requests": stat.get('active_requests', 0),
                    "total_requests": stat.get('total_requests', 0),
                    "request_rate": stat.get('request_rate', 0),
                    "success_rate": stat.get('success_rate', 100),
                    "avg_response_time": stat.get('avg_response_time', 0)
                }
                
                if 'server_loads' in stat:
                    formatted_stat['server_loads'] = stat.get('server_loads', {})
                
                formatted_stats.append(formatted_stat)
            
            return {
                "count": len(formatted_stats),
                "time_range": time_range,
                "stats": formatted_stats
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques historiques: {e}")
            return {
                "error": str(e),
                "count": 0,
                "stats": []
            }
    
    def get_aggregated_stats(self, period: str = "hourly") -> Dict[str, Any]:
        """
        Récupère les statistiques agrégées par période.
        
        Args:
            period: La période d'agrégation ('hourly', 'daily', 'weekly')
            
        Returns:
            Un dictionnaire contenant les statistiques agrégées
        """
        try:
            # Récupérer les données depuis TinyDB
            stats_data = self.db_manager.get_collection(f'stats_aggregated_{period}').all()
            
            # Trier par timestamp décroissant
            stats_data = sorted(stats_data, key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Formater la sortie
            formatted_stats = []
            for stat in stats_data:
                formatted_stat = {
                    "timestamp": stat.get('timestamp'),
                    "datetime": datetime.datetime.fromtimestamp(stat.get('timestamp')).strftime("%Y-%m-%d %H:%M:%S"),
                    "period_start": stat.get('period_start'),
                    "period_end": stat.get('period_end'),
                    "total_requests": stat.get('total_requests', 0),
                    "avg_request_rate": stat.get('avg_request_rate', 0),
                    "avg_response_time": stat.get('avg_response_time', 0),
                    "success_rate": stat.get('success_rate', 100),
                    "max_concurrent": stat.get('max_concurrent', 0)
                }
                
                if 'model_distribution' in stat:
                    formatted_stat['model_distribution'] = stat.get('model_distribution', {})
                
                if 'server_distribution' in stat:
                    formatted_stat['server_distribution'] = stat.get('server_distribution', {})
                
                formatted_stats.append(formatted_stat)
            
            return {
                "period": period,
                "count": len(formatted_stats),
                "stats": formatted_stats
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques agrégées: {e}")
            return {
                "error": str(e),
                "period": period,
                "count": 0,
                "stats": []
            }
    
    def get_model_usage_stats(self, model_name: str = None, time_range: str = "day") -> Dict[str, Any]:
        """
        Récupère les statistiques d'utilisation pour un modèle spécifique ou pour tous les modèles.
        
        Args:
            model_name: Le nom du modèle (None pour tous les modèles)
            time_range: La plage de temps ('day', 'week', 'month', 'all')
            
        Returns:
            Un dictionnaire contenant les statistiques d'utilisation des modèles
        """
        try:
            # Convertir time_range en secondes pour le filtrage
            now = datetime.datetime.now().timestamp()
            time_filter = 0
            
            if time_range == "day":
                time_filter = now - 86400  # 24 heures
            elif time_range == "week":
                time_filter = now - 604800  # 7 jours
            elif time_range == "month":
                time_filter = now - 2592000  # 30 jours
            
            # Récupérer les données depuis TinyDB
            model_stats_collection = self.db_manager.get_collection('model_stats')
            
            if time_range == "all":
                if model_name:
                    stats_data = model_stats_collection.search(
                        lambda doc: doc.get('model_name') == model_name
                    )
                else:
                    stats_data = model_stats_collection.all()
            else:
                if model_name:
                    stats_data = model_stats_collection.search(
                        lambda doc: doc.get('timestamp', 0) >= time_filter and doc.get('model_name') == model_name
                    )
                else:
                    stats_data = model_stats_collection.search(
                        lambda doc: doc.get('timestamp', 0) >= time_filter
                    )
            
            # Agréger les données par modèle
            model_usage = {}
            for stat in stats_data:
                model = stat.get('model_name', 'unknown')
                if model not in model_usage:
                    model_usage[model] = {
                        "total_requests": 0,
                        "successful_requests": 0,
                        "failed_requests": 0,
                        "total_tokens": 0,
                        "avg_response_time": 0,
                        "requests_by_type": {
                            "generate": 0,
                            "chat": 0,
                            "embeddings": 0
                        }
                    }
                
                # Incrémenter les compteurs
                model_usage[model]["total_requests"] += 1
                if stat.get('success', True):
                    model_usage[model]["successful_requests"] += 1
                else:
                    model_usage[model]["failed_requests"] += 1
                
                model_usage[model]["total_tokens"] += stat.get('total_tokens', 0)
                
                # Mettre à jour la moyenne de temps de réponse
                current_avg = model_usage[model]["avg_response_time"]
                current_count = model_usage[model]["total_requests"]
                new_time = stat.get('response_time', 0)
                model_usage[model]["avg_response_time"] = (current_avg * (current_count - 1) + new_time) / current_count
                
                # Incrémenter le compteur par type
                req_type = stat.get('request_type', 'generate')
                model_usage[model]["requests_by_type"][req_type] = model_usage[model]["requests_by_type"].get(req_type, 0) + 1
            
            # Calculer les pourcentages et formater
            result = []
            for model, stats in model_usage.items():
                success_rate = 0
                if stats["total_requests"] > 0:
                    success_rate = (stats["successful_requests"] / stats["total_requests"]) * 100
                
                result.append({
                    "model_name": model,
                    "total_requests": stats["total_requests"],
                    "successful_requests": stats["successful_requests"],
                    "failed_requests": stats["failed_requests"],
                    "success_rate": round(success_rate, 2),
                    "total_tokens": stats["total_tokens"],
                    "avg_response_time": round(stats["avg_response_time"], 3),
                    "requests_by_type": stats["requests_by_type"]
                })
            
            # Trier par nombre total de requêtes
            result = sorted(result, key=lambda x: x["total_requests"], reverse=True)
            
            return {
                "time_range": time_range,
                "model_count": len(result),
                "models": result
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques d'utilisation des modèles: {e}")
            return {
                "error": str(e),
                "time_range": time_range,
                "model_count": 0,
                "models": []
            }
    
    def get_server_usage_stats(self, server_address: str = None, time_range: str = "day") -> Dict[str, Any]:
        """
        Récupère les statistiques d'utilisation pour un serveur spécifique ou pour tous les serveurs.
        
        Args:
            server_address: L'adresse du serveur (None pour tous les serveurs)
            time_range: La plage de temps ('day', 'week', 'month', 'all')
            
        Returns:
            Un dictionnaire contenant les statistiques d'utilisation des serveurs
        """
        try:
            # Convertir time_range en secondes pour le filtrage
            now = datetime.datetime.now().timestamp()
            time_filter = 0
            
            if time_range == "day":
                time_filter = now - 86400  # 24 heures
            elif time_range == "week":
                time_filter = now - 604800  # 7 jours
            elif time_range == "month":
                time_filter = now - 2592000  # 30 jours
            
            # Récupérer les données depuis TinyDB
            server_stats_collection = self.db_manager.get_collection('server_stats')
            
            if time_range == "all":
                if server_address:
                    stats_data = server_stats_collection.search(
                        lambda doc: doc.get('server_address') == server_address
                    )
                else:
                    stats_data = server_stats_collection.all()
            else:
                if server_address:
                    stats_data = server_stats_collection.search(
                        lambda doc: doc.get('timestamp', 0) >= time_filter and doc.get('server_address') == server_address
                    )
                else:
                    stats_data = server_stats_collection.search(
                        lambda doc: doc.get('timestamp', 0) >= time_filter
                    )
            
            # Agréger les données par serveur
            server_usage = {}
            for stat in stats_data:
                server = stat.get('server_address', 'unknown')
                if server not in server_usage:
                    server_usage[server] = {
                        "total_requests": 0,
                        "successful_requests": 0,
                        "failed_requests": 0,
                        "avg_response_time": 0,
                        "avg_load": 0,
                        "load_samples": 0,
                        "model_distribution": {}
                    }
                
                # Incrémenter les compteurs
                server_usage[server]["total_requests"] += 1
                if stat.get('success', True):
                    server_usage[server]["successful_requests"] += 1
                else:
                    server_usage[server]["failed_requests"] += 1
                
                # Mettre à jour la moyenne de temps de réponse
                current_avg = server_usage[server]["avg_response_time"]
                current_count = server_usage[server]["total_requests"]
                new_time = stat.get('response_time', 0)
                server_usage[server]["avg_response_time"] = (current_avg * (current_count - 1) + new_time) / current_count
                
                # Mettre à jour la moyenne de charge
                if 'server_load' in stat:
                    current_load_avg = server_usage[server]["avg_load"]
                    current_load_count = server_usage[server]["load_samples"]
                    new_load = stat.get('server_load', 0)
                    server_usage[server]["load_samples"] += 1
                    server_usage[server]["avg_load"] = (current_load_avg * current_load_count + new_load) / server_usage[server]["load_samples"]
                
                # Mettre à jour la distribution des modèles
                model = stat.get('model_name', 'unknown')
                server_usage[server]["model_distribution"][model] = server_usage[server]["model_distribution"].get(model, 0) + 1
            
            # Calculer les pourcentages et formater
            result = []
            for server, stats in server_usage.items():
                success_rate = 0
                if stats["total_requests"] > 0:
                    success_rate = (stats["successful_requests"] / stats["total_requests"]) * 100
                
                result.append({
                    "server_address": server,
                    "total_requests": stats["total_requests"],
                    "successful_requests": stats["successful_requests"],
                    "failed_requests": stats["failed_requests"],
                    "success_rate": round(success_rate, 2),
                    "avg_response_time": round(stats["avg_response_time"], 3),
                    "avg_load": round(stats["avg_load"], 2),
                    "model_distribution": stats["model_distribution"]
                })
            
            # Trier par nombre total de requêtes
            result = sorted(result, key=lambda x: x["total_requests"], reverse=True)
            
            return {
                "time_range": time_range,
                "server_count": len(result),
                "servers": result
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques d'utilisation des serveurs: {e}")
            return {
                "error": str(e),
                "time_range": time_range,
                "server_count": 0,
                "servers": []
            }