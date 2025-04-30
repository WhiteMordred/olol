"""
Module d'ordonnancement des requêtes d'inférence.

Ce module fournit un système intelligent d'allocation des requêtes
aux serveurs les plus appropriés, en tenant compte de la charge,
des ressources et de la disponibilité des modèles.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import random

from osync.proxy.cluster.manager import get_cluster_manager
from osync.proxy.cluster.registry import get_model_registry
from osync.proxy.queue.queue import get_queue_manager
from osync.sync.client import OllamaClient

# Configuration du logging
logger = logging.getLogger(__name__)


class RequestScheduler:
    """
    Ordonnanceur de requêtes pour l'allocation intelligente aux serveurs.
    
    Cette classe gère l'attribution des requêtes d'inférence aux serveurs
    appropriés, en tenant compte de multiples facteurs comme la charge,
    la disponibilité des modèles et les priorités.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Implémentation du pattern Singleton pour RequestScheduler."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RequestScheduler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialise l'ordonnanceur de requêtes."""
        # Éviter la réinitialisation du singleton
        if self._initialized:
            return
            
        self.cluster_manager = get_cluster_manager()
        self.model_registry = get_model_registry()
        self.queue_manager = get_queue_manager()
        
        # Paramètres d'ordonnancement
        self._config = {
            "dequeue_batch_size": 5,  # Nombre max de requêtes à déqueuer en une fois
            "scheduling_interval": 1.0,  # Intervalle entre deux cycles d'ordonnancement (secondes)
            "max_retry_count": 3,  # Nombre maximal de tentatives pour une requête
            "health_check_threshold": 0.3,  # Seuil pour considérer un serveur en bonne santé (0-1)
            "load_threshold": 0.8,  # Seuil de charge maximale pour un serveur (0-1)
            "model_weight": 0.6,  # Poids pour la disponibilité du modèle
            "load_weight": 0.3,  # Poids pour la charge du serveur
            "latency_weight": 0.1,  # Poids pour la latence du serveur
            "enable_load_balancing": True,  # Activer la répartition de charge
            "enable_auto_retry": True  # Réessayer automatiquement en cas d'échec
        }
        
        # Cache des performances des serveurs
        self._server_metrics = {}  # server_address -> metric_data
        self._server_models = {}   # server_address -> [models]
        
        # Pour l'exécution en arrière-plan
        self._scheduler_thread = None
        self._stop_scheduler = threading.Event()
        
        # Verrous
        self._metrics_lock = threading.RLock()
        
        # Statistiques d'ordonnancement
        self._stats = {
            "total_scheduled": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "by_model": {},
            "by_server": {}
        }
        
        # Initialisation complète
        self._initialized = True
        logger.info("RequestScheduler initialisé")
    
    def start(self):
        """Démarre l'ordonnanceur en arrière-plan."""
        if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
            self._stop_scheduler.clear()
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True
            )
            self._scheduler_thread.start()
            logger.info("Thread d'ordonnancement démarré")
    
    def stop(self):
        """Arrête l'ordonnanceur."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._stop_scheduler.set()
            self._scheduler_thread.join(timeout=5.0)
            self._scheduler_thread = None
            logger.info("Thread d'ordonnancement arrêté")
    
    def _scheduler_loop(self):
        """Boucle principale d'ordonnancement."""
        while not self._stop_scheduler.is_set():
            try:
                # 1. Vérifier la disponibilité et l'état des serveurs
                self._update_server_metrics()
                
                # 2. Regrouper les requêtes similaires si le batching est activé
                if self.queue_manager._config["enable_batching"]:
                    self.queue_manager.batch_similar_requests()
                
                # 3. Déqueuer les requêtes les plus prioritaires
                requests = self.queue_manager.dequeue(
                    batch_size=self._config["dequeue_batch_size"]
                )
                
                # 4. Pour chaque requête, sélectionner un serveur approprié et l'exécuter
                for request in requests:
                    self._process_request(request)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'ordonnancement: {str(e)}")
            
            # Attendre avant le prochain cycle
            self._stop_scheduler.wait(self._config["scheduling_interval"])
    
    def _update_server_metrics(self):
        """Met à jour les métriques des serveurs."""
        try:
            # Récupérer les serveurs actifs
            server_addresses = self.cluster_manager.get_server_addresses()
            healthy_servers = self.cluster_manager.get_healthy_servers()
            
            # Récupérer les informations de modèles par serveur
            model_status = self.model_registry.get_model_status()
            
            with self._metrics_lock:
                # Mettre à jour la liste des modèles par serveur
                self._server_models = {}
                
                # Traiter les modèles disponibles
                if "models" in model_status:
                    for model_info in model_status["models"]:
                        model_name = model_info.get("name")
                        if model_name:
                            servers = model_info.get("servers", [])
                            
                            for server in servers:
                                if server not in self._server_models:
                                    self._server_models[server] = []
                                self._server_models[server].append(model_name)
                
                # Mettre à jour les métriques des serveurs
                for address in server_addresses:
                    if address not in self._server_metrics:
                        self._server_metrics[address] = {
                            "healthy": address in healthy_servers,
                            "load": self.cluster_manager.get_server_load(address),
                            "last_latency_ms": 0,
                            "avg_latency_ms": 0,
                            "success_rate": 1.0,
                            "total_requests": 0,
                            "successful_requests": 0,
                            "failed_requests": 0,
                            "last_updated": datetime.now().isoformat()
                        }
                    else:
                        # Mettre à jour l'état de santé et la charge
                        self._server_metrics[address]["healthy"] = address in healthy_servers
                        self._server_metrics[address]["load"] = self.cluster_manager.get_server_load(address)
                
                # Supprimer les métriques des serveurs qui ne sont plus dans le cluster
                for address in list(self._server_metrics.keys()):
                    if address not in server_addresses:
                        del self._server_metrics[address]
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques des serveurs: {str(e)}")
    
    def _process_request(self, request):
        """
        Traite une requête en sélectionnant un serveur et en l'exécutant.
        
        Args:
            request: La requête à traiter
        """
        try:
            # Extraire les informations essentielles
            request_id = request.get("id")
            model = request.get("model")
            
            # Incrémenter les statistiques
            self._stats["total_scheduled"] += 1
            if model not in self._stats["by_model"]:
                self._stats["by_model"][model] = {
                    "scheduled": 0, "successful": 0, "failed": 0
                }
            self._stats["by_model"][model]["scheduled"] += 1
            
            logger.info(f"Traitement de la requête {request_id} pour le modèle {model}")
            
            # Sélectionner le serveur optimal
            server = self._select_optimal_server(request)
            
            if server:
                # Mettre à jour les statistiques du serveur
                if server not in self._stats["by_server"]:
                    self._stats["by_server"][server] = {
                        "scheduled": 0, "successful": 0, "failed": 0
                    }
                self._stats["by_server"][server]["scheduled"] += 1
                
                # Exécuter la requête sur le serveur sélectionné
                success = self._execute_request(request, server)
                
                if success:
                    self._stats["successful"] += 1
                    self._stats["by_model"][model]["successful"] += 1
                    self._stats["by_server"][server]["successful"] += 1
                else:
                    self._stats["failed"] += 1
                    self._stats["by_model"][model]["failed"] += 1
                    self._stats["by_server"][server]["failed"] += 1
                    
                    # Réessayer si configuré pour le faire et pas trop de tentatives
                    retry_count = request.get("retry_count", 0)
                    if (self._config["enable_auto_retry"] and 
                        retry_count < self._config["max_retry_count"]):
                        
                        # Préparer une nouvelle tentative
                        self._retry_request(request)
                        self._stats["retried"] += 1
            else:
                # Aucun serveur disponible, marquer comme échoué
                logger.warning(f"Aucun serveur disponible pour la requête {request_id}, modèle {model}")
                
                # Mettre à jour le statut de la requête
                self.queue_manager.update_request_status(
                    request_id=request_id,
                    status=self.queue_manager.STATUS_FAILED,
                    result={"error": "No suitable server available"}
                )
                
                self._stats["failed"] += 1
                self._stats["by_model"][model]["failed"] += 1
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête {request.get('id')}: {str(e)}")
            
            # Essayer de mettre à jour le statut en cas d'erreur
            try:
                self.queue_manager.update_request_status(
                    request_id=request.get("id"),
                    status=self.queue_manager.STATUS_FAILED,
                    result={"error": str(e)}
                )
            except:
                pass
    
    def _select_optimal_server(self, request) -> Optional[str]:
        """
        Sélectionne le serveur optimal pour une requête.
        
        Args:
            request: La requête à traiter
            
        Returns:
            L'adresse du serveur sélectionné ou None si aucun n'est disponible
        """
        try:
            model = request.get("model")
            
            # Obtenir tous les serveurs qui ont ce modèle
            candidate_servers = []
            with self._metrics_lock:
                for server, models in self._server_models.items():
                    if model in models:
                        # Vérifier si le serveur est en bonne santé
                        metrics = self._server_metrics.get(server, {})
                        if (metrics.get("healthy", False) and 
                            metrics.get("load", 1.0) < self._config["load_threshold"]):
                            candidate_servers.append(server)
            
            if not candidate_servers:
                return None
                
            # Si un seul serveur est disponible, le retourner directement
            if len(candidate_servers) == 1:
                return candidate_servers[0]
                
            # Calculer un score pour chaque serveur
            server_scores = []
            for server in candidate_servers:
                metrics = self._server_metrics.get(server, {})
                
                # Score basé sur la charge (inversée pour que moins de charge = meilleur score)
                load = metrics.get("load", 0.5)
                load_score = 1.0 - load
                
                # Score basé sur la latence (inversée pour que moins de latence = meilleur score)
                latency = metrics.get("avg_latency_ms", 100)
                latency_score = 1.0 / (1.0 + (latency / 100.0))  # Normaliser pour rester entre 0-1
                
                # Score basé sur le taux de succès
                success_rate = metrics.get("success_rate", 1.0)
                
                # Calculer le score final
                final_score = (
                    load_score * self._config["load_weight"] +
                    latency_score * self._config["latency_weight"] +
                    success_rate * self._config["model_weight"]
                )
                
                server_scores.append((server, final_score))
            
            # Trier par score décroissant
            server_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Retourner le meilleur serveur
            return server_scores[0][0] if server_scores else None
            
        except Exception as e:
            logger.error(f"Erreur lors de la sélection du serveur optimal: {str(e)}")
            return None
    
    def _execute_request(self, request, server) -> bool:
        """
        Exécute une requête sur un serveur spécifique.
        
        Args:
            request: La requête à exécuter
            server: L'adresse du serveur
            
        Returns:
            True si l'exécution a réussi, False sinon
        """
        request_id = request.get("id")
        model = request.get("model")
        start_time = datetime.now()
        
        try:
            # Créer un client pour ce serveur
            client = OllamaClient(server)
            
            # Préparer les paramètres selon le type de requête
            if "prompt" in request:
                # Requête de génération simple
                result = client.generate(
                    model=model,
                    prompt=request["prompt"],
                    options=request.get("options", {})
                )
                success = True
            elif "messages" in request:
                # Requête de chat
                result = client.chat(
                    model=model,
                    messages=request["messages"],
                    options=request.get("options", {})
                )
                success = True
            else:
                raise ValueError("La requête ne contient ni prompt ni messages")
            
            # Calculer le temps d'exécution
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Mettre à jour les métriques du serveur
            self._update_server_performance(server, True, execution_time_ms)
            
            # Mettre à jour le statut de la requête
            self.queue_manager.update_request_status(
                request_id=request_id,
                status=self.queue_manager.STATUS_COMPLETED,
                result=result
            )
            
            logger.info(f"Requête {request_id} exécutée avec succès sur {server} en {execution_time_ms:.2f}ms")
            return True
            
        except Exception as e:
            # Calculer le temps jusqu'à l'échec
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Mettre à jour les métriques du serveur
            self._update_server_performance(server, False, execution_time_ms)
            
            logger.error(f"Échec de l'exécution de la requête {request_id} sur {server}: {str(e)}")
            return False
    
    def _retry_request(self, request):
        """
        Remet une requête en file d'attente pour une nouvelle tentative.
        
        Args:
            request: La requête à réessayer
        """
        try:
            request_id = request.get("id")
            
            # Incrémenter le compteur de tentatives
            retry_count = request.get("retry_count", 0) + 1
            
            # Créer une nouvelle requête avec les mêmes données
            new_request = request.copy()
            new_request["retry_count"] = retry_count
            new_request["retry_of"] = request_id
            new_request["created_at"] = datetime.now().isoformat()
            new_request["status"] = self.queue_manager.STATUS_PENDING
            new_request["batch_id"] = None
            new_request["server_assigned"] = None
            
            # Ajouter une priorité plus élevée pour les réessais
            new_request["priority"] = min(100, request.get("priority", 10) + 5)
            
            # Générer un nouvel ID
            new_request_id = self.queue_manager.enqueue(new_request)
            
            # Mettre à jour le statut de la requête originale
            self.queue_manager.update_request_status(
                request_id=request_id,
                status=self.queue_manager.STATUS_FAILED,
                result={"error": "Execution failed, retrying", "retry_id": new_request_id}
            )
            
            logger.info(f"Requête {request_id} mise en file d'attente pour réessai (nouvelle requête {new_request_id}, tentative {retry_count})")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise en file d'attente pour réessai: {str(e)}")
    
    def _update_server_performance(self, server, success, latency_ms):
        """
        Met à jour les métriques de performance d'un serveur.
        
        Args:
            server: Adresse du serveur
            success: True si la requête a réussi, False sinon
            latency_ms: Temps d'exécution en millisecondes
        """
        try:
            with self._metrics_lock:
                if server not in self._server_metrics:
                    self._server_metrics[server] = {
                        "healthy": True,
                        "load": 0.0,
                        "last_latency_ms": latency_ms,
                        "avg_latency_ms": latency_ms,
                        "success_rate": 1.0 if success else 0.0,
                        "total_requests": 1,
                        "successful_requests": 1 if success else 0,
                        "failed_requests": 0 if success else 1,
                        "last_updated": datetime.now().isoformat()
                    }
                else:
                    metrics = self._server_metrics[server]
                    
                    # Mettre à jour les compteurs
                    metrics["total_requests"] += 1
                    if success:
                        metrics["successful_requests"] += 1
                    else:
                        metrics["failed_requests"] += 1
                    
                    # Recalculer le taux de succès
                    metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
                    
                    # Mettre à jour la latence avec une moyenne pondérée
                    metrics["last_latency_ms"] = latency_ms
                    if metrics["avg_latency_ms"] == 0:
                        metrics["avg_latency_ms"] = latency_ms
                    else:
                        # Donner un poids de 25% à la nouvelle valeur
                        metrics["avg_latency_ms"] = metrics["avg_latency_ms"] * 0.75 + latency_ms * 0.25
                    
                    metrics["last_updated"] = datetime.now().isoformat()
                    
                # Notifier le gestionnaire de cluster des métriques mises à jour
                self.cluster_manager.update_server_metrics(
                    server_id=server,
                    response_time_ms=latency_ms
                )
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des performances du serveur {server}: {str(e)}")
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """
        Obtient des statistiques sur l'ordonnanceur.
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = self._stats.copy()
        
        # Ajouter des informations sur la configuration
        stats["config"] = self._config.copy()
        
        # Ajouter des informations temporelles
        stats["timestamp"] = datetime.now().isoformat()
        
        # Ajouter des informations sur les serveurs
        stats["servers"] = {}
        with self._metrics_lock:
            for server, metrics in self._server_metrics.items():
                stats["servers"][server] = metrics.copy()
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Met à jour la configuration de l'ordonnanceur.
        
        Args:
            new_config: Nouvelles valeurs de configuration
            
        Returns:
            Configuration mise à jour
        """
        # Fusionner la nouvelle configuration avec l'existante
        self._config.update(new_config)
        
        logger.info(f"Configuration de l'ordonnanceur mise à jour: {new_config}")
        return self._config.copy()
    
    def assign_request_to_server(self, request_id: str, server: str) -> bool:
        """
        Attribue manuellement une requête à un serveur spécifique.
        
        Args:
            request_id: ID de la requête
            server: Adresse du serveur
            
        Returns:
            True si l'attribution a réussi, False sinon
        """
        try:
            # Récupérer la requête
            request = self.queue_manager.get_request(request_id)
            if not request:
                logger.warning(f"Requête {request_id} non trouvée")
                return False
            
            # Vérifier si la requête est en attente
            if request.get("status") != self.queue_manager.STATUS_PENDING:
                logger.warning(f"La requête {request_id} n'est pas en attente (statut: {request.get('status')})")
                return False
            
            # Vérifier si le serveur existe et est en bonne santé
            if server not in self._server_metrics or not self._server_metrics[server].get("healthy", False):
                logger.warning(f"Serveur {server} non disponible ou en mauvaise santé")
                return False
            
            # Vérifier si le serveur a le modèle requis
            model = request.get("model")
            if server not in self._server_models or model not in self._server_models[server]:
                logger.warning(f"Le serveur {server} n'a pas le modèle {model}")
                return False
            
            # Mettre à jour la requête pour l'attribuer à ce serveur
            request["server_assigned"] = server
            
            # Persister les changements
            self.queue_manager.sync_manager.write_and_sync("inference_queue", request)
            
            logger.info(f"Requête {request_id} attribuée manuellement au serveur {server}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'attribution de la requête {request_id} au serveur {server}: {str(e)}")
            return False
    
    def reset_stats(self):
        """Réinitialise les statistiques d'ordonnancement."""
        self._stats = {
            "total_scheduled": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "by_model": {},
            "by_server": {}
        }
        logger.info("Statistiques d'ordonnancement réinitialisées")


# Fonction singleton pour accéder à l'ordonnanceur
def get_request_scheduler() -> RequestScheduler:
    """
    Récupère l'instance unique de l'ordonnanceur de requêtes.
    
    Returns:
        Instance du RequestScheduler
    """
    return RequestScheduler()