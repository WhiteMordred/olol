"""
Module de gestion de la file d'attente pour les inférences.

Ce module implémente un système de file d'attente persistant basé sur TinyDB
pour stocker et gérer les requêtes d'inférence, permettant l'optimisation
du traitement par batching et la priorisation des requêtes.
"""

import uuid
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import json

from osync.proxy.db.sync_manager import get_sync_manager
from osync.proxy.db.database import get_db
from osync.proxy.cluster.manager import get_cluster_manager

# Configuration du logging
logger = logging.getLogger(__name__)


class QueueManager:
    """
    Gestionnaire de file d'attente pour les requêtes d'inférence.
    
    Cette classe implémente une file d'attente persistante qui stocke les requêtes
    dans TinyDB et les synchronise avec la RAM pour des performances optimales.
    Elle prend également en charge le regroupement (batching) et la priorisation
    des requêtes.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Implémentation du pattern Singleton pour QueueManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QueueManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialise le gestionnaire de file d'attente."""
        # Éviter la réinitialisation du singleton
        if self._initialized:
            return
            
        self.db = get_db()
        self.sync_manager = get_sync_manager()
        self.cluster_manager = get_cluster_manager()
        
        # Cache en mémoire des requêtes
        self._requests_cache = {}  # request_id -> request_data
        self._priority_queues = {}  # model -> [request_ids ordonnés par priorité]
        self._batch_cache = {}     # batch_id -> batch_data
        
        # Verrous pour l'accès concurrent
        self._queue_lock = threading.RLock()
        
        # Intervalles de nettoyage (en secondes)
        self.clean_interval = 3600  # 1 heure
        
        # Pour le nettoyage périodique
        self._clean_thread = None
        self._stop_clean = threading.Event()
        
        # Valeurs de configuration par défaut
        self._config = {
            "max_batch_size": 5,  # Nombre maximum de requêtes dans un lot
            "batch_wait_time": 2.0,  # Temps d'attente max pour former un lot (secondes)
            "max_queue_size": 1000,  # Taille maximale de la file d'attente par modèle
            "request_timeout": 3600,  # Timeout des requêtes en secondes (1 heure)
            "default_priority": 10,  # Priorité par défaut (0-100, plus élevé = plus prioritaire)
            "enable_batching": True,  # Activer le regroupement des requêtes
            "cleanup_age_hours": 24   # Durée de conservation des requêtes terminées
        }
        
        # Statuts possibles des requêtes
        self.STATUS_PENDING = "pending"
        self.STATUS_PROCESSING = "processing"
        self.STATUS_COMPLETED = "completed"
        self.STATUS_FAILED = "failed"
        self.STATUS_CANCELED = "canceled"
        
        # Initialisation complète
        self._load_config_from_db()
        self._load_queue_from_db()
        self._start_clean_thread()
        self._initialized = True
        
        logger.info("QueueManager initialisé")
    
    def _load_config_from_db(self):
        """Charge la configuration depuis la base de données."""
        try:
            config_data = self.db.search_one("config", lambda q: q.key == "queue_settings")
            
            if config_data and "value" in config_data:
                # Fusionner les valeurs de configuration avec les valeurs par défaut
                self._config.update(config_data["value"])
                logger.info("Configuration de la file d'attente chargée depuis la base de données")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration de la file d'attente: {str(e)}")
    
    def _save_config_to_db(self):
        """Sauvegarde la configuration dans la base de données."""
        try:
            config_data = {
                "key": "queue_settings",
                "value": self._config,
                "updated_at": datetime.now().isoformat()
            }
            
            # Utiliser upsert pour créer ou mettre à jour
            self.db.upsert("config", config_data, lambda q: q.key == "queue_settings")
            logger.debug("Configuration de la file d'attente sauvegardée dans la base de données")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration de la file d'attente: {str(e)}")
    
    def _load_queue_from_db(self):
        """Charge l'état de la file d'attente depuis la base de données."""
        try:
            # Utiliser le SyncManager pour lire depuis la RAM si possible, sinon lire depuis la DB
            requests = self.sync_manager.read_from_ram("inference_queue")
            
            with self._queue_lock:
                self._requests_cache = {}
                self._priority_queues = {}
                
                # Traiter chaque requête
                for request in requests:
                    # Ignorer les requêtes terminées/échouées depuis trop longtemps
                    status = request.get("status")
                    if status in [self.STATUS_COMPLETED, self.STATUS_FAILED]:
                        completed_at = request.get("completed_at")
                        if completed_at:
                            try:
                                completed_time = datetime.fromisoformat(completed_at)
                                age_hours = (datetime.now() - completed_time).total_seconds() / 3600
                                
                                if age_hours > self._config["cleanup_age_hours"]:
                                    continue  # Ignorer cette requête trop ancienne
                            except (ValueError, TypeError):
                                pass
                    
                    # Ajouter la requête au cache
                    request_id = request.get("id")
                    if request_id:
                        self._requests_cache[request_id] = request
                        
                        # Ajouter à la file de priorité si en attente
                        model = request.get("model")
                        if status == self.STATUS_PENDING and model:
                            if model not in self._priority_queues:
                                self._priority_queues[model] = []
                            self._priority_queues[model].append(request_id)
                
                # Trier les files de priorité
                for model in self._priority_queues:
                    self._sort_priority_queue(model)
            
            # Charger également les lots existants
            batches = self.sync_manager.read_from_ram("inference_batches")
            for batch in batches:
                batch_id = batch.get("id")
                if batch_id:
                    self._batch_cache[batch_id] = batch
            
            logger.info(f"File d'attente chargée: {len(self._requests_cache)} requêtes, {len(self._priority_queues)} modèles, {len(self._batch_cache)} lots")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la file d'attente depuis la base de données: {str(e)}")
    
    def _start_clean_thread(self):
        """Démarre le thread de nettoyage périodique."""
        if self._clean_thread is None or not self._clean_thread.is_alive():
            self._stop_clean.clear()
            self._clean_thread = threading.Thread(
                target=self._periodic_clean,
                daemon=True
            )
            self._clean_thread.start()
            logger.info("Thread de nettoyage périodique démarré")
    
    def _periodic_clean(self):
        """Effectue des nettoyages périodiques de la file d'attente."""
        while not self._stop_clean.is_set():
            try:
                # Nettoyer les requêtes anciennes
                self.clean_old_requests()
                
            except Exception as e:
                logger.error(f"Erreur lors du nettoyage périodique: {str(e)}")
            
            # Attendre jusqu'au prochain nettoyage
            self._stop_clean.wait(self.clean_interval)
    
    def enqueue(self, request_data: Dict[str, Any]) -> str:
        """
        Ajoute une requête à la file d'attente.
        
        Args:
            request_data: Données de la requête
            
        Returns:
            ID de la requête générée
        """
        try:
            # Valider les données minimales requises
            if "model" not in request_data:
                raise ValueError("Le modèle doit être spécifié dans la requête")
                
            if "prompt" not in request_data and "messages" not in request_data:
                raise ValueError("Le prompt ou les messages doivent être spécifiés dans la requête")
            
            # Générer un ID unique
            request_id = str(uuid.uuid4())
            
            # Compléter les données de la requête
            request = {
                "id": request_id,
                "model": request_data["model"],
                "status": self.STATUS_PENDING,
                "created_at": datetime.now().isoformat(),
                "batch_id": None,
                "server_assigned": None
            }
            
            # Copier les champs essentiels
            for field in ["prompt", "messages", "options", "priority", "client_id"]:
                if field in request_data:
                    request[field] = request_data[field]
            
            # Définir la priorité si non spécifiée
            if "priority" not in request:
                request["priority"] = self._config["default_priority"]
            
            # Persister dans la base de données
            doc_id = self.sync_manager.write_and_sync("inference_queue", request)
            
            # Mettre à jour le cache en mémoire
            with self._queue_lock:
                self._requests_cache[request_id] = request
                
                # Ajouter à la file de priorité
                model = request["model"]
                if model not in self._priority_queues:
                    self._priority_queues[model] = []
                self._priority_queues[model].append(request_id)
                
                # Trier la file de priorité
                self._sort_priority_queue(model)
                
                # Vérifier la taille maximale de la file
                if len(self._priority_queues[model]) > self._config["max_queue_size"]:
                    # Supprimer les requêtes les moins prioritaires en excès
                    excess = len(self._priority_queues[model]) - self._config["max_queue_size"]
                    removed_ids = self._priority_queues[model][-excess:]
                    self._priority_queues[model] = self._priority_queues[model][:-excess]
                    
                    # Mettre à jour le statut des requêtes supprimées
                    for removed_id in removed_ids:
                        if removed_id in self._requests_cache:
                            self._requests_cache[removed_id]["status"] = self.STATUS_CANCELED
                            self._requests_cache[removed_id]["canceled_reason"] = "queue_overflow"
                            
                            # Persister le changement
                            self.sync_manager.write_and_sync("inference_queue", self._requests_cache[removed_id])
                            
                    logger.warning(f"File d'attente du modèle {model} débordée, {excess} requêtes annulées")
            
            logger.info(f"Requête {request_id} ajoutée à la file d'attente pour le modèle {model}")
            return request_id
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout à la file d'attente: {str(e)}")
            raise
    
    def dequeue(self, batch_size: int = 1, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère des requêtes à traiter.
        
        Args:
            batch_size: Nombre maximum de requêtes à récupérer
            model: Modèle spécifique (optionnel, tous les modèles si None)
            
        Returns:
            Liste des requêtes à traiter
        """
        results = []
        batch_id = None
        
        try:
            with self._queue_lock:
                # Déterminer les modèles à traiter
                models = [model] if model else list(self._priority_queues.keys())
                
                for current_model in models:
                    # Vérifier si ce modèle a des requêtes en attente
                    if current_model not in self._priority_queues or not self._priority_queues[current_model]:
                        continue
                    
                    # Récupérer les requêtes les plus prioritaires
                    request_ids = self._priority_queues[current_model][:batch_size]
                    
                    if not request_ids:
                        continue
                        
                    # Générer un ID de lot pour ces requêtes
                    batch_id = f"batch-{uuid.uuid4()}"
                    current_time = datetime.now().isoformat()
                    
                    # Créer et persister les informations du lot
                    batch_data = {
                        "id": batch_id,
                        "model": current_model,
                        "request_count": len(request_ids),
                        "request_ids": request_ids.copy(),
                        "created_at": current_time,
                        "started_at": current_time,
                        "completed_at": None,
                        "status": "processing"
                    }
                    
                    self.sync_manager.write_and_sync("inference_batches", batch_data)
                    self._batch_cache[batch_id] = batch_data
                    
                    # Mettre à jour et récupérer les requêtes
                    for request_id in request_ids:
                        if request_id in self._requests_cache:
                            request = self._requests_cache[request_id].copy()
                            
                            # Mettre à jour le statut et les informations du lot
                            request["status"] = self.STATUS_PROCESSING
                            request["batch_id"] = batch_id
                            request["started_at"] = current_time
                            
                            # Persister les changements
                            self.sync_manager.write_and_sync("inference_queue", request)
                            
                            # Mettre à jour le cache
                            self._requests_cache[request_id] = request
                            
                            # Ajouter à la liste des résultats
                            results.append(request)
                    
                    # Retirer les requêtes de la file d'attente
                    self._priority_queues[current_model] = self._priority_queues[current_model][len(request_ids):]
                    
                    # Une fois qu'on a trouvé un modèle avec des requêtes, on s'arrête
                    break
            
            logger.info(f"Requêtes déqueues: {len(results)}, batch_id: {batch_id}")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du déqueueage des requêtes: {str(e)}")
            
            # Essayer de nettoyer en cas d'erreur
            if batch_id and batch_id in self._batch_cache:
                try:
                    self._batch_cache[batch_id]["status"] = "failed"
                    self.sync_manager.write_and_sync("inference_batches", self._batch_cache[batch_id])
                except:
                    pass
                    
            return []
    
    def update_request_status(self, request_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Met à jour le statut d'une requête.
        
        Args:
            request_id: ID de la requête
            status: Nouveau statut
            result: Résultat de l'inférence (optionnel)
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        try:
            with self._queue_lock:
                # Vérifier si la requête existe
                if request_id not in self._requests_cache:
                    logger.warning(f"Tentative de mise à jour d'une requête inconnue: {request_id}")
                    return False
                
                # Récupérer la requête
                request = self._requests_cache[request_id]
                current_status = request.get("status")
                
                # Valider la transition de statut
                valid_transitions = {
                    self.STATUS_PENDING: [self.STATUS_PROCESSING, self.STATUS_CANCELED, self.STATUS_FAILED],
                    self.STATUS_PROCESSING: [self.STATUS_COMPLETED, self.STATUS_FAILED, self.STATUS_CANCELED],
                    self.STATUS_COMPLETED: [],  # État final
                    self.STATUS_FAILED: [],     # État final
                    self.STATUS_CANCELED: []    # État final
                }
                
                if status not in valid_transitions.get(current_status, []):
                    logger.warning(f"Transition de statut invalide pour la requête {request_id}: {current_status} -> {status}")
                    return False
                
                # Mettre à jour le statut
                request["status"] = status
                current_time = datetime.now().isoformat()
                
                if status == self.STATUS_COMPLETED or status == self.STATUS_FAILED:
                    request["completed_at"] = current_time
                
                # Ajouter le résultat si fourni
                if result is not None:
                    request["result"] = result
                
                # Mettre à jour et persister le lot associé
                batch_id = request.get("batch_id")
                if batch_id and batch_id in self._batch_cache:
                    batch = self._batch_cache[batch_id]
                    
                    # Vérifier si toutes les requêtes du lot sont terminées
                    all_completed = True
                    for req_id in batch.get("request_ids", []):
                        if req_id in self._requests_cache:
                            req_status = self._requests_cache[req_id].get("status")
                            if req_status not in [self.STATUS_COMPLETED, self.STATUS_FAILED, self.STATUS_CANCELED]:
                                all_completed = False
                                break
                    
                    if all_completed:
                        batch["status"] = "completed"
                        batch["completed_at"] = current_time
                        self.sync_manager.write_and_sync("inference_batches", batch)
                
                # Persister les changements
                self.sync_manager.write_and_sync("inference_queue", request)
                
            logger.info(f"Statut de la requête {request_id} mis à jour: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du statut de la requête {request_id}: {str(e)}")
            return False
    
    def batch_similar_requests(self) -> Dict[str, List[str]]:
        """
        Regroupe les requêtes similaires pour optimiser le traitement.
        
        Returns:
            Dictionnaire avec les IDs de lots comme clés et listes d'IDs de requêtes comme valeurs
        """
        batches = {}
        
        if not self._config["enable_batching"]:
            return batches
            
        try:
            with self._queue_lock:
                # Traiter chaque modèle séparément
                for model, request_ids in self._priority_queues.items():
                    if not request_ids:
                        continue
                    
                    # Regrouper les requêtes par similarité
                    batch_groups = {}  # hash -> [request_ids]
                    
                    for request_id in request_ids:
                        if request_id not in self._requests_cache:
                            continue
                            
                        request = self._requests_cache[request_id]
                        
                        # Générer une clé de hachage pour les propriétés de la requête
                        # qui déterminent si elle peut être regroupée avec d'autres
                        hash_components = [
                            model,
                            str(request.get("options", {}))
                        ]
                        
                        batch_key = "_".join(hash_components)
                        
                        if batch_key not in batch_groups:
                            batch_groups[batch_key] = []
                            
                        batch_groups[batch_key].append(request_id)
                    
                    # Créer des lots pour les groupes qui dépassent un certain seuil
                    for batch_key, group_ids in batch_groups.items():
                        # Ne créer un lot que si le groupe contient au moins 2 requêtes
                        if len(group_ids) < 2:
                            continue
                            
                        # Limiter au max_batch_size
                        if len(group_ids) > self._config["max_batch_size"]:
                            group_ids = group_ids[:self._config["max_batch_size"]]
                            
                        # Créer un ID de lot
                        batch_id = f"batch-{uuid.uuid4()}"
                        batches[batch_id] = group_ids.copy()
                        
                        # Enregistrer les informations du lot
                        batch_data = {
                            "id": batch_id,
                            "model": model,
                            "request_count": len(group_ids),
                            "request_ids": group_ids.copy(),
                            "created_at": datetime.now().isoformat(),
                            "started_at": None,
                            "completed_at": None,
                            "status": "pending"
                        }
                        
                        # Persister le lot
                        self.sync_manager.write_and_sync("inference_batches", batch_data)
                        self._batch_cache[batch_id] = batch_data
                        
                        # Mettre à jour chaque requête avec l'ID du lot
                        for req_id in group_ids:
                            if req_id in self._requests_cache:
                                self._requests_cache[req_id]["batch_id"] = batch_id
                                self.sync_manager.write_and_sync("inference_queue", self._requests_cache[req_id])
                        
                        logger.info(f"Lot {batch_id} créé avec {len(group_ids)} requêtes similaires pour le modèle {model}")
            
            return batches
            
        except Exception as e:
            logger.error(f"Erreur lors du regroupement des requêtes similaires: {str(e)}")
            return {}
    
    def clean_old_requests(self, max_age_hours: Optional[int] = None) -> int:
        """
        Nettoie les requêtes anciennes ou traitées.
        
        Args:
            max_age_hours: Âge maximal des requêtes à conserver (optionnel)
            
        Returns:
            Nombre de requêtes supprimées
        """
        if max_age_hours is None:
            max_age_hours = self._config["cleanup_age_hours"]
            
        removed_count = 0
        
        try:
            # Calculer la date limite
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cutoff_iso = cutoff_time.isoformat()
            
            with self._queue_lock:
                # Identifier les requêtes à supprimer
                to_remove = []
                
                for request_id, request in list(self._requests_cache.items()):
                    # Nettoyer les requêtes terminées anciennes
                    if request["status"] in [self.STATUS_COMPLETED, self.STATUS_FAILED, self.STATUS_CANCELED]:
                        completed_at = request.get("completed_at")
                        if completed_at and completed_at < cutoff_iso:
                            to_remove.append(request_id)
                    
                    # Nettoyer les requêtes en attente ou en traitement depuis trop longtemps (timeout)
                    elif request["status"] in [self.STATUS_PENDING, self.STATUS_PROCESSING]:
                        created_at = request.get("created_at")
                        if created_at and created_at < cutoff_iso:
                            # Mettre à jour le statut avant de supprimer
                            request["status"] = self.STATUS_FAILED
                            request["failed_reason"] = "timeout"
                            request["completed_at"] = datetime.now().isoformat()
                            
                            # Persister le changement de statut
                            self.sync_manager.write_and_sync("inference_queue", request)
                            
                            # Ajouter à la liste à supprimer
                            to_remove.append(request_id)
                
                # Supprimer les requêtes identifiées
                for request_id in to_remove:
                    # Supprimer de la file de priorité
                    model = self._requests_cache[request_id].get("model")
                    if model in self._priority_queues and request_id in self._priority_queues[model]:
                        self._priority_queues[model].remove(request_id)
                    
                    # Supprimer du cache
                    del self._requests_cache[request_id]
                    
                    # Supprimer de la base de données (utiliser une fonction qui recherche par id)
                    query_func = lambda q: q.id == request_id
                    self.sync_manager.delete_and_sync("inference_queue", None, query_func)
                    removed_count += 1
                
                # Nettoyer également les lots anciens
                for batch_id, batch in list(self._batch_cache.items()):
                    completed_at = batch.get("completed_at")
                    if completed_at and completed_at < cutoff_iso:
                        # Supprimer du cache
                        del self._batch_cache[batch_id]
                        
                        # Supprimer de la base de données
                        query_func = lambda q: q.id == batch_id
                        self.sync_manager.delete_and_sync("inference_batches", None, query_func)
            
            if removed_count > 0:
                logger.info(f"Nettoyage terminé: {removed_count} requêtes anciennes supprimées")
                
            return removed_count
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des requêtes anciennes: {str(e)}")
            return 0
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Obtient des statistiques sur la file d'attente actuelle.
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            "total_requests": 0,
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "canceled": 0,
            "models": {},
            "batches": {
                "total": len(self._batch_cache),
                "pending": 0,
                "processing": 0,
                "completed": 0
            }
        }
        
        try:
            with self._queue_lock:
                # Compiler les statistiques des requêtes
                for request in self._requests_cache.values():
                    status = request.get("status")
                    model = request.get("model")
                    
                    # Incrémenter les compteurs globaux
                    stats["total_requests"] += 1
                    if status in stats:
                        stats[status] += 1
                    
                    # Incrémenter les compteurs par modèle
                    if model:
                        if model not in stats["models"]:
                            stats["models"][model] = {
                                "total": 0,
                                "pending": 0,
                                "processing": 0,
                                "completed": 0,
                                "failed": 0,
                                "canceled": 0
                            }
                        
                        stats["models"][model]["total"] += 1
                        if status in stats["models"][model]:
                            stats["models"][model][status] += 1
                
                # Compiler les statistiques des lots
                for batch in self._batch_cache.values():
                    status = batch.get("status")
                    if status in stats["batches"]:
                        stats["batches"][status] += 1
                
                # Ajouter des informations sur la configuration
                stats["config"] = self._config.copy()
                
                # Ajouter des informations temporelles
                stats["timestamp"] = datetime.now().isoformat()
                
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques de la file d'attente: {str(e)}")
            return {"error": str(e)}
    
    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'une requête spécifique.
        
        Args:
            request_id: ID de la requête
            
        Returns:
            Informations de la requête ou None si non trouvée
        """
        with self._queue_lock:
            if request_id in self._requests_cache:
                return self._requests_cache[request_id].copy()
            
        return None
    
    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'un lot spécifique.
        
        Args:
            batch_id: ID du lot
            
        Returns:
            Informations du lot ou None si non trouvé
        """
        with self._queue_lock:
            if batch_id in self._batch_cache:
                return self._batch_cache[batch_id].copy()
            
        return None
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Met à jour la configuration de la file d'attente.
        
        Args:
            new_config: Nouvelles valeurs de configuration
            
        Returns:
            Configuration mise à jour
        """
        try:
            # Fusionner la nouvelle configuration avec l'existante
            self._config.update(new_config)
            
            # Persister la configuration
            self._save_config_to_db()
            
            logger.info(f"Configuration de la file d'attente mise à jour: {new_config}")
            return self._config.copy()
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la configuration: {str(e)}")
            return {"error": str(e)}
    
    def _sort_priority_queue(self, model: str) -> None:
        """
        Trie une file de priorité pour un modèle spécifique.
        
        Args:
            model: Nom du modèle dont la file doit être triée
        """
        if model not in self._priority_queues:
            return
            
        # Trier par priorité décroissante, puis par date de création croissante (FIFO à priorité égale)
        self._priority_queues[model].sort(key=lambda rid: 
            (-self._requests_cache.get(rid, {}).get("priority", 0),
             self._requests_cache.get(rid, {}).get("created_at", "")))
    
    def cancel_request(self, request_id: str, reason: str = "user_cancelled") -> bool:
        """
        Annule une requête en attente.
        
        Args:
            request_id: ID de la requête à annuler
            reason: Raison de l'annulation
            
        Returns:
            True si l'annulation a réussi, False sinon
        """
        try:
            with self._queue_lock:
                # Vérifier si la requête existe et est annulable
                if request_id not in self._requests_cache:
                    logger.warning(f"Tentative d'annulation d'une requête inconnue: {request_id}")
                    return False
                
                request = self._requests_cache[request_id]
                if request["status"] not in [self.STATUS_PENDING, self.STATUS_PROCESSING]:
                    logger.warning(f"Tentative d'annulation d'une requête non annulable: {request_id} (statut: {request['status']})")
                    return False
                
                # Mettre à jour le statut
                request["status"] = self.STATUS_CANCELED
                request["canceled_reason"] = reason
                request["completed_at"] = datetime.now().isoformat()
                
                # Persister les changements
                self.sync_manager.write_and_sync("inference_queue", request)
                
                # Retirer de la file de priorité si en attente
                model = request.get("model")
                if model in self._priority_queues and request["status"] == self.STATUS_PENDING:
                    if request_id in self._priority_queues[model]:
                        self._priority_queues[model].remove(request_id)
            
            logger.info(f"Requête {request_id} annulée: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de la requête {request_id}: {str(e)}")
            return False
    
    def get_pending_count(self, model: Optional[str] = None) -> int:
        """
        Obtient le nombre de requêtes en attente.
        
        Args:
            model: Modèle spécifique (optionnel)
            
        Returns:
            Nombre de requêtes en attente
        """
        with self._queue_lock:
            if model:
                # Compter pour un modèle spécifique
                return len(self._priority_queues.get(model, []))
            else:
                # Compter pour tous les modèles
                return sum(len(queue) for queue in self._priority_queues.values())
    
    def shutdown(self) -> None:
        """
        Arrête proprement le gestionnaire de file d'attente et persiste son état.
        """
        logger.info("Arrêt du QueueManager")
        
        try:
            # Arrêter le thread de nettoyage périodique
            if self._clean_thread and self._clean_thread.is_alive():
                self._stop_clean.set()
                self._clean_thread.join(timeout=2.0)
                self._clean_thread = None
            
            # Persister la configuration une dernière fois
            self._save_config_to_db()
            
            # Marquer toutes les requêtes en traitement comme échouées
            with self._queue_lock:
                for request_id, request in list(self._requests_cache.items()):
                    if request["status"] == self.STATUS_PROCESSING:
                        request["status"] = self.STATUS_FAILED
                        request["failed_reason"] = "shutdown"
                        request["completed_at"] = datetime.now().isoformat()
                        
                        # Persister le changement de statut
                        self.sync_manager.write_and_sync("inference_queue", request)
                
                # Marquer tous les lots en traitement comme échoués
                for batch_id, batch in list(self._batch_cache.items()):
                    if batch["status"] == "processing":
                        batch["status"] = "failed"
                        batch["completed_at"] = datetime.now().isoformat()
                        
                        # Persister le changement de statut
                        self.sync_manager.write_and_sync("inference_batches", batch)
                
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du QueueManager: {str(e)}")


# Fonction singleton pour accéder au gestionnaire de file d'attente
def get_queue_manager() -> QueueManager:
    """
    Récupère l'instance unique du gestionnaire de file d'attente.
    
    Returns:
        Instance du QueueManager
    """
    return QueueManager()