
"""
Module de gestion centralisée des modèles à travers le cluster.

Ce module fournit un système de registre pour la gestion des modèles,
permettant leur distribution intelligente, leur suivi et leur synchronisation
entre les différents nœuds du cluster.
"""

import logging
import threading
import time
import json
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime, timedelta
import random

from osync.proxy.cluster.manager import get_cluster_manager
from osync.sync.client import OllamaClient
from osync.proxy.db.sync_manager import get_sync_manager
from osync.proxy.db.database import get_db

# Configuration du logging
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Système centralisé de gestion des modèles à travers le cluster.
    
    Cette classe offre des fonctionnalités avancées pour gérer la distribution
    des modèles, suivre leur utilisation et assurer leur disponibilité.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Implémentation du pattern Singleton pour ModelRegistry."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialise le registre des modèles."""
        # Éviter la réinitialisation du singleton
        if self._initialized:
            return
            
        self.cluster_manager = get_cluster_manager()
        self.sync_manager = get_sync_manager()
        self.db = get_db()
        
        # Cache local des modèles et de leurs métadonnées
        self._models_cache = {}  # model_name -> metadata
        self._model_servers = {}  # model_name -> [server1, server2, ...]
        
        # Verrous pour l'accès concurrent
        self._models_lock = threading.RLock()
        
        # Intervalle pour la vérification périodique en secondes
        self.check_interval = 300  # 5 minutes
        
        # Pour la vérification périodique
        self._check_thread = None
        self._stop_check = threading.Event()
        
        # Initialisation complète
        self._load_models_from_db()
        self._start_check_thread()
        self._initialized = True
        
        logger.info("ModelRegistry initialisé")
    
    def _load_models_from_db(self):
        """Charge les informations des modèles depuis la base de données."""
        try:
            # Utiliser le SyncManager pour lire depuis la RAM si possible
            models = self.sync_manager.read_from_ram("models")
            
            with self._models_lock:
                self._models_cache = {}
                self._model_servers = {}
                
                for model in models:
                    model_name = model.get("name")
                    if model_name:
                        self._models_cache[model_name] = model
                        self._model_servers[model_name] = set(model.get("servers", []))
            
            logger.info(f"Chargement des modèles depuis la base de données: {len(models)} modèles trouvés")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles depuis la base de données: {str(e)}")
    
    def _start_check_thread(self):
        """Démarre le thread de vérification périodique."""
        if self._check_thread is None or not self._check_thread.is_alive():
            self._stop_check.clear()
            self._check_thread = threading.Thread(
                target=self._periodic_check,
                daemon=True
            )
            self._check_thread.start()
            logger.info("Thread de vérification périodique des modèles démarré")
    
    def _periodic_check(self):
        """Effectue des vérifications périodiques du registre des modèles."""
        while not self._stop_check.is_set():
            try:
                # Synchroniser avec les nœuds et mettre à jour la base de données
                self.synchronize_models()
                
                # Optimiser la distribution des modèles si nécessaire
                self.optimize_model_distribution()
                
            except Exception as e:
                logger.error(f"Erreur lors de la vérification périodique des modèles: {str(e)}")
            
            # Attendre jusqu'à la prochaine vérification
            self._stop_check.wait(self.check_interval)
    
    def pull_model(self, model_name: str, target_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Pull un modèle sur les nœuds spécifiés ou auto-sélectionnés.
        
        Args:
            model_name: Nom du modèle à récupérer
            target_nodes: Liste des nœuds cibles (optionnel)
            
        Returns:
            Dictionnaire avec les résultats de l'opération pour chaque nœud
        """
        logger.info(f"Démarrage du pull du modèle {model_name} sur les nœuds: {target_nodes if target_nodes else 'auto-sélection'}")
        
        results = {}
        
        try:
            # Si aucun nœud n'est spécifié, sélectionner automatiquement en fonction des ressources
            if not target_nodes:
                target_nodes = self._select_optimal_nodes_for_model(model_name)
                logger.info(f"Nœuds auto-sélectionnés pour {model_name}: {target_nodes}")
            
            # Pull le modèle sur chaque nœud
            for node in target_nodes:
                try:
                    # Créer un client pour ce nœud
                    client = OllamaClient(node)
                    
                    # Appeler l'API de pull sur le nœud
                    success, message = client.pull_model(model_name)
                    
                    results[node] = {
                        "success": success,
                        "message": message
                    }
                    
                    # Mettre à jour le registre si succès
                    if success:
                        self._update_model_location(model_name, node, True)
                        logger.info(f"Modèle {model_name} pullé avec succès sur le nœud {node}")
                    else:
                        logger.warning(f"Échec du pull du modèle {model_name} sur le nœud {node}: {message}")
                        
                except Exception as e:
                    error_msg = f"Erreur lors du pull du modèle {model_name} sur le nœud {node}: {str(e)}"
                    logger.error(error_msg)
                    results[node] = {
                        "success": False,
                        "message": error_msg
                    }
            
            # Mettre à jour la base de données avec les résultats
            self._persist_model_changes(model_name)
            
        except Exception as e:
            error_msg = f"Erreur globale lors du pull du modèle {model_name}: {str(e)}"
            logger.error(error_msg)
            results["global_error"] = error_msg
        
        return results
    
    def remove_model(self, model_name: str, target_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Supprime un modèle des nœuds spécifiés.
        
        Args:
            model_name: Nom du modèle à supprimer
            target_nodes: Liste des nœuds cibles (optionnel, tous les nœuds si non spécifié)
            
        Returns:
            Dictionnaire avec les résultats de l'opération pour chaque nœud
        """
        results = {}
        
        try:
            # Si aucun nœud n'est spécifié, utiliser tous les nœuds qui ont ce modèle
            if not target_nodes:
                with self._models_lock:
                    target_nodes = list(self._model_servers.get(model_name, []))
            
            # Supprimer le modèle de chaque nœud
            for node in target_nodes:
                try:
                    # Créer un client pour ce nœud
                    client = OllamaClient(node)
                    
                    # Appeler l'API de suppression sur le nœud
                    success, message = client.remove_model(model_name)
                    
                    results[node] = {
                        "success": success,
                        "message": message
                    }
                    
                    # Mettre à jour le registre si succès
                    if success:
                        self._update_model_location(model_name, node, False)
                        logger.info(f"Modèle {model_name} supprimé avec succès du nœud {node}")
                    else:
                        logger.warning(f"Échec de la suppression du modèle {model_name} du nœud {node}: {message}")
                        
                except Exception as e:
                    error_msg = f"Erreur lors de la suppression du modèle {model_name} du nœud {node}: {str(e)}"
                    logger.error(error_msg)
                    results[node] = {
                        "success": False,
                        "message": error_msg
                    }
            
            # Mettre à jour la base de données avec les résultats
            self._persist_model_changes(model_name)
            
        except Exception as e:
            error_msg = f"Erreur globale lors de la suppression du modèle {model_name}: {str(e)}"
            logger.error(error_msg)
            results["global_error"] = error_msg
        
        return results
    
    def synchronize_models(self) -> Dict[str, Any]:
        """
        Synchronise la disponibilité des modèles entre tous les nœuds.
        Cette méthode interroge chaque nœud pour connaître ses modèles disponibles
        et met à jour le registre central.
        
        Returns:
            Dictionnaire avec les résultats de la synchronisation
        """
        results = {
            "success": True,
            "nodes_checked": 0,
            "models_found": 0,
            "errors": [],
            "new_models": [],
            "removed_models": []
        }
        
        try:
            # Obtenir la liste des serveurs en bonne santé
            healthy_servers = self.cluster_manager.get_healthy_servers()
            results["nodes_checked"] = len(healthy_servers)
            
            # Pour suivre les modèles découverts
            all_discovered_models = set()
            model_to_servers = {}
            
            # Interroger chaque serveur pour ses modèles
            for server in healthy_servers:
                try:
                    # Créer un client pour ce serveur
                    client = OllamaClient(server)
                    
                    # Obtenir la liste des modèles
                    models = client.list_models()
                    
                    # Traiter chaque modèle découvert
                    for model in models:
                        model_name = model.get("name")
                        if model_name:
                            all_discovered_models.add(model_name)
                            
                            # Enregistrer ce serveur pour ce modèle
                            if model_name not in model_to_servers:
                                model_to_servers[model_name] = set()
                            model_to_servers[model_name].add(server)
                            
                            # Enrichir les métadonnées du modèle si possible
                            if model_name not in self._models_cache:
                                self._models_cache[model_name] = model
                                results["new_models"].append(model_name)
                            else:
                                # Mettre à jour les métadonnées existantes avec les nouvelles informations
                                for key, value in model.items():
                                    if key != "servers" and value is not None:
                                        self._models_cache[model_name][key] = value
                    
                except Exception as e:
                    error_msg = f"Erreur lors de la synchronisation avec le serveur {server}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    results["success"] = False
            
            # Mettre à jour notre registre avec les serveurs découverts pour chaque modèle
            with self._models_lock:
                # Identifier les modèles qui ont disparu de tous les serveurs
                for model_name in list(self._model_servers.keys()):
                    if model_name not in all_discovered_models:
                        results["removed_models"].append(model_name)
                        del self._model_servers[model_name]
                        if model_name in self._models_cache:
                            del self._models_cache[model_name]
                
                # Mettre à jour la cartographie modèle-serveurs
                for model_name, servers in model_to_servers.items():
                    self._model_servers[model_name] = servers
            
            # Persister les changements dans la base de données
            self._persist_all_models()
            
            results["models_found"] = len(all_discovered_models)
            logger.info(f"Synchronisation des modèles terminée: {results['models_found']} modèles sur {results['nodes_checked']} nœuds")
            
        except Exception as e:
            error_msg = f"Erreur globale lors de la synchronisation des modèles: {str(e)}"
            logger.error(error_msg)
            results["success"] = False
            results["errors"].append(error_msg)
        
        return results
    
    def get_model_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère le statut du/des modèle(s) dans le cluster.
        
        Args:
            model_name: Nom du modèle spécifique ou None pour tous les modèles
            
        Returns:
            Dictionnaire avec les statuts des modèles
        """
        result = {}
        
        try:
            if model_name:
                # Récupérer le statut d'un modèle spécifique
                with self._models_lock:
                    if model_name in self._models_cache:
                        model_info = self._models_cache[model_name].copy()
                        model_info["servers"] = list(self._model_servers.get(model_name, []))
                        result = model_info
                    else:
                        result = {"error": f"Modèle {model_name} non trouvé"}
            else:
                # Récupérer le statut de tous les modèles
                models = []
                with self._models_lock:
                    for name, info in self._models_cache.items():
                        model_info = info.copy()
                        model_info["servers"] = list(self._model_servers.get(name, []))
                        models.append(model_info)
                
                result = {"models": models, "count": len(models)}
        
        except Exception as e:
            error_msg = f"Erreur lors de la récupération du statut des modèles: {str(e)}"
            logger.error(error_msg)
            result = {"error": error_msg}
        
        return result
    
    def optimize_model_distribution(self) -> Dict[str, Any]:
        """
        Optimise la distribution des modèles selon la charge et les ressources.
        Cette méthode analyse l'utilisation actuelle des modèles et peut décider
        de déplacer ou dupliquer des modèles pour optimiser les performances.
        
        Returns:
            Dictionnaire avec les résultats de l'optimisation
        """
        results = {
            "success": True,
            "actions_taken": [],
            "errors": []
        }
        
        try:
            # 1. Identifier les modèles fréquemment utilisés mais peu distribués
            popular_models = self._identify_popular_models()
            
            # 2. Identifier les serveurs sous-utilisés avec des ressources disponibles
            available_servers = self._identify_available_servers()
            
            # 3. Pour chaque modèle populaire, déterminer s'il faut le distribuer davantage
            for model in popular_models:
                model_name = model["name"]
                current_servers = set(self._model_servers.get(model_name, []))
                
                # Si le modèle n'est pas assez distribué et qu'il y a des serveurs disponibles
                if len(current_servers) < model.get("optimal_server_count", 2) and available_servers:
                    # Choisir un serveur disponible
                    target_server = available_servers.pop(0) if available_servers else None
                    
                    if target_server and target_server not in current_servers:
                        # Tenter de déployer le modèle sur ce serveur
                        action = {
                            "action": "deploy",
                            "model": model_name,
                            "server": target_server,
                            "reason": "optimize_distribution"
                        }
                        
                        try:
                            # Pull le modèle sur le serveur cible
                            result = self.pull_model(model_name, [target_server])
                            
                            if result.get(target_server, {}).get("success", False):
                                action["status"] = "success"
                                logger.info(f"Distribution optimisée: modèle {model_name} déployé sur {target_server}")
                            else:
                                action["status"] = "failed"
                                action["error"] = result.get(target_server, {}).get("message", "Unknown error")
                                
                            results["actions_taken"].append(action)
                            
                        except Exception as e:
                            error_msg = f"Erreur lors du déploiement du modèle {model_name} sur {target_server}: {str(e)}"
                            logger.error(error_msg)
                            action["status"] = "failed"
                            action["error"] = error_msg
                            results["actions_taken"].append(action)
                            results["errors"].append(error_msg)
            
            # 4. Identifier les modèles peu utilisés sur des serveurs surchargés
            underused_models = self._identify_underused_models()
            
            # 5. Pour chaque modèle peu utilisé, déterminer s'il faut le supprimer de certains serveurs
            for model in underused_models:
                model_name = model["name"]
                current_servers = list(self._model_servers.get(model_name, []))
                
                # Si le modèle est trop distribué par rapport à son utilisation
                if len(current_servers) > model.get("optimal_server_count", 1):
                    # Identifier les serveurs les plus chargés ou les moins appropriés
                    servers_to_remove = self._select_servers_for_cleanup(model_name, current_servers, 
                                                                         max(1, len(current_servers) - model.get("optimal_server_count", 1)))
                    
                    for server in servers_to_remove:
                        action = {
                            "action": "remove",
                            "model": model_name,
                            "server": server,
                            "reason": "optimize_distribution"
                        }
                        
                        try:
                            # Supprimer le modèle du serveur
                            result = self.remove_model(model_name, [server])
                            
                            if result.get(server, {}).get("success", False):
                                action["status"] = "success"
                                logger.info(f"Distribution optimisée: modèle {model_name} supprimé de {server}")
                            else:
                                action["status"] = "failed"
                                action["error"] = result.get(server, {}).get("message", "Unknown error")
                                
                            results["actions_taken"].append(action)
                            
                        except Exception as e:
                            error_msg = f"Erreur lors de la suppression du modèle {model_name} de {server}: {str(e)}"
                            logger.error(error_msg)
                            action["status"] = "failed"
                            action["error"] = error_msg
                            results["actions_taken"].append(action)
                            results["errors"].append(error_msg)
            
        except Exception as e:
            error_msg = f"Erreur globale lors de l'optimisation de la distribution des modèles: {str(e)}"
            logger.error(error_msg)
            results["success"] = False
            results["errors"].append(error_msg)
        
        return results
    
    def _identify_popular_models(self) -> List[Dict[str, Any]]:
        """
        Identifie les modèles populaires qui pourraient bénéficier d'une distribution plus large.
        
        Returns:
            Liste de modèles populaires avec des métadonnées additionnelles
        """
        popular_models = []
        
        try:
            # Récupérer les statistiques d'utilisation des modèles
            stats = self.cluster_manager.get_model_stats()
            
            with self._models_lock:
                for model_name, metadata in self._models_cache.items():
                    usage_count = metadata.get("usage_count", 0)
                    
                    # Calculer un score de popularité basé sur l'utilisation récente
                    # et la fréquence d'utilisation
                    popularity_score = 0
                    
                    # Score basé sur le nombre total d'utilisations
                    popularity_score += min(usage_count / 10, 10)  # Capé à 10 pour éviter les valeurs extrêmes
                    
                    # Score basé sur l'utilisation récente
                    last_used = metadata.get("last_used")
                    if last_used:
                        try:
                            last_used_date = datetime.fromisoformat(last_used)
                            days_since_use = (datetime.now() - last_used_date).days
                            
                            # Plus récent = score plus élevé
                            if days_since_use == 0:  # Utilisé aujourd'hui
                                popularity_score += 5
                            elif days_since_use <= 7:  # Utilisé cette semaine
                                popularity_score += 3
                            elif days_since_use <= 30:  # Utilisé ce mois
                                popularity_score += 1
                        except (ValueError, TypeError):
                            # Ignorer si le format de date est incorrect
                            pass
                    
                    # Déterminer le nombre optimal de serveurs en fonction du score de popularité
                    optimal_server_count = max(1, min(int(popularity_score / 3), 4))  # Entre 1 et 4 serveurs
                    
                    model_info = metadata.copy()
                    model_info["name"] = model_name
                    model_info["popularity_score"] = popularity_score
                    model_info["optimal_server_count"] = optimal_server_count
                    model_info["current_server_count"] = len(self._model_servers.get(model_name, []))
                    
                    # Si le modèle est populaire et sous-distribué
                    if popularity_score >= 5 and model_info["current_server_count"] < optimal_server_count:
                        popular_models.append(model_info)
            
            # Trier par importance (popularité * sous-distribution)
            popular_models.sort(
                key=lambda m: (m["popularity_score"] * (m["optimal_server_count"] - m["current_server_count"])),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'identification des modèles populaires: {str(e)}")
        
        return popular_models
    
    def _identify_underused_models(self) -> List[Dict[str, Any]]:
        """
        Identifie les modèles peu utilisés qui pourraient être supprimés de certains serveurs.
        
        Returns:
            Liste de modèles peu utilisés avec des métadonnées additionnelles
        """
        underused_models = []
        
        try:
            with self._models_lock:
                for model_name, metadata in self._models_cache.items():
                    usage_count = metadata.get("usage_count", 0)
                    
                    # Calculer un score d'utilisation
                    usage_score = 0
                    
                    # Score basé sur le nombre total d'utilisations
                    usage_score += min(usage_count / 10, 10)  # Capé à 10
                    
                    # Score basé sur l'utilisation récente
                    last_used = metadata.get("last_used")
                    if last_used:
                        try:
                            last_used_date = datetime.fromisoformat(last_used)
                            days_since_use = (datetime.now() - last_used_date).days
                            
                            # Plus récent = score plus élevé
                            if days_since_use == 0:  # Utilisé aujourd'hui
                                usage_score += 5
                            elif days_since_use <= 7:  # Utilisé cette semaine
                                usage_score += 3
                            elif days_since_use <= 30:  # Utilisé ce mois
                                usage_score += 1
                        except (ValueError, TypeError):
                            # Ignorer si le format de date est incorrect
                            pass
                    
                    # Déterminer le nombre optimal de serveurs en fonction du score d'utilisation
                    optimal_server_count = max(1, min(int(usage_score / 3), 3))  # Entre 1 et 3 serveurs
                    
                    model_info = metadata.copy()
                    model_info["name"] = model_name
                    model_info["usage_score"] = usage_score
                    model_info["optimal_server_count"] = optimal_server_count
                    model_info["current_server_count"] = len(self._model_servers.get(model_name, []))
                    
                    # Si le modèle est peu utilisé et sur-distribué
                    if usage_score < 5 and model_info["current_server_count"] > optimal_server_count:
                        underused_models.append(model_info)
            
            # Trier par importance de nettoyage (sur-distribution * faible utilisation)
            underused_models.sort(
                key=lambda m: ((m["current_server_count"] - m["optimal_server_count"]) * (5 - m["usage_score"])),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'identification des modèles peu utilisés: {str(e)}")
        
        return underused_models
    
    def _identify_available_servers(self) -> List[str]:
        """
        Identifie les serveurs disponibles pour héberger des modèles supplémentaires.
        
        Returns:
            Liste des serveurs disponibles, triés par préférence
        """
        available_servers = []
        
        try:
            # Récupérer tous les serveurs en bonne santé
            healthy_servers = self.cluster_manager.get_healthy_servers()
            
            # Récupérer les informations de charge pour chaque serveur
            server_loads = {
                server: self.cluster_manager.get_server_load(server)
                for server in healthy_servers
            }
            
            # Récupérer les statistiques matérielles si disponibles
            hardware_stats = {}
            for server in healthy_servers:
                server_data = self.db.search_one("servers", lambda q: q.address == server)
                if server_data and "hardware_info" in server_data:
                    hardware_stats[server] = server_data["hardware_info"]
            
            # Filtrer et trier les serveurs par disponibilité
            for server in healthy_servers:
                # Exclure les serveurs trop chargés
                if server_loads.get(server, 1.0) > 0.8:  # Charge > 80%
                    continue
                
                # Calculer un score de disponibilité
                availability_score = 0
                
                # Score basé sur la charge inverse (moins de charge = meilleur score)
                availability_score += 10 * (1 - server_loads.get(server, 0.5))
                
                # Score basé sur les ressources matérielles si disponibles
                if server in hardware_stats:
                    hw_info = hardware_stats[server]
                    
                    # Bonus pour GPU
                    if hw_info.get("gpu", {}).get("detected", False):
                        availability_score += 5
                        
                    # Bonus pour RAM disponible
                    mem_info = hw_info.get("memory", {})
                    if mem_info:
                        mem_percent_available = 100 - mem_info.get("percent", 50)
                        availability_score += mem_percent_available / 10  # 0-10 points
                
                available_servers.append((server, availability_score))
            
            # Trier par score de disponibilité décroissant
            available_servers.sort(key=lambda x: x[1], reverse=True)
            
            # Retourner uniquement les adresses de serveurs
            return [server for server, _ in available_servers]
            
        except Exception as e:
            logger.error(f"Erreur lors de l'identification des serveurs disponibles: {str(e)}")
            return []
    
    def _select_servers_for_cleanup(self, model_name: str, current_servers: List[str], count: int) -> List[str]:
        """
        Sélectionne les serveurs les moins appropriés pour héberger un modèle.
        
        Args:
            model_name: Nom du modèle
            current_servers: Liste des serveurs hébergeant actuellement ce modèle
            count: Nombre de serveurs à sélectionner pour nettoyage
            
        Returns:
            Liste des serveurs à nettoyer
        """
        servers_scores = []
        
        try:
            # Calculer un score pour chaque serveur (plus bas = meilleur candidat pour suppression)
            for server in current_servers:
                cleanup_score = 0
                
                # Pénalité pour charge élevée (les serveurs chargés sont de meilleurs candidats pour nettoyage)
                load = self.cluster_manager.get_server_load(server)
                cleanup_score += load * 10  # 0-10 points
                
                # Bonus pour les serveurs qui hébergent beaucoup de modèles (réduire la concentration)
                server_model_count = 0
                with self._models_lock:
                    for m, servers in self._model_servers.items():
                        if server in servers:
                            server_model_count += 1
                
                cleanup_score += min(server_model_count / 2, 5)  # 0-5 points
                
                # Malus pour serveur dédié à ce modèle (stats d'utilisation)
                model_stats = self.cluster_manager.get_model_stats(model_name, server)
                if model_stats:
                    usage_ratio = model_stats.get("usage_count", 0) / max(1, server_model_count)
                    cleanup_score -= min(usage_ratio / 10, 3)  # -3-0 points
                
                servers_scores.append((server, cleanup_score))
            
            # Trier par score de nettoyage décroissant
            servers_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Prendre les premiers serveurs selon le compte demandé, en gardant toujours au moins un
            return [server for server, _ in servers_scores[:min(count, len(current_servers)-1)]]
            
        except Exception as e:
            logger.error(f"Erreur lors de la sélection des serveurs pour nettoyage: {str(e)}")
            
            # En cas d'erreur, sélectionner des serveurs aléatoirement mais garder au moins un
            if len(current_servers) <= 1:
                return []
                
            servers_to_clean = current_servers.copy()
            random.shuffle(servers_to_clean)
            return servers_to_clean[:min(count, len(current_servers)-1)]
    
    def _select_optimal_nodes_for_model(self, model_name: str, count: int = 1) -> List[str]:
        """
        Sélectionne les nœuds optimaux pour héberger un modèle spécifique.
        
        Args:
            model_name: Nom du modèle
            count: Nombre de nœuds à sélectionner
            
        Returns:
            Liste des nœuds optimaux
        """
        try:
            # Récupérer les serveurs qui ont déjà ce modèle
            existing_servers = set()
            with self._models_lock:
                existing_servers = self._model_servers.get(model_name, set())
            
            # Récupérer les serveurs disponibles (non filtrés par charge)
            all_healthy_servers = self.cluster_manager.get_healthy_servers()
            
            # Si tous les serveurs ont déjà ce modèle, renvoyer une liste vide
            if all(server in existing_servers for server in all_healthy_servers):
                return []
            
            # Filtrer les serveurs qui n'ont pas encore ce modèle
            candidate_servers = [server for server in all_healthy_servers if server not in existing_servers]
            
            # Récupérer les informations matérielles et de charge des serveurs candidats
            server_scores = []
            for server in candidate_servers:
                # Score de base
                score = 0
                
                # Pénalité pour charge élevée
                load = self.cluster_manager.get_server_load(server)
                score -= load * 10  # -10-0 points
                
                # Récupérer les informations matérielles
                server_data = self.db.search_one("servers", lambda q: q.address == server)
                if server_data and "hardware_info" in server_data:
                    hw_info = server_data["hardware_info"]
                    
                    # Bonus pour GPU
                    if hw_info.get("gpu", {}).get("detected", False):
                        score += 15
                        
                    # Bonus pour RAM disponible
                    mem_info = hw_info.get("memory", {})
                    if mem_info:
                        mem_percent_available = 100 - mem_info.get("percent", 50)
                        score += mem_percent_available / 10  # 0-10 points
                
                server_scores.append((server, score))
            
            # Trier les serveurs par score décroissant
            server_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Retourner les meilleurs serveurs selon le compte demandé
            return [server for server, _ in server_scores[:count]]
            
        except Exception as e:
            logger.error(f"Erreur lors de la sélection de nœuds optimaux pour le modèle {model_name}: {str(e)}")
            
            # En cas d'erreur, sélectionner des serveurs aléatoirement
            healthy_servers = self.cluster_manager.get_healthy_servers()
            
            # Filtrer ceux qui n'ont pas déjà ce modèle
            with self._models_lock:
                existing_servers = self._model_servers.get(model_name, set())
            
            candidate_servers = [server for server in healthy_servers if server not in existing_servers]
            
            # Mélanger et sélectionner
            if candidate_servers:
                random.shuffle(candidate_servers)
                return candidate_servers[:count]
            
            return []
    
    def _update_model_location(self, model_name: str, server: str, present: bool) -> None:
        """
        Met à jour la localisation d'un modèle dans le registre.
        
        Args:
            model_name: Nom du modèle
            server: Adresse du serveur
            present: True si le modèle est présent, False s'il a été supprimé
        """
        with self._models_lock:
            # Initialiser si le modèle n'est pas encore dans le registre
            if model_name not in self._model_servers:
                self._model_servers[model_name] = set()
                
                # Initialiser les métadonnées si nécessaire
                if model_name not in self._models_cache:
                    self._models_cache[model_name] = {
                        "name": model_name,
                        "first_seen": datetime.now().isoformat(),
                        "usage_count": 0
                    }
            
            # Mettre à jour la liste des serveurs
            if present:
                self._model_servers[model_name].add(server)
            elif server in self._model_servers[model_name]:
                self._model_servers[model_name].remove(server)
    
    def _persist_model_changes(self, model_name: str) -> None:
        """
        Persiste les changements d'un modèle spécifique dans la base de données.
        
        Args:
            model_name: Nom du modèle à persister
        """
        try:
            with self._models_lock:
                if model_name in self._models_cache:
                    # Récupérer les métadonnées du modèle
                    model_data = self._models_cache[model_name].copy()
                    
                    # Ajouter la liste des serveurs
                    model_data["servers"] = list(self._model_servers.get(model_name, []))
                    
                    # Mettre à jour la date de dernière modification
                    model_data["last_update"] = datetime.now().isoformat()
                    
                    # Utilisez le SyncManager pour la mise à jour (persiste en DB et synchronise en RAM)
                    self.sync_manager.write_and_sync("models", model_data, 
                                                    self.db.search_one("models", lambda q: q.name == model_name).doc_id 
                                                    if self.db.search_one("models", lambda q: q.name == model_name) else None)
                    
                    logger.debug(f"Modèle {model_name} persisté dans la base de données")
        except Exception as e:
            logger.error(f"Erreur lors de la persistance du modèle {model_name}: {str(e)}")
    
    def _persist_all_models(self) -> None:
        """
        Persiste tous les modèles dans la base de données.
        """
        try:
            with self._models_lock:
                for model_name in self._models_cache:
                    self._persist_model_changes(model_name)
            
            logger.info(f"Tous les modèles ont été persistés dans la base de données")
        except Exception as e:
            logger.error(f"Erreur lors de la persistance de tous les modèles: {str(e)}")
    
    def shutdown(self) -> None:
        """
        Arrête proprement le registre des modèles et persiste son état.
        """
        logger.info("Arrêt du ModelRegistry")
        
        try:
            # Persister tous les modèles une dernière fois
            self._persist_all_models()
            
            # Arrêter le thread de vérification périodique
            if self._check_thread and self._check_thread.is_alive():
                self._stop_check.set()
                self._check_thread.join(timeout=2.0)
                self._check_thread = None
                
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du ModelRegistry: {str(e)}")


# Fonction singleton pour accéder au registre des modèles
def get_model_registry() -> ModelRegistry:
    """
    Récupère l'instance unique du registre des modèles.
    
    Returns:
        Instance du ModelRegistry
    """
    return ModelRegistry()