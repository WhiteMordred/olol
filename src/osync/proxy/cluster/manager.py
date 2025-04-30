import json
import logging
import threading
import time
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

from osync.sync.client import OllamaClient
from osync.utils.cluster import OllamaCluster
from osync.proxy.db.database import get_db

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusterManager:
    """
    Gestionnaire de cluster qui évite les verrous bloquants et offre une interface
    thread-safe pour accéder aux informations du cluster.
    """

    def __init__(self, cluster_config=None):
        """
        Initialise le gestionnaire de cluster.
        
        Args:
            cluster_config: Configuration du cluster (optionnel)
        """
        # Référence à l'objet cluster principal
        self._cluster = None
        
        # Cache des informations pour éviter les verrous
        self._cached_server_addresses = set()
        self._cached_server_health = {}  # address -> bool
        self._cached_server_loads = {}   # address -> float
        self._cached_model_servers = {}  # model -> [server1, server2, ...]
        
        # Verrous légers pour l'accès au cache
        self._cache_lock = threading.RLock()
        
        # Pour le rafraîchissement périodique
        self._refresh_thread = None
        self._stop_refresh = threading.Event()
        
        # Intervalle de rafraîchissement du cache en secondes
        self.refresh_interval = 5
        
        # Initialiser le cluster
        self.initialize(cluster_config)
    
    def initialize(self, cluster_config=None):
        """
        Initialise ou réinitialise le cluster et charge les données depuis TinyDB.
        
        Args:
            cluster_config: Configuration du cluster (optionnel)
        """
        try:
            # Créer ou mettre à jour le cluster Ollama
            if cluster_config:
                # S'assurer que server_addresses est présent dans la configuration
                if 'server_addresses' not in cluster_config:
                    # Essayer de récupérer les serveurs depuis la base de données
                    db = get_db()
                    servers = db.get_all("servers")
                    if servers:
                        server_addresses = [server["address"] for server in servers]
                        cluster_config['server_addresses'] = server_addresses
                    else:
                        # Si pas de serveurs trouvés, utiliser une liste vide
                        cluster_config['server_addresses'] = []
                        logger.warning("Aucun serveur trouvé dans la base de données. Le cluster sera initialisé avec une liste vide.")
                
                # Initialiser le cluster avec la configuration
                self._cluster = OllamaCluster(**cluster_config)
            else:
                # Sans configuration, initialiser avec les serveurs de la base de données
                db = get_db()
                servers = db.get_all("servers")
                server_addresses = [server["address"] for server in servers] if servers else []
                
                # Initialiser le cluster avec les adresses de serveurs
                self._cluster = OllamaCluster(server_addresses=server_addresses)
                logger.info(f"Cluster initialisé avec {len(server_addresses)} serveurs depuis la base de données")
            
            # Charger les données depuis TinyDB
            self._load_data_from_db()
            
            # Démarrer le thread de rafraîchissement si nécessaire
            self._start_refresh_thread()
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du cluster: {str(e)}")
            return False
    
    # Méthodes d'accès aux données du cluster pour l'interface web
    
    def get_server_addresses(self) -> Set[str]:
        """
        Retourne l'ensemble des adresses de serveurs.
        
        Returns:
            Un ensemble d'adresses de serveurs sous forme de chaînes de caractères.
        """
        with self._cache_lock:
            return self._cached_server_addresses.copy()
    
    def get_server_health(self, server_address: str) -> bool:
        """
        Retourne l'état de santé d'un serveur spécifique.
        
        Args:
            server_address: L'adresse du serveur
            
        Returns:
            True si le serveur est en bonne santé, False sinon.
        """
        with self._cache_lock:
            return self._cached_server_health.get(server_address, False)
    
    def get_server_load(self, server_address: str) -> float:
        """
        Retourne la charge actuelle d'un serveur spécifique.
        
        Args:
            server_address: L'adresse du serveur
            
        Returns:
            La charge du serveur (float entre 0.0 et 1.0).
        """
        with self._cache_lock:
            return self._cached_server_loads.get(server_address, 0.0)
    
    def get_healthy_servers(self) -> List[str]:
        """
        Retourne la liste des serveurs en bonne santé.
        
        Returns:
            Liste d'adresses de serveurs en bonne santé.
        """
        with self._cache_lock:
            return [addr for addr, healthy in self._cached_server_health.items() if healthy]
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """
        Retourne tous les modèles avec leurs serveurs associés.
        
        Returns:
            Dictionnaire avec les noms de modèles comme clés et listes de serveurs comme valeurs.
        """
        with self._cache_lock:
            # Convertir les ensembles en listes pour la sérialisation
            result = {}
            for model, servers in self._cached_model_servers.items():
                if isinstance(servers, set):
                    result[model] = list(servers)
                else:
                    # Si c'est déjà une liste, la copier simplement
                    result[model] = servers.copy() if isinstance(servers, list) else list(servers)
            return result
            
    def _load_data_from_db(self):
        """
        Charge les données de TinyDB et les réconcilie avec l'état actuel.
        """
        try:
            db = get_db()
            
            # Charger les serveurs
            servers = db.get_all("servers")
            if servers:
                with self._cache_lock:
                    # Récupérer les adresses de serveurs
                    addresses = set(server["address"] for server in servers)
                    self._cached_server_addresses = addresses
                    
                    # Récupérer les états de santé
                    health = {server["address"]: server["healthy"] for server in servers}
                    self._cached_server_health.update(health)
                    
                    # Récupérer les charges
                    loads = {server["address"]: server.get("load", 0.0) for server in servers}
                    self._cached_server_loads.update(loads)
                    
                    # Mettre à jour le cluster avec les données récupérées
                    if self._cluster:
                        with self._cluster.server_lock:
                            for addr in addresses:
                                if addr not in self._cluster.server_addresses:
                                    self._cluster.server_addresses.append(addr)
                        
                        with self._cluster.health_lock:
                            self._cluster.server_health.update(health)
                            
                        with self._cluster.server_lock:
                            self._cluster.server_loads.update(loads)
            
            # Charger les modèles
            models = db.get_all("models")
            if models:
                model_servers = {}
                for model in models:
                    model_servers[model["name"]] = model["servers"]
                
                with self._cache_lock:
                    self._cached_model_servers.update(model_servers)
                    
                    # Mettre à jour le cluster
                    if self._cluster:
                        with self._cluster.model_lock:
                            for model_name, servers in model_servers.items():
                                self._cluster.model_server_map[model_name] = set(servers)
            
            logger.info(f"Données chargées depuis la base de données: {len(servers)} serveurs, {len(models)} modèles")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données depuis TinyDB: {str(e)}")
    
    def _start_refresh_thread(self):
        """Démarre le thread de rafraîchissement du cache."""
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            self._stop_refresh.clear()
            self._refresh_thread = threading.Thread(
                target=self._refresh_cache_periodically,
                daemon=True
            )
            self._refresh_thread.start()
    
    def _refresh_cache_periodically(self):
        """
        Rafraîchit périodiquement le cache des informations du cluster
        sans bloquer l'interface utilisateur.
        """
        while not self._stop_refresh.is_set():
            try:
                self.refresh_cache()
            except Exception as e:
                logger.error(f"Erreur lors du rafraîchissement du cache: {str(e)}")
            
            # Attendre jusqu'au prochain rafraîchissement
            self._stop_refresh.wait(self.refresh_interval)
    
    def refresh_cache(self):
        """
        Met à jour le cache des informations du cluster et persiste dans TinyDB.
        """
        if not self._cluster:
            return
        
        try:
            # Mettre à jour les adresses des serveurs
            server_addresses = set()
            with self._cluster.server_lock:
                server_addresses = set(self._cluster.server_addresses)
            
            # Mettre à jour les informations de santé
            server_health = {}
            with self._cluster.health_lock:
                server_health = dict(self._cluster.server_health)
            
            # Mettre à jour les charges des serveurs
            server_loads = {}
            with self._cluster.server_lock:
                server_loads = dict(self._cluster.server_loads)
            
            # Mettre à jour la cartographie des modèles aux serveurs
            model_servers = {}
            with self._cluster.model_lock:
                model_servers = {model: list(servers) 
                               for model, servers in self._cluster.model_server_map.items()}
            
            # Mettre à jour le cache de manière atomique
            with self._cache_lock:
                self._cached_server_addresses = server_addresses
                self._cached_server_health = server_health
                self._cached_server_loads = server_loads
                self._cached_model_servers = model_servers
            
            # Persister dans TinyDB
            self._persist_to_db(server_addresses, server_health, server_loads, model_servers)
        
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du cache: {str(e)}")
    
    def _persist_to_db(self, server_addresses, server_health, server_loads, model_servers):
        """
        Persiste les données du cluster dans TinyDB.
        """
        try:
            db = get_db()
            current_time = datetime.now().isoformat()
            
            # Persister les serveurs
            for address in server_addresses:
                server_data = {
                    "address": address,
                    "healthy": server_health.get(address, False),
                    "load": server_loads.get(address, 0.0),
                    "last_check": current_time
                }
                
                # Vérifier si ce serveur existe déjà
                existing_server = db.search_one("servers", lambda q: q.address == address)
                
                if existing_server:
                    # Conserver la date de première découverte
                    server_data["first_seen"] = existing_server.get("first_seen", current_time)
                    
                    # Conserver les infos matérielles si elles existent
                    if "hardware_info" in existing_server:
                        server_data["hardware_info"] = existing_server["hardware_info"]
                        
                    # Mettre à jour
                    db.update("servers", server_data, lambda q: q.address == address)
                else:
                    # Ajouter la date de première découverte pour un nouveau serveur
                    server_data["first_seen"] = current_time
                    
                    # Insérer
                    db.insert("servers", server_data)
            
            # Persister les modèles
            for model_name, servers in model_servers.items():
                model_data = {
                    "name": model_name,
                    "servers": list(servers)
                }
                
                # Vérifier si ce modèle existe déjà
                existing_model = db.search_one("models", lambda q: q.name == model_name)
                
                if existing_model:
                    # Conserver les statistiques et métadonnées
                    model_data["first_seen"] = existing_model.get("first_seen", current_time)
                    model_data["usage_count"] = existing_model.get("usage_count", 0)
                    model_data["size_gb"] = existing_model.get("size_gb", None)
                    model_data["parameter_count"] = existing_model.get("parameter_count", None)
                    model_data["quantization"] = existing_model.get("quantization", None)
                    
                    # Mettre à jour
                    db.update("models", model_data, lambda q: q.name == model_name)
                else:
                    # Ajouter la date de première découverte pour un nouveau modèle
                    model_data["first_seen"] = current_time
                    model_data["usage_count"] = 0
                    
                    # Insérer
                    db.insert("models", model_data)
                
        except Exception as e:
            logger.error(f"Erreur lors de la persistance des données dans TinyDB: {str(e)}")
    
    def persist_server(self, server_address: str, health: bool = None, load: float = None):
        """
        Persiste les informations d'un serveur spécifique dans TinyDB.
        
        Args:
            server_address: Adresse du serveur
            health: État de santé (optionnel)
            load: Charge du serveur (optionnel)
        """
        try:
            db = get_db()
            current_time = datetime.now().isoformat()
            
            # Récupérer les valeurs actuelles si non fournies
            if health is None:
                with self._cache_lock:
                    health = self._cached_server_health.get(server_address, False)
                    
            if load is None:
                with self._cache_lock:
                    load = self._cached_server_loads.get(server_address, 0.0)
            
            # Mettre à jour ou créer le serveur
            server_data = {
                "address": server_address,
                "healthy": health,
                "load": load,
                "last_check": current_time
            }
            
            # Vérifier si ce serveur existe déjà
            db.upsert("servers", server_data, lambda q: q.address == server_address)
            
        except Exception as e:
            logger.error(f"Erreur lors de la persistance du serveur {server_address}: {str(e)}")

    def persist_model_map(self, model_name: str, servers: List[str]):
        """
        Persiste le mapping modèle-serveurs dans TinyDB.
        
        Args:
            model_name: Nom du modèle
            servers: Liste des serveurs qui ont ce modèle
        """
        try:
            db = get_db()
            current_time = datetime.now().isoformat()
            
            # Données du modèle
            model_data = {
                "name": model_name,
                "servers": list(servers),
                "last_update": current_time
            }
            
            # Vérifier si ce modèle existe déjà
            existing_model = db.search_one("models", lambda q: q.name == model_name)
            
            if existing_model:
                # Conserver les statistiques et métadonnées
                model_data["first_seen"] = existing_model.get("first_seen", current_time)
                model_data["usage_count"] = existing_model.get("usage_count", 0)
                model_data["size_gb"] = existing_model.get("size_gb", None)
                model_data["parameter_count"] = existing_model.get("parameter_count", None)
                model_data["quantization"] = existing_model.get("quantization", None)
                
                # Mettre à jour
                db.update("models", model_data, lambda q: q.name == model_name)
            else:
                # Ajouter la date de première découverte pour un nouveau modèle
                model_data["first_seen"] = current_time
                model_data["usage_count"] = 0
                
                # Insérer
                db.insert("models", model_data)
                
        except Exception as e:
            logger.error(f"Erreur lors de la persistance du modèle {model_name}: {str(e)}")

    def update_model_usage(self, model_name: str, tokens_generated: int = None, inference_time_ms: int = None):
        """
        Incrémente le compteur d'utilisation d'un modèle et met à jour ses statistiques.
        
        Args:
            model_name: Nom du modèle
            tokens_generated: Nombre de tokens générés dans cette session (optionnel)
            inference_time_ms: Temps d'inférence en millisecondes (optionnel)
        """
        try:
            db = get_db()
            
            # Vérifier si le modèle existe
            existing_model = db.search_one("models", lambda q: q.name == model_name)
            
            if existing_model:
                update_data = {
                    "usage_count": existing_model.get("usage_count", 0) + 1,
                    "last_used": datetime.now().isoformat()
                }
                
                # Mettre à jour les statistiques cumulatives si fournies
                if tokens_generated is not None:
                    current_tokens = existing_model.get("total_tokens_generated", 0)
                    update_data["total_tokens_generated"] = current_tokens + tokens_generated
                
                if inference_time_ms is not None:
                    current_time = existing_model.get("total_inference_time_ms", 0)
                    update_data["total_inference_time_ms"] = current_time + inference_time_ms
                    
                    # Calculer et stocker le temps moyen par inférence
                    update_data["avg_inference_time_ms"] = update_data["total_inference_time_ms"] / update_data["usage_count"]
                
                # Mettre à jour
                db.update("models", update_data, lambda q: q.name == model_name)
                logger.info(f"Statistiques mises à jour pour le modèle {model_name}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'utilisation du modèle {model_name}: {str(e)}")

    def update_model_usage_stats(self, model_name: str, server_id: str, inference_time_ms: int = None, tokens_generated: int = None):
        """
        Met à jour les statistiques d'utilisation d'un modèle avec des métriques détaillées.
        
        Args:
            model_name: Nom du modèle
            server_id: Identifiant du serveur où le modèle a été utilisé
            inference_time_ms: Temps d'inférence en millisecondes (optionnel)
            tokens_generated: Nombre de tokens générés (optionnel)
        """
        try:
            # Incrémenter le compteur d'utilisation standard
            self.update_model_usage(model_name, tokens_generated, inference_time_ms)
            
            db = get_db()
            current_time = datetime.now().isoformat()
            
            # Rechercher les statistiques existantes pour ce modèle sur ce serveur
            model_stats = db.search_one("model_stats", lambda q: q.model_name == model_name and q.server_id == server_id)
            
            if model_stats:
                # Mettre à jour les statistiques existantes
                update_data = {
                    "last_used": current_time,
                    "usage_count": model_stats.get("usage_count", 0) + 1
                }
                
                # Mettre à jour les statistiques cumulatives si fournies
                if tokens_generated is not None:
                    current_tokens = model_stats.get("total_tokens_generated", 0)
                    update_data["total_tokens_generated"] = current_tokens + tokens_generated
                
                if inference_time_ms is not None:
                    current_time = model_stats.get("total_inference_time_ms", 0)
                    update_data["total_inference_time_ms"] = current_time + inference_time_ms
                    
                    # Calculer et stocker le temps moyen par inférence
                    update_data["avg_inference_time_ms"] = update_data["total_inference_time_ms"] / update_data["usage_count"]
                
                # Mettre à jour
                db.update("model_stats", update_data, lambda q: q.model_name == model_name and q.server_id == server_id)
            else:
                # Créer de nouvelles statistiques
                new_stats = {
                    "model_name": model_name,
                    "server_id": server_id,
                    "first_used": current_time,
                    "last_used": current_time,
                    "usage_count": 1
                }
                
                # Ajouter les statistiques si fournies
                if tokens_generated is not None:
                    new_stats["total_tokens_generated"] = tokens_generated
                
                if inference_time_ms is not None:
                    new_stats["total_inference_time_ms"] = inference_time_ms
                    new_stats["avg_inference_time_ms"] = inference_time_ms
                
                # Insérer
                db.insert("model_stats", new_stats)
            
            logger.info(f"Statistiques d'utilisation mises à jour pour le modèle {model_name} sur le serveur {server_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des statistiques d'utilisation du modèle {model_name} sur le serveur {server_id}: {str(e)}")

    def get_model_stats(self, model_name: str = None, server_id: str = None):
        """
        Récupère les statistiques d'utilisation des modèles.
        
        Args:
            model_name: Filtrer par nom de modèle (optionnel)
            server_id: Filtrer par serveur (optionnel)
            
        Returns:
            dict: Statistiques d'utilisation des modèles
        """
        try:
            db = get_db()
            
            # Construire la requête en fonction des paramètres fournis
            if model_name and server_id:
                # Statistiques pour un modèle spécifique sur un serveur spécifique
                stats = db.search_one("model_stats", lambda q: q.model_name == model_name and q.server_id == server_id)
                return stats if stats else {}
                
            elif model_name:
                # Statistiques pour un modèle spécifique sur tous les serveurs
                stats = db.search("model_stats", lambda q: q.model_name == model_name)
                
                # Agréger les statistiques
                if not stats:
                    # Récupérer les informations de base du modèle
                    model_info = db.search_one("models", lambda q: q.name == model_name)
                    return model_info if model_info else {}
                
                # Calculer des statistiques agrégées
                aggregated = {
                    "model_name": model_name,
                    "usage_count": sum(s.get("usage_count", 0) for s in stats),
                    "total_tokens_generated": sum(s.get("total_tokens_generated", 0) for s in stats),
                    "total_inference_time_ms": sum(s.get("total_inference_time_ms", 0) for s in stats),
                    "server_count": len(stats),
                    "servers": [s.get("server_id") for s in stats]
                }
                
                # Calculer le temps moyen global
                if aggregated["usage_count"] > 0:
                    aggregated["avg_inference_time_ms"] = aggregated["total_inference_time_ms"] / aggregated["usage_count"]
                
                return aggregated
                
            elif server_id:
                # Statistiques pour tous les modèles sur un serveur spécifique
                stats = db.search("model_stats", lambda q: q.server_id == server_id)
                return stats if stats else []
                
            else:
                # Statistiques globales pour tous les modèles
                models = db.get_all("models")
                
                # Enrichir avec les statistiques détaillées
                for model in models:
                    model_stats = db.search("model_stats", lambda q: q.model_name == model["name"])
                    if model_stats:
                        model["detailed_stats"] = model_stats
                
                return models
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques des modèles: {str(e)}")
            return {}

    def update_server_metrics(self, server_id: str, response_time_ms: int = None, tokens_per_second: float = None):
        """
        Met à jour les métriques de performance d'un serveur.
        
        Args:
            server_id: Identifiant unique du serveur
            response_time_ms: Temps de réponse en millisecondes (optionnel)
            tokens_per_second: Débit en tokens par seconde (optionnel)
        """
        try:
            db = get_db()
            server = db.search_one("servers", lambda q: q.id == server_id)
            
            if server:
                update_data = {}
                
                # Mise à jour du temps de réponse
                if response_time_ms is not None:
                    # Calculer la moyenne glissante pour le temps de réponse
                    current_avg = server.get("avg_response_time_ms", 0)
                    current_count = server.get("response_time_samples", 0)
                    
                    if current_count == 0:
                        new_avg = response_time_ms
                        new_count = 1
                    else:
                        # Limiter à 100 échantillons pour la moyenne glissante
                        weight = min(current_count, 100)
                        new_avg = (current_avg * weight + response_time_ms) / (weight + 1)
                        new_count = current_count + 1
                    
                    update_data["avg_response_time_ms"] = new_avg
                    update_data["response_time_samples"] = new_count
                    update_data["last_response_time_ms"] = response_time_ms
                
                # Mise à jour du débit de tokens
                if tokens_per_second is not None:
                    current_avg = server.get("avg_tokens_per_second", 0)
                    current_count = server.get("tokens_rate_samples", 0)
                    
                    if current_count == 0:
                        new_avg = tokens_per_second
                        new_count = 1
                    else:
                        # Limiter à 100 échantillons pour la moyenne glissante
                        weight = min(current_count, 100)
                        new_avg = (current_avg * weight + tokens_per_second) / (weight + 1)
                        new_count = current_count + 1
                    
                    update_data["avg_tokens_per_second"] = new_avg
                    update_data["tokens_rate_samples"] = new_count
                    update_data["last_tokens_per_second"] = tokens_per_second
                
                if update_data:
                    # Ajouter l'horodatage de la dernière mise à jour
                    update_data["metrics_updated_at"] = datetime.now().isoformat()
                    db.update("servers", update_data, lambda q: q.id == server_id)
                    logger.info(f"Métriques mises à jour pour le serveur {server_id}")
            else:
                logger.warning(f"Tentative de mise à jour des métriques pour un serveur inconnu: {server_id}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques du serveur {server_id}: {str(e)}")
    
    def shutdown(self):
        """Arrête le thread de rafraîchissement et persiste l'état final."""
        if self._refresh_thread and self._refresh_thread.is_alive():
            # Effectuer une dernière persistance des données
            try:
                self.refresh_cache()
            except Exception as e:
                logger.error(f"Erreur lors de la persistance finale: {str(e)}")
                
            # Arrêter le thread
            self._stop_refresh.set()
            self._refresh_thread.join(timeout=2.0)
            self._refresh_thread = None

    def check_server_health(self, server_address: str) -> bool:
        """
        Vérifie la santé d'un serveur spécifique en essayant de s'y connecter.
        
        Args:
            server_address: L'adresse du serveur à vérifier
            
        Returns:
            bool: True si le serveur est en bonne santé, False sinon
        """
        try:
            # Format host:port simple
            if server_address.count(':') == 1:
                host, port_str = server_address.split(':')
                port = int(port_str)
                
                from osync.sync.client import OllamaClient
                client = OllamaClient(host=host, port=port, timeout=2.0)  # Timeout réduit pour éviter les blocages
                
                try:
                    # Vérifier la santé du serveur avec un timeout court
                    is_healthy = client.check_health(timeout=1.0)
                    
                    # Mettre à jour le cache
                    with self._cache_lock:
                        self._cached_server_health[server_address] = is_healthy
                    
                    # Si le cluster existe, mettre à jour son état
                    if self._cluster:
                        with self._cluster.health_lock:
                            self._cluster.server_health[server_address] = is_healthy
                    
                    return is_healthy
                    
                except Exception as e:
                    logger.warning(f"Erreur lors de la vérification de santé de {server_address}: {str(e)}")
                    
                    # Marquer comme non sain en cas d'erreur
                    with self._cache_lock:
                        self._cached_server_health[server_address] = False
                    
                    if self._cluster:
                        with self._cluster.health_lock:
                            self._cluster.server_health[server_address] = False
                    
                    return False
                finally:
                    # Fermer le client pour libérer les ressources
                    client.close()
            else:
                # Format d'adresse non reconnu
                logger.warning(f"Format d'adresse non supporté: {server_address}")
                return False
                
        except Exception as e:
            logger.warning(f"Erreur de connexion au serveur {server_address}: {str(e)}")
            
            # Marquer comme non sain en cas d'erreur
            with self._cache_lock:
                self._cached_server_health[server_address] = False
            
            if self._cluster:
                with self._cluster.health_lock:
                    self._cluster.server_health[server_address] = False
            
            return False


# Instance globale du gestionnaire de cluster
_cluster_manager = None


def get_cluster_manager() -> ClusterManager:
    """
    Retourne l'instance globale du gestionnaire de cluster.
    
    Returns:
        ClusterManager: Instance du gestionnaire de cluster
    """
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager