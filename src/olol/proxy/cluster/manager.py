import json
import logging
import threading
import time
from typing import Dict, List, Optional, Set, Any

from olol.sync.client import OllamaClient
from olol.utils.cluster import OllamaCluster

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
        Initialise ou réinitialise le cluster.
        
        Args:
            cluster_config: Configuration du cluster (optionnel)
        """
        try:
            # Créer ou mettre à jour le cluster Ollama
            if cluster_config:
                self._cluster = OllamaCluster(**cluster_config)
            else:
                self._cluster = OllamaCluster()
            
            # Démarrer le thread de rafraîchissement si nécessaire
            self._start_refresh_thread()
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du cluster: {str(e)}")
            return False
    
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
        Met à jour le cache des informations du cluster.
        Cette méthode est conçue pour être rapide et ne pas bloquer.
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
        
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du cache: {str(e)}")
    
    def get_server_addresses(self) -> List[str]:
        """
        Retourne la liste des adresses de serveurs sans bloquer.
        
        Returns:
            List[str]: Liste des adresses de serveurs
        """
        with self._cache_lock:
            return list(self._cached_server_addresses)
    
    def get_healthy_servers(self) -> List[str]:
        """
        Retourne la liste des serveurs sains sans bloquer.
        
        Returns:
            List[str]: Liste des serveurs sains
        """
        with self._cache_lock:
            return [server for server, healthy in self._cached_server_health.items() 
                   if healthy and server in self._cached_server_addresses]
    
    def get_server_health(self, server_address: str) -> bool:
        """
        Retourne l'état de santé d'un serveur sans bloquer.
        
        Args:
            server_address: Adresse du serveur
            
        Returns:
            bool: État de santé du serveur
        """
        with self._cache_lock:
            return self._cached_server_health.get(server_address, False)
    
    def get_server_load(self, server_address: str) -> float:
        """
        Retourne la charge d'un serveur sans bloquer.
        
        Args:
            server_address: Adresse du serveur
            
        Returns:
            float: Charge du serveur
        """
        with self._cache_lock:
            return self._cached_server_loads.get(server_address, 0.0)
    
    def get_servers_for_model(self, model_name: str) -> List[str]:
        """
        Retourne la liste des serveurs qui ont un modèle particulier.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            List[str]: Liste des serveurs qui ont le modèle
        """
        with self._cache_lock:
            return self._cached_model_servers.get(model_name, [])
    
    def get_optimal_server(self, model_name: str = None) -> Optional[str]:
        """
        Retourne le serveur optimal pour un modèle donné.
        
        Args:
            model_name: Nom du modèle (optionnel)
            
        Returns:
            Optional[str]: Adresse du serveur optimal ou None si aucun n'est disponible
        """
        # Si un modèle est spécifié, n'utiliser que les serveurs qui l'ont
        candidate_servers = []
        if model_name:
            with self._cache_lock:
                candidate_servers = [server for server in self._cached_model_servers.get(model_name, [])
                                   if self._cached_server_health.get(server, False)]
        else:
            with self._cache_lock:
                candidate_servers = [server for server in self._cached_server_addresses
                                   if self._cached_server_health.get(server, False)]
        
        if not candidate_servers:
            return None
        
        # Trouver le serveur avec la charge minimale
        min_load = float('inf')
        optimal_server = None
        
        with self._cache_lock:
            for server in candidate_servers:
                server_load = self._cached_server_loads.get(server, 0.0)
                if server_load < min_load:
                    min_load = server_load
                    optimal_server = server
        
        return optimal_server
    
    def get_client_for_server(self, server_address: str) -> Optional[OllamaClient]:
        """
        Crée un client Ollama pour un serveur donné.
        
        Args:
            server_address: Adresse du serveur au format host:port
            
        Returns:
            Optional[OllamaClient]: Client Ollama ou None en cas d'erreur
        """
        try:
            if server_address.count(':') == 1:
                host, port_str = server_address.split(':')
                port = int(port_str)
                return OllamaClient(host=host, port=port)
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la création du client: {str(e)}")
            return None
    
    def get_client_for_model(self, model_name: str) -> Optional[OllamaClient]:
        """
        Crée un client Ollama pour un modèle donné.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Optional[OllamaClient]: Client Ollama ou None en cas d'erreur
        """
        server = self.get_optimal_server(model_name)
        if server:
            return self.get_client_for_server(server)
        return None
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """
        Retourne tous les modèles disponibles et les serveurs qui les ont.
        
        Returns:
            Dict[str, List[str]]: Dictionnaire {nom_modèle: [serveur1, serveur2, ...]}
        """
        with self._cache_lock:
            return {model: list(servers) for model, servers in self._cached_model_servers.items()}
    
    def check_server_health(self, server_address: str) -> bool:
        """
        Vérifie activement la santé d'un serveur et met à jour le cache.
        
        Args:
            server_address: Adresse du serveur
            
        Returns:
            bool: True si le serveur est sain, False sinon
        """
        try:
            client = self.get_client_for_server(server_address)
            if client:
                try:
                    is_healthy = client.check_health()
                    
                    # Mettre à jour le cache
                    with self._cache_lock:
                        self._cached_server_health[server_address] = is_healthy
                    
                    # Mettre à jour le cluster également
                    if self._cluster:
                        with self._cluster.health_lock:
                            self._cluster.server_health[server_address] = is_healthy
                    
                    return is_healthy
                finally:
                    client.close()
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de santé: {str(e)}")
            return False
    
    def update_server_load(self, server_address: str, load: float) -> None:
        """
        Met à jour la charge d'un serveur dans le cache et dans le cluster.
        
        Args:
            server_address: Adresse du serveur
            load: Charge du serveur
        """
        try:
            # Mettre à jour le cache
            with self._cache_lock:
                self._cached_server_loads[server_address] = load
            
            # Mettre à jour le cluster également
            if self._cluster:
                with self._cluster.server_lock:
                    self._cluster.server_loads[server_address] = load
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la charge: {str(e)}")
    
    def shutdown(self):
        """Arrête le thread de rafraîchissement."""
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._stop_refresh.set()
            self._refresh_thread.join(timeout=2.0)
            self._refresh_thread = None


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