"""
Module de gestion de la santé du cluster Ollama.

Ce module fournit des fonctionnalités pour surveiller la santé des serveurs
dans un cluster Ollama, effectuer des vérifications régulières et générer des rapports.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Set, Tuple, Any, Optional
import datetime

from olol.sync.client import OllamaClient
from olol.proxy.cluster.manager import get_cluster_manager

# Configuration du logging
logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Moniteur de santé qui vérifie régulièrement l'état des serveurs du cluster.
    """

    def __init__(self):
        """
        Initialise le moniteur de santé.
        """
        self._cluster_manager = get_cluster_manager()
        
        # Historique de santé des serveurs: {server_address: [(timestamp, is_healthy), ...]}
        self._health_history = {}
        
        # Historique de charge des serveurs: {server_address: [(timestamp, load), ...]}
        self._load_history = {}
        
        # Historique de latence des serveurs: {server_address: [(timestamp, latency_ms), ...]}
        self._latency_history = {}
        
        # Taille maximale de l'historique (24 heures avec des points toutes les 5 minutes)
        self.max_history_size = 24 * 12
        
        # Verrou pour l'accès à l'historique
        self._history_lock = threading.RLock()
        
        # Pour le rafraîchissement périodique
        self._health_thread = None
        self._stop_health_check = threading.Event()
        
        # Intervalle de vérification de santé en secondes
        self.health_check_interval = 60
    
    def start(self):
        """
        Démarre le moniteur de santé.
        """
        if self._health_thread is None or not self._health_thread.is_alive():
            self._stop_health_check.clear()
            self._health_thread = threading.Thread(
                target=self._run_health_checks,
                daemon=True
            )
            self._health_thread.start()
            logger.info("Moniteur de santé démarré")
    
    def stop(self):
        """
        Arrête le moniteur de santé.
        """
        if self._health_thread and self._health_thread.is_alive():
            self._stop_health_check.set()
            self._health_thread.join(timeout=2.0)
            self._health_thread = None
            logger.info("Moniteur de santé arrêté")
    
    def _run_health_checks(self):
        """
        Exécute les vérifications de santé périodiques sur tous les serveurs.
        """
        while not self._stop_health_check.is_set():
            try:
                self.check_all_servers_health()
            except Exception as e:
                logger.error(f"Erreur lors des vérifications de santé: {str(e)}")
            
            # Attendre jusqu'à la prochaine vérification
            self._stop_health_check.wait(self.health_check_interval)
    
    def check_all_servers_health(self):
        """
        Vérifie la santé de tous les serveurs et met à jour l'historique.
        """
        current_time = time.time()
        servers = self._cluster_manager.get_server_addresses()
        
        for server in servers:
            try:
                # Vérifier la santé
                start_time = time.time()
                is_healthy = self._cluster_manager.check_server_health(server)
                latency_ms = (time.time() - start_time) * 1000
                
                # Obtenir la charge
                load = self._cluster_manager.get_server_load(server)
                
                # Mettre à jour l'historique
                with self._history_lock:
                    # Mettre à jour l'historique de santé
                    if server not in self._health_history:
                        self._health_history[server] = []
                    self._health_history[server].append((current_time, is_healthy))
                    
                    # Mettre à jour l'historique de charge
                    if server not in self._load_history:
                        self._load_history[server] = []
                    self._load_history[server].append((current_time, load))
                    
                    # Mettre à jour l'historique de latence
                    if server not in self._latency_history:
                        self._latency_history[server] = []
                    self._latency_history[server].append((current_time, latency_ms))
                    
                    # Limiter la taille de l'historique
                    self._trim_history(server)
                
                logger.debug(f"Santé du serveur {server}: {'OK' if is_healthy else 'NOK'}, "
                           f"Charge: {load:.2f}, Latence: {latency_ms:.2f}ms")
            
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de santé pour {server}: {str(e)}")
    
    def _trim_history(self, server):
        """
        Limite la taille de l'historique pour un serveur.
        
        Args:
            server: Adresse du serveur
        """
        if server in self._health_history and len(self._health_history[server]) > self.max_history_size:
            self._health_history[server] = self._health_history[server][-self.max_history_size:]
        
        if server in self._load_history and len(self._load_history[server]) > self.max_history_size:
            self._load_history[server] = self._load_history[server][-self.max_history_size:]
        
        if server in self._latency_history and len(self._latency_history[server]) > self.max_history_size:
            self._latency_history[server] = self._latency_history[server][-self.max_history_size:]
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Génère un rapport de santé complet pour tous les serveurs.
        
        Returns:
            Dict[str, Any]: Rapport de santé
        """
        current_time = time.time()
        servers = self._cluster_manager.get_server_addresses()
        report = {
            "timestamp": current_time,
            "datetime": datetime.datetime.fromtimestamp(current_time).isoformat(),
            "cluster_health": {
                "total_servers": len(servers),
                "healthy_servers": 0,
                "unhealthy_servers": 0,
                "average_load": 0.0,
                "average_latency": 0.0
            },
            "servers": {}
        }
        
        total_load = 0.0
        total_latency = 0.0
        
        for server in servers:
            is_healthy = self._cluster_manager.get_server_health(server)
            load = self._cluster_manager.get_server_load(server)
            
            # Obtenir la latence la plus récente
            latency = 0.0
            with self._history_lock:
                if server in self._latency_history and self._latency_history[server]:
                    latency = self._latency_history[server][-1][1]
            
            # Compter les serveurs sains/malsains
            if is_healthy:
                report["cluster_health"]["healthy_servers"] += 1
            else:
                report["cluster_health"]["unhealthy_servers"] += 1
            
            # Calculer les totaux pour les moyennes
            total_load += load
            total_latency += latency
            
            # Obtenir les modèles disponibles sur ce serveur
            models = []
            all_models = self._cluster_manager.get_all_models()
            for model, servers in all_models.items():
                if server in servers:
                    models.append(model)
            
            # Ajouter les détails du serveur
            report["servers"][server] = {
                "healthy": is_healthy,
                "load": load,
                "latency_ms": latency,
                "models": models
            }
        
        # Calculer les moyennes
        if servers:
            report["cluster_health"]["average_load"] = total_load / len(servers)
            report["cluster_health"]["average_latency"] = total_latency / len(servers)
        
        return report
    
    def get_health_history(self, server: str, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retourne l'historique de santé pour un serveur donné.
        
        Args:
            server: Adresse du serveur
            hours: Nombre d'heures d'historique à retourner (par défaut 24)
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Historique de santé
        """
        with self._history_lock:
            cutoff_time = time.time() - (hours * 3600)
            
            health_history = []
            if server in self._health_history:
                health_history = [
                    {"timestamp": ts, "healthy": status}
                    for ts, status in self._health_history[server]
                    if ts >= cutoff_time
                ]
            
            load_history = []
            if server in self._load_history:
                load_history = [
                    {"timestamp": ts, "load": load}
                    for ts, load in self._load_history[server]
                    if ts >= cutoff_time
                ]
            
            latency_history = []
            if server in self._latency_history:
                latency_history = [
                    {"timestamp": ts, "latency_ms": latency}
                    for ts, latency in self._latency_history[server]
                    if ts >= cutoff_time
                ]
            
            return {
                "health": health_history,
                "load": load_history,
                "latency": latency_history
            }
    
    def get_cluster_health_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Retourne des statistiques agrégées sur la santé du cluster.
        
        Args:
            hours: Nombre d'heures d'historique à analyser (par défaut 24)
            
        Returns:
            Dict[str, Any]: Statistiques de santé du cluster
        """
        cutoff_time = time.time() - (hours * 3600)
        stats = {
            "uptime_percentage": {},
            "average_load": {},
            "average_latency": {},
            "status_changes": {}
        }
        
        with self._history_lock:
            for server in self._health_history:
                # Filtrer les données par période
                health_data = [(ts, status) for ts, status in self._health_history[server] if ts >= cutoff_time]
                load_data = [(ts, load) for ts, load in self._load_history.get(server, []) if ts >= cutoff_time]
                latency_data = [(ts, latency) for ts, latency in self._latency_history.get(server, []) if ts >= cutoff_time]
                
                if health_data:
                    # Calculer le pourcentage de disponibilité
                    uptime = sum(1 for _, status in health_data if status) / len(health_data) * 100
                    stats["uptime_percentage"][server] = uptime
                    
                    # Compter les changements d'état
                    status_changes = 0
                    for i in range(1, len(health_data)):
                        if health_data[i][1] != health_data[i-1][1]:
                            status_changes += 1
                    stats["status_changes"][server] = status_changes
                
                # Calculer la charge moyenne
                if load_data:
                    avg_load = sum(load for _, load in load_data) / len(load_data)
                    stats["average_load"][server] = avg_load
                
                # Calculer la latence moyenne
                if latency_data:
                    avg_latency = sum(latency for _, latency in latency_data) / len(latency_data)
                    stats["average_latency"][server] = avg_latency
        
        return stats


# Instance globale du moniteur de santé
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """
    Retourne l'instance globale du moniteur de santé.
    
    Returns:
        HealthMonitor: Instance du moniteur de santé
    """
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def start_health_monitoring():
    """
    Démarre le monitoring de santé du cluster.
    """
    monitor = get_health_monitor()
    monitor.start()


def stop_health_monitoring():
    """
    Arrête le monitoring de santé du cluster.
    """
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop()