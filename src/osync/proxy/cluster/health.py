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

from osync.sync.client import OllamaClient
from osync.proxy.cluster.manager import get_cluster_manager
from osync.proxy.db.database import get_db

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
        
        # Charger l'historique depuis la base de données
        self._load_history_from_db()
    
    def _load_history_from_db(self):
        """
        Charge l'historique de santé depuis la base de données.
        """
        try:
            db = get_db()
            cutoff_time = time.time() - (24 * 3600)  # 24 heures
            iso_cutoff = datetime.datetime.fromtimestamp(cutoff_time).isoformat()
            
            # Récupérer les statistiques récentes (24 dernières heures)
            stats = db.search("server_stats", lambda q: q.timestamp >= iso_cutoff)
            
            # Regrouper par serveur et par type
            for stat in stats:
                server = stat.get("server")
                stat_type = stat.get("type")
                timestamp = datetime.datetime.fromisoformat(stat.get("timestamp")).timestamp()
                value = stat.get("value")
                
                # Ajouter aux historiques en mémoire en fonction du type
                with self._history_lock:
                    if stat_type == "health":
                        if server not in self._health_history:
                            self._health_history[server] = []
                        self._health_history[server].append((timestamp, bool(value)))
                    
                    elif stat_type == "load":
                        if server not in self._load_history:
                            self._load_history[server] = []
                        self._load_history[server].append((timestamp, float(value)))
                    
                    elif stat_type == "latency":
                        if server not in self._latency_history:
                            self._latency_history[server] = []
                        self._latency_history[server].append((timestamp, float(value)))
            
            # Trier et limiter les historiques
            with self._history_lock:
                for server in self._health_history:
                    self._health_history[server].sort(key=lambda x: x[0])
                    self._trim_history(server)
                
                for server in self._load_history:
                    self._load_history[server].sort(key=lambda x: x[0])
                    self._trim_history(server)
                
                for server in self._latency_history:
                    self._latency_history[server].sort(key=lambda x: x[0])
                    self._trim_history(server)
            
            logger.info(f"Historique de santé chargé depuis la base de données pour {len(self._health_history)} serveurs")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique depuis la base de données: {str(e)}")
    
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
        Cette méthode est conçue pour persister les données dans TinyDB.
        """
        current_time = time.time()
        current_iso = datetime.datetime.fromtimestamp(current_time).isoformat()
        servers = self._cluster_manager.get_server_addresses()
        db = get_db()
        
        for server in servers:
            try:
                # Vérifier la santé
                start_time = time.time()
                is_healthy = self._cluster_manager.check_server_health(server)
                latency_ms = (time.time() - start_time) * 1000
                
                # Obtenir la charge
                load = self._cluster_manager.get_server_load(server)
                
                # Mettre à jour l'historique en mémoire
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
                    
                    # Limiter la taille de l'historique en mémoire
                    self._trim_history(server)
                
                # Persister dans TinyDB
                try:
                    # Enregistrer la santé
                    db.insert("server_stats", {
                        "server": server,
                        "timestamp": current_iso,
                        "type": "health",
                        "value": is_healthy
                    })
                    
                    # Enregistrer la charge
                    db.insert("server_stats", {
                        "server": server,
                        "timestamp": current_iso,
                        "type": "load",
                        "value": load
                    })
                    
                    # Enregistrer la latence
                    db.insert("server_stats", {
                        "server": server,
                        "timestamp": current_iso,
                        "type": "latency",
                        "value": latency_ms
                    })
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la persistance des stats pour {server}: {str(e)}")
                
                logger.debug(f"Santé du serveur {server}: {'OK' if is_healthy else 'NOK'}, "
                           f"Charge: {load:.2f}, Latence: {latency_ms:.2f}ms")
            
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de santé pour {server}: {str(e)}")
        
        # Nettoyer les anciennes données périodiquement (une fois par jour)
        if current_time % (24 * 3600) < self.health_check_interval:
            self.cleanup_old_stats()
    
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
    
    def cleanup_old_stats(self, days_to_keep=30):
        """
        Nettoie les anciennes statistiques de la base de données.
        
        Args:
            days_to_keep: Nombre de jours de données à conserver
        """
        try:
            db = get_db()
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            iso_cutoff = datetime.datetime.fromtimestamp(cutoff_time).isoformat()
            
            # Supprimer les anciennes statistiques
            removed = db.remove("server_stats", lambda q: q.timestamp < iso_cutoff)
            
            if removed:
                logger.info(f"Nettoyage des statistiques anciennes: {removed} enregistrements supprimés")
                
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des anciennes statistiques: {str(e)}")
    
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
        Cette méthode récupère les données depuis TinyDB pour des périodes longues.
        
        Args:
            server: Adresse du serveur
            hours: Nombre d'heures d'historique à retourner (par défaut 24)
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Historique de santé
        """
        try:
            cutoff_time = time.time() - (hours * 3600)
            iso_cutoff = datetime.datetime.fromtimestamp(cutoff_time).isoformat()
            
            # Pour des périodes courtes (≤ 24h), utiliser le cache en mémoire
            if hours <= 24:
                with self._history_lock:
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
            
            # Pour des périodes plus longues, requêter la base de données
            db = get_db()
            server_stats = db.search(
                "server_stats", 
                lambda q: q.server == server and q.timestamp >= iso_cutoff
            )
            
            # Organiser par type
            health_history = []
            load_history = []
            latency_history = []
            
            for stat in server_stats:
                ts = datetime.datetime.fromisoformat(stat["timestamp"]).timestamp()
                
                if stat["type"] == "health":
                    health_history.append({
                        "timestamp": ts,
                        "healthy": bool(stat["value"])
                    })
                elif stat["type"] == "load":
                    load_history.append({
                        "timestamp": ts,
                        "load": float(stat["value"])
                    })
                elif stat["type"] == "latency":
                    latency_history.append({
                        "timestamp": ts,
                        "latency_ms": float(stat["value"])
                    })
            
            # Trier les listes par timestamp
            health_history.sort(key=lambda x: x["timestamp"])
            load_history.sort(key=lambda x: x["timestamp"])
            latency_history.sort(key=lambda x: x["timestamp"])
            
            return {
                "health": health_history,
                "load": load_history,
                "latency": latency_history
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique pour {server}: {str(e)}")
            return {"health": [], "load": [], "latency": []}
    
    def get_cluster_health_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Retourne des statistiques agrégées sur la santé du cluster.
        Pour des périodes plus longues, cette méthode utilise TinyDB.
        
        Args:
            hours: Nombre d'heures d'historique à analyser (par défaut 24)
            
        Returns:
            Dict[str, Any]: Statistiques de santé du cluster
        """
        # Pour de longues périodes, utiliser la base de données
        if hours > 24:
            return self._get_cluster_health_stats_from_db(hours)
        
        # Pour des périodes courtes, utiliser le cache en mémoire
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
    
    def _get_cluster_health_stats_from_db(self, hours: int = 168) -> Dict[str, Any]:
        """
        Récupère les statistiques de santé du cluster depuis la base de données.
        
        Args:
            hours: Nombre d'heures d'historique à analyser (par défaut 168 = 7 jours)
            
        Returns:
            Dict[str, Any]: Statistiques de santé du cluster
        """
        try:
            cutoff_time = time.time() - (hours * 3600)
            iso_cutoff = datetime.datetime.fromtimestamp(cutoff_time).isoformat()
            
            db = get_db()
            stats_records = db.search("server_stats", lambda q: q.timestamp >= iso_cutoff)
            
            # Organiser par serveur et par type
            server_stats = {}
            
            for stat in stats_records:
                server = stat["server"]
                stat_type = stat["type"]
                ts = datetime.datetime.fromisoformat(stat["timestamp"]).timestamp()
                value = stat["value"]
                
                if server not in server_stats:
                    server_stats[server] = {"health": [], "load": [], "latency": []}
                
                if stat_type == "health":
                    server_stats[server]["health"].append((ts, value))
                elif stat_type == "load":
                    server_stats[server]["load"].append((ts, value))
                elif stat_type == "latency":
                    server_stats[server]["latency"].append((ts, value))
            
            # Calculer les statistiques
            result = {
                "uptime_percentage": {},
                "average_load": {},
                "average_latency": {},
                "status_changes": {}
            }
            
            for server, data in server_stats.items():
                # Trier les données par timestamp
                health_data = sorted(data["health"], key=lambda x: x[0])
                load_data = sorted(data["load"], key=lambda x: x[0])
                latency_data = sorted(data["latency"], key=lambda x: x[0])
                
                # Calculer le pourcentage de disponibilité
                if health_data:
                    uptime = sum(1 for _, status in health_data if status) / len(health_data) * 100
                    result["uptime_percentage"][server] = uptime
                    
                    # Compter les changements d'état
                    status_changes = 0
                    for i in range(1, len(health_data)):
                        if health_data[i][1] != health_data[i-1][1]:
                            status_changes += 1
                    result["status_changes"][server] = status_changes
                
                # Calculer la charge moyenne
                if load_data:
                    avg_load = sum(load for _, load in load_data) / len(load_data)
                    result["average_load"][server] = avg_load
                
                # Calculer la latence moyenne
                if latency_data:
                    avg_latency = sum(latency for _, latency in latency_data) / len(latency_data)
                    result["average_latency"][server] = avg_latency
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats depuis la DB: {str(e)}")
            return {
                "uptime_percentage": {},
                "average_load": {},
                "average_latency": {},
                "status_changes": {}
            }

    def get_aggregated_stats(self, period: str = "hourly", days: int = 7) -> Dict[str, Any]:
        """
        Retourne des statistiques agrégées pour une période donnée.
        
        Args:
            period: Période d'agrégation ('hourly', 'daily', 'weekly')
            days: Nombre de jours d'historique à analyser
            
        Returns:
            Dict[str, Any]: Statistiques agrégées
        """
        try:
            db = get_db()
            cutoff_time = time.time() - (days * 24 * 3600)
            iso_cutoff = datetime.datetime.fromtimestamp(cutoff_time).isoformat()
            
            # Récupérer toutes les statistiques sur la période
            stats = db.search("server_stats", lambda q: q.timestamp >= iso_cutoff)
            
            # Préparer le résultat
            result = {}
            
            # Définir l'intervalle de temps selon la période
            if period == "hourly":
                interval_seconds = 3600
            elif period == "daily":
                interval_seconds = 24 * 3600
            elif period == "weekly":
                interval_seconds = 7 * 24 * 3600
            else:
                interval_seconds = 3600  # Par défaut: horaire
            
            # Regrouper par serveur, par type et par intervalle
            for stat in stats:
                server = stat["server"]
                stat_type = stat["type"]
                ts = datetime.datetime.fromisoformat(stat["timestamp"]).timestamp()
                value = stat["value"]
                
                # Calculer l'intervalle de temps
                interval_start = int(ts / interval_seconds) * interval_seconds
                interval_key = datetime.datetime.fromtimestamp(interval_start).isoformat()
                
                # Créer les dictionnaires si nécessaire
                if server not in result:
                    result[server] = {}
                
                if stat_type not in result[server]:
                    result[server][stat_type] = {}
                
                if interval_key not in result[server][stat_type]:
                    result[server][stat_type][interval_key] = {
                        "count": 0,
                        "sum": 0,
                        "min": float('inf'),
                        "max": float('-inf')
                    }
                
                # Ajouter les valeurs pour les agrégations
                interval_data = result[server][stat_type][interval_key]
                interval_data["count"] += 1
                
                if stat_type in ["load", "latency"]:
                    # Pour les métriques numériques
                    value = float(value)
                    interval_data["sum"] += value
                    interval_data["min"] = min(interval_data["min"], value)
                    interval_data["max"] = max(interval_data["max"], value)
                elif stat_type == "health":
                    # Pour la santé (booléen)
                    value = bool(value)
                    interval_data["sum"] += 1 if value else 0
            
            # Calculer les moyennes
            for server in result:
                for stat_type in result[server]:
                    for interval in result[server][stat_type]:
                        data = result[server][stat_type][interval]
                        
                        if data["count"] > 0:
                            if stat_type in ["load", "latency"]:
                                data["avg"] = data["sum"] / data["count"]
                            elif stat_type == "health":
                                data["availability"] = (data["sum"] / data["count"]) * 100
                            
                            # Nettoyer les valeurs infinies
                            if data["min"] == float('inf'):
                                data["min"] = 0
                            if data["max"] == float('-inf'):
                                data["max"] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'agrégation des statistiques: {str(e)}")
            return {}


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