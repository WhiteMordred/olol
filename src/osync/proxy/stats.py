"""Statistics tracking for Ollama proxy server with persistent storage."""

import threading
import time
import datetime
import logging
from typing import Dict, Any, List, Optional
from osync.proxy.db.database import get_db

# Configuration du logging
logger = logging.getLogger(__name__)

# Statistics state management
stats_lock = threading.Lock()
request_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "generate_requests": 0,
    "chat_requests": 0,
    "embedding_requests": 0,
    "server_stats": {},
    "start_time": time.time(),
    "last_persist_time": 0,  # Timestamp de la dernière persistance
    "average_latency_ms": 0.0,
    "latency_samples": 0
}

# Intervalle de persistance en secondes (toutes les 10 minutes)
PERSIST_INTERVAL = 600


def update_request_stats(request_type: str, increment: bool = True, latency_ms: Optional[float] = None) -> None:
    """Update request statistics.
    
    Args:
        request_type: Type of request ('chat', 'generate', 'embedding')
        increment: True to increment, False to decrement (for active requests)
        latency_ms: Latence de la requête en millisecondes (optionnel)
    """
    with stats_lock:
        # Always increment total
        if increment:
            request_stats["total_requests"] += 1
            request_stats[f"{request_type}_requests"] += 1
            
        # Update active count
        delta = 1 if increment else -1
        request_stats["active_requests"] += delta
        
        # Ensure we don't go negative
        if request_stats["active_requests"] < 0:
            request_stats["active_requests"] = 0
        
        # Mettre à jour les statistiques de latence si fournies
        if latency_ms is not None and increment:
            current_avg = request_stats["average_latency_ms"]
            current_samples = request_stats["latency_samples"]
            
            # Calculer la nouvelle moyenne
            new_samples = current_samples + 1
            new_avg = ((current_avg * current_samples) + latency_ms) / new_samples
            
            request_stats["average_latency_ms"] = new_avg
            request_stats["latency_samples"] = new_samples
        
        # Vérifier si nous devons persister les statistiques
        current_time = time.time()
        if current_time - request_stats["last_persist_time"] > PERSIST_INTERVAL:
            persist_stats()
            request_stats["last_persist_time"] = current_time


def update_server_stats(server_address: str, stats_data: Dict[str, Any]) -> None:
    """Update statistics for a specific server.
    
    Args:
        server_address: Address of the server
        stats_data: Dictionary of statistics to update
    """
    with stats_lock:
        if server_address not in request_stats["server_stats"]:
            request_stats["server_stats"][server_address] = {}
            
        # Update with new data
        request_stats["server_stats"][server_address].update(stats_data)
        # Add timestamp of last update
        request_stats["server_stats"][server_address]["last_update"] = time.time()


def get_stats_snapshot() -> Dict[str, Any]:
    """Get a snapshot of the current statistics.
    
    Returns:
        Dictionary containing statistics
    """
    with stats_lock:
        # Return a copy to avoid modification while being used
        stats_copy = request_stats.copy()
        
        # Calculer des métriques supplémentaires
        uptime = time.time() - stats_copy["start_time"]
        stats_copy["uptime_seconds"] = uptime
        
        if uptime > 0:
            stats_copy["requests_per_second"] = stats_copy["total_requests"] / uptime
        else:
            stats_copy["requests_per_second"] = 0
            
        return stats_copy


def reset_stats() -> None:
    """Reset all statistics except for server_stats and start_time."""
    with stats_lock:
        server_stats = request_stats["server_stats"].copy()
        start_time = request_stats["start_time"]
        last_persist_time = time.time()
        
        # Reset counters
        request_stats.update({
            "total_requests": 0,
            "active_requests": 0,
            "generate_requests": 0,
            "chat_requests": 0,
            "embedding_requests": 0,
            "server_stats": server_stats,
            "start_time": start_time,
            "last_persist_time": last_persist_time,
            "average_latency_ms": 0.0,
            "latency_samples": 0
        })
        
        # Persister que les stats ont été réinitialisées
        persist_stats(reset=True)


def persist_stats(reset: bool = False) -> None:
    """Persiste les statistiques dans TinyDB.
    
    Args:
        reset: Indique si les statistiques ont été réinitialisées
    """
    try:
        db = get_db()
        current_time = time.time()
        current_iso = datetime.datetime.fromtimestamp(current_time).isoformat()
        
        # Préparer les données
        with stats_lock:
            data = {
                "timestamp": current_iso,
                "period": "hourly",  # Par défaut, nous agrégeons par heure
                "total_requests": request_stats["total_requests"],
                "generate_requests": request_stats["generate_requests"],
                "chat_requests": request_stats["chat_requests"],
                "embedding_requests": request_stats["embedding_requests"],
                "average_latency_ms": request_stats["average_latency_ms"],
                "reset": reset
            }
        
        # Insérer dans la base de données
        db.insert("request_stats", data)
        
        logger.debug(f"Statistiques de requêtes persistées: {data['total_requests']} requêtes au total")
        
    except Exception as e:
        logger.error(f"Erreur lors de la persistance des statistiques: {str(e)}")


def aggregate_stats(period: str = "hourly") -> None:
    """Agrège les statistiques pour la période spécifiée.
    
    Args:
        period: Période d'agrégation ('hourly', 'daily', 'weekly')
    """
    try:
        db = get_db()
        current_time = time.time()
        
        # Déterminer la période de coupure
        if period == "hourly":
            cutoff_seconds = 3600
            new_period = "hourly"
        elif period == "daily":
            cutoff_seconds = 24 * 3600
            new_period = "daily"
        elif period == "weekly":
            cutoff_seconds = 7 * 24 * 3600
            new_period = "weekly"
        else:
            logger.error(f"Période d'agrégation non reconnue: {period}")
            return
        
        cutoff_time = current_time - cutoff_seconds
        iso_cutoff = datetime.datetime.fromtimestamp(cutoff_time).isoformat()
        
        # Récupérer les statistiques à agréger
        stats_to_aggregate = db.search(
            "request_stats", 
            lambda q: q.timestamp < iso_cutoff and q.period == period and not q.reset
        )
        
        # S'il n'y a pas assez de données, ne pas agréger
        if len(stats_to_aggregate) < 2:
            return
        
        # Agréger les données
        total_requests = 0
        generate_requests = 0
        chat_requests = 0
        embedding_requests = 0
        latency_sum = 0.0
        latency_count = 0
        
        earliest_timestamp = None
        latest_timestamp = None
        
        for stat in stats_to_aggregate:
            ts = datetime.datetime.fromisoformat(stat["timestamp"]).timestamp()
            
            if earliest_timestamp is None or ts < earliest_timestamp:
                earliest_timestamp = ts
                
            if latest_timestamp is None or ts > latest_timestamp:
                latest_timestamp = ts
            
            total_requests += stat.get("total_requests", 0)
            generate_requests += stat.get("generate_requests", 0)
            chat_requests += stat.get("chat_requests", 0)
            embedding_requests += stat.get("embedding_requests", 0)
            
            latency = stat.get("average_latency_ms")
            if latency is not None:
                latency_sum += latency
                latency_count += 1
        
        # Calculer la latence moyenne
        average_latency = latency_sum / latency_count if latency_count > 0 else 0
        
        # Créer l'enregistrement agrégé
        aggregated_data = {
            "timestamp": datetime.datetime.fromtimestamp(latest_timestamp).isoformat(),
            "period": new_period,
            "start_time": datetime.datetime.fromtimestamp(earliest_timestamp).isoformat(),
            "end_time": datetime.datetime.fromtimestamp(latest_timestamp).isoformat(),
            "total_requests": total_requests,
            "generate_requests": generate_requests,
            "chat_requests": chat_requests,
            "embedding_requests": embedding_requests,
            "average_latency_ms": average_latency,
            "aggregated": True
        }
        
        # Insérer les données agrégées
        db.insert("request_stats", aggregated_data)
        
        # Supprimer les données agrégées
        for stat in stats_to_aggregate:
            db.remove("request_stats", lambda q: q.timestamp == stat["timestamp"])
        
        logger.info(f"Agrégation des stats {period} terminée: {len(stats_to_aggregate)} entrées en 1 agrégat")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'agrégation des statistiques: {str(e)}")


def get_historical_stats(period: str = "daily", days: int = 7) -> List[Dict[str, Any]]:
    """Récupère les statistiques historiques pour la période spécifiée.
    
    Args:
        period: Période d'agrégation ('hourly', 'daily', 'weekly')
        days: Nombre de jours d'historique à récupérer
        
    Returns:
        Liste de dictionnaires contenant les statistiques historiques
    """
    try:
        db = get_db()
        cutoff_time = time.time() - (days * 24 * 3600)
        iso_cutoff = datetime.datetime.fromtimestamp(cutoff_time).isoformat()
        
        # Récupérer les statistiques historiques
        stats = db.search(
            "request_stats", 
            lambda q: q.timestamp >= iso_cutoff and q.period == period
        )
        
        # Trier par timestamp
        stats.sort(key=lambda x: x["timestamp"])
        
        return stats
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques historiques: {str(e)}")
        return []


def cleanup_old_stats(days_to_keep: int = 90) -> None:
    """Nettoie les anciennes statistiques.
    
    Args:
        days_to_keep: Nombre de jours de statistiques à conserver
    """
    try:
        db = get_db()
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        iso_cutoff = datetime.datetime.fromtimestamp(cutoff_time).isoformat()
        
        # Supprimer les anciennes statistiques
        removed = db.remove("request_stats", lambda q: q.timestamp < iso_cutoff)
        
        if removed:
            logger.info(f"Nettoyage des anciennes statistiques: {removed} entrées supprimées")
            
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des anciennes statistiques: {str(e)}")


# Planificateur d'agrégation
def _aggregation_scheduler():
    """Fonction pour exécuter les agrégations périodiquement."""
    try:
        # Agrégation horaire -> journalière
        aggregate_stats("hourly")
        
        # Agrégation journalière -> hebdomadaire (une fois par semaine)
        current_day = datetime.datetime.now().weekday()
        if current_day == 0:  # Lundi
            aggregate_stats("daily")
        
    except Exception as e:
        logger.error(f"Erreur dans le planificateur d'agrégation: {str(e)}")


def schedule_aggregations(interval_seconds: int = 3600 * 6) -> threading.Thread:
    """Démarre un thread pour exécuter les agrégations périodiquement.
    
    Args:
        interval_seconds: Intervalle entre les agrégations en secondes (par défaut 6 heures)
        
    Returns:
        Thread d'agrégation
    """
    stop_event = threading.Event()
    
    def aggregation_thread():
        while not stop_event.is_set():
            try:
                _aggregation_scheduler()
                # Nettoyer les anciennes statistiques une fois par semaine
                current_day = datetime.datetime.now().weekday()
                if current_day == 0:  # Lundi
                    cleanup_old_stats()
            except Exception as e:
                logger.error(f"Erreur dans le thread d'agrégation: {str(e)}")
                
            # Attendre l'intervalle ou jusqu'à ce que le thread soit arrêté
            stop_event.wait(interval_seconds)
    
    thread = threading.Thread(target=aggregation_thread, daemon=True)
    thread.start()
    
    return thread, stop_event


# Démarrer le planificateur d'agrégation au chargement du module
aggregation_thread, stop_aggregation = schedule_aggregations()