import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from flask import Flask, Response, jsonify, request, render_template, stream_with_context, url_for

from osync.sync.client import OllamaClient
from osync.utils.cluster import OllamaCluster

# Import local modules
from .stats import update_request_stats, request_stats, stats_lock
from .console_ui import RichUI as ConsoleUI, run_console_ui, ui_exit_event
from .utils import create_grpc_client, adjust_context_length
from .health import health_checker

# Import API modules
from .api.routes import register_api_routes
from .api.swagger import init_swagger
from .cluster.manager import get_cluster_manager, ClusterManager
from .cluster.health import get_health_monitor, start_health_monitoring
from .queue.queue import get_queue_manager

try:
    from osync.rpc.coordinator import InferenceCoordinator
    DISTRIBUTED_INFERENCE_AVAILABLE = True
except ImportError:
    DISTRIBUTED_INFERENCE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for Flask app
app = Flask(__name__, 
            static_folder='web/static', 
            template_folder='web/templates')
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
cluster: Optional[OllamaCluster] = None
cluster_manager: Optional[ClusterManager] = None
coordinator: Optional['InferenceCoordinator'] = None
health_check_interval = 30  # seconds
use_distributed_inference = False  # Set to True to enable distributed inference

# UI state
ui_active = False
ui_thread = None

# Ajout d'un filtre personnalisé yesno (similaire à celui de Django)
@app.template_filter('yesno')
def yesno_filter(value, choices="oui,non"):
    choices_list = choices.split(",")
    if len(choices_list) < 2:
        choices_list = ["oui", "non"]
    
    return choices_list[0] if value else choices_list[1]

# Ajout d'un filtre personnalisé date pour formater les timestamps
@app.template_filter('date')
def date_filter(value, format='%d/%m/%Y %H:%M:%S'):
    """Formater un timestamp (nombre à virgule flottante) en date lisible."""
    if not value:
        return ""
    try:
        # Convertir le timestamp (secondes depuis epoch) en objet datetime
        dt = datetime.fromtimestamp(float(value))
        return dt.strftime(format)
    except (ValueError, TypeError):
        # En cas d'erreur, retourner la valeur inchangée
        return value

# Web routes
@app.route('/')
def index():
    """Render the dashboard page."""
    return render_template('dashboard.html', 
                          stats=get_cluster_stats(), 
                          servers=get_servers_list(),
                          popular_models=get_popular_models(),
                          load_chart_data=get_load_chart_data(),
                          health_chart_data=get_health_chart_data())

@app.route('/servers')
def servers():
    """Render the servers management page."""
    stats = {
        'total': 0,
        'online': 0,
        'offline': 0,
        'capacity': 0
    }
    
    servers_list = get_servers_list()
    
    # Calculate stats
    stats['total'] = len(servers_list)
    stats['online'] = sum(1 for server in servers_list if server['healthy'])
    stats['offline'] = stats['total'] - stats['online']
    
    return render_template('servers.html', stats=stats, servers=servers_list)

@app.route('/models')
def models():
    """Render the models page."""
    return render_template('models.html', models=get_models_list())

@app.route('/health')
def health():
    """Render the health monitoring page."""
    health_monitor = get_health_monitor()
    health_report = health_monitor.get_health_report()
    
    return render_template('health.html', report=health_report)

@app.route('/playground')
def playground():
    """Render the model playground page."""
    model_name = request.args.get('model', '')
    models = get_models_list()
    
    return render_template('playground.html', 
                          selected_model=model_name,
                          models=models)

@app.route('/settings')
def settings():
    """Render the settings page."""
    return render_template('settings.html')

@app.route('/queue')
def queue():
    """Render the task queue page."""
    # Récupérer les données de la file d'attente via le gestionnaire de queue
    queue_manager = get_queue_manager()
    queue_stats = queue_manager.get_queue_stats()
    
    # Créer des collections de tâches filtrées par statut
    tasks = {
        'active': [],
        'pending': [],
        'completed': [],
        'failed': []
    }
    
    # Récupérer les requêtes de chaque statut en filtrant le cache de requêtes
    with queue_manager._queue_lock:  # Utiliser le verrou pour éviter les problèmes de concurrence
        for request_id, request in queue_manager._requests_cache.items():
            status = request.get("status", "")
            
            # Créer une version formatée de la tâche pour l'affichage
            formatted_task = format_task_for_display(request)
            
            if status == queue_manager.STATUS_PROCESSING:
                if len(tasks['active']) < 20:  # Limiter à 20 tâches actives
                    tasks['active'].append(formatted_task)
            elif status == queue_manager.STATUS_PENDING:
                if len(tasks['pending']) < 20:  # Limiter à 20 tâches en attente
                    tasks['pending'].append(formatted_task)
            elif status == queue_manager.STATUS_COMPLETED:
                if len(tasks['completed']) < 12:  # Limiter à 12 tâches complétées
                    tasks['completed'].append(formatted_task)
            elif status in [queue_manager.STATUS_FAILED, queue_manager.STATUS_CANCELED]:
                if len(tasks['failed']) < 5:  # Limiter à 5 tâches échouées
                    tasks['failed'].append(formatted_task)
    
    # Trier les collections par date de création ou mise à jour
    tasks['active'].sort(key=lambda x: x.get('started_at', ''), reverse=True)
    tasks['pending'].sort(key=lambda x: x.get('created_at', ''), reverse=True)
    tasks['completed'].sort(key=lambda x: x.get('completed_at', ''), reverse=True)
    tasks['failed'].sort(key=lambda x: x.get('completed_at', ''), reverse=True)
    
    # Ajouter le nombre total de tâches pour l'affichage dans le footer
    tasks['total'] = len(tasks['active']) + len(tasks['pending']) + len(tasks['completed']) + len(tasks['failed'])
    
    # Préparer les données pour les graphiques
    queue_data = {
        'active': queue_stats.get('processing', 0),
        'pending': queue_stats.get('pending', 0),
        'completed': queue_stats.get('completed', 0),
        'failed': queue_stats.get('failed', 0) + queue_stats.get('canceled', 0)
    }
    
    # Données pour le graphique de performance (générées pour l'exemple)
    performance_data = generate_performance_data()
    
    # Récupérer la liste des modèles et serveurs disponibles pour le modal "Nouvelle tâche"
    available_models = get_models_list()
    available_servers = get_servers_list()
    
    return render_template('queue.html', 
                          stats=queue_stats, 
                          tasks=tasks,
                          queue_data=queue_data,
                          performance_data=performance_data,
                          available_models=available_models,
                          available_servers=available_servers)

def format_task_for_display(task):
    """
    Formate une tâche pour l'affichage dans l'interface utilisateur.
    """
    formatted = task.copy()
    
    # Formater les dates relatives
    created_at = task.get('created_at')
    started_at = task.get('started_at')
    completed_at = task.get('completed_at')
    
    if created_at:
        try:
            created_time = datetime.fromisoformat(created_at)
            formatted['created_relative'] = format_relative_time(created_time)
        except (ValueError, TypeError):
            formatted['created_relative'] = "Inconnu"
    
    if started_at:
        try:
            started_time = datetime.fromisoformat(started_at)
            formatted['started_relative'] = format_relative_time(started_time)
        except (ValueError, TypeError):
            formatted['started_relative'] = "Inconnu"
    
    if completed_at:
        try:
            completed_time = datetime.fromisoformat(completed_at)
            formatted['completed_relative'] = format_relative_time(completed_time)
            
            if started_at:
                try:
                    started_time = datetime.fromisoformat(started_at)
                    duration_seconds = (completed_time - started_time).total_seconds()
                    formatted['duration'] = format_duration(duration_seconds)
                except (ValueError, TypeError):
                    formatted['duration'] = "Inconnu"
        except (ValueError, TypeError):
            formatted['completed_relative'] = "Inconnu"
    
    # Formater l'état avec la classe CSS appropriée
    status = task.get('status', '').lower()
    if status == 'processing':
        formatted['status'] = "En cours"
        formatted['status_class'] = "info"
    elif status == 'pending':
        formatted['status'] = "En attente"
        formatted['status_class'] = "secondary"
    elif status == 'completed':
        formatted['status'] = "Terminée"
        formatted['status_class'] = "success"
    elif status == 'failed':
        formatted['status'] = "Échouée"
        formatted['status_class'] = "danger"
        formatted['error'] = task.get('failed_reason', 'Erreur inconnue')
    elif status == 'canceled':
        formatted['status'] = "Annulée"
        formatted['status_class'] = "warning"
        formatted['error'] = task.get('canceled_reason', 'Annulation')
    
    # Formater la priorité
    priority = task.get('priority', 10)
    if priority > 15:
        formatted['priority'] = "Haute"
        formatted['priority_class'] = "danger"
    elif priority > 5:
        formatted['priority'] = "Normale"
        formatted['priority_class'] = "secondary"
    else:
        formatted['priority'] = "Basse"
        formatted['priority_class'] = "info"
    
    # Formater la progression (simulée pour l'exemple)
    if status == 'processing':
        # Calculer une progression basée sur le temps écoulé (simulé)
        if started_at:
            try:
                started_time = datetime.fromisoformat(started_at)
                elapsed = (datetime.now() - started_time).total_seconds()
                # Supposer que les tâches prennent en moyenne 60 secondes
                progress = min(int(elapsed / 60.0 * 100), 95)
                formatted['progress'] = progress
            except (ValueError, TypeError):
                formatted['progress'] = 50  # Valeur par défaut
        else:
            formatted['progress'] = 10  # Valeur par défaut si pas de temps de démarrage
    
    # Server assignment
    formatted['server'] = task.get('server_assigned', 'Auto')
    
    # Position in queue (for pending tasks)
    if status == 'pending':
        formatted['position'] = task.get('queue_position', 1)
    
    return formatted

def format_relative_time(dt):
    """
    Formate un datetime en temps relatif (ex: "il y a 5 min").
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    if seconds < 60:
        return "À l'instant"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"Il y a {minutes} min"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"Il y a {hours}h"
    else:
        days = int(seconds / 86400)
        return f"Il y a {days}j"

def format_duration(seconds):
    """
    Formate une durée en secondes en format lisible (ex: "45s", "2m 30s").
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds / 3600)
        remaining_minutes = int((seconds % 3600) / 60)
        return f"{hours}h {remaining_minutes}m"

def generate_performance_data():
    """
    Génère des données de performance simulées pour le graphique.
    """
    # Labels pour les dernières heures
    labels = []
    for i in range(8, 0, -1):
        if i == 1:
            labels.append("Il y a 1h")
        else:
            labels.append(f"Il y a {i}h")
    labels.append("Maintenant")
    
    # Générer des données simulées de temps d'attente
    import random
    base_waiting_time = 15  # secondes
    waiting_times = []
    for _ in range(len(labels)):
        variation = random.uniform(-5, 10)
        waiting_times.append(max(0, base_waiting_time + variation))
    
    # Générer des données simulées de tâches par heure
    base_tasks_per_hour = 8
    tasks_per_hour = []
    for _ in range(len(labels)):
        variation = random.uniform(-3, 5)
        tasks_per_hour.append(max(1, round(base_tasks_per_hour + variation)))
    
    return {
        'labels': labels,
        'waiting_times': waiting_times,
        'tasks_per_hour': tasks_per_hour
    }

@app.route('/log')
def logs():
    """Render the logs page."""
    # Récupérer les paramètres de requête pour la pagination et le filtrage
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    level = request.args.get('level', None)
    
    # Récupérer les journaux depuis un fichier de log ou la base de données
    # Au lieu d'utiliser une méthode inexistante, créer une structure de données appropriée
    log_entries = get_system_logs(level=level, page=page, limit=limit)
    
    # Calculer le nombre total de pages pour la pagination
    total_entries = len(log_entries)
    pages_total = max(1, total_entries // limit + (1 if total_entries % limit else 0))
    
    return render_template('log.html', 
                          logs=log_entries, 
                          current_page=page, 
                          pages_total=pages_total,
                          selected_level=level)

def get_system_logs(level=None, page=1, limit=50):
    """
    Récupère les journaux système avec pagination et filtrage.
    
    Args:
        level: Niveau de log pour le filtrage (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        page: Numéro de page pour la pagination
        limit: Nombre d'entrées par page
    
    Returns:
        Liste des entrées de journal formatées
    """
    # Utiliser sync_manager pour accéder aux logs stockés dans la base de données
    sync_manager = get_sync_manager()
    
    # Essayer de récupérer les journaux depuis la table system_logs
    try:
        logs_data = sync_manager.read_from_ram("system_logs")
    except Exception:
        # Si la table n'existe pas, créer une liste vide
        logs_data = []
    
    # Si pas de logs dans la base, créer quelques logs de base pour indiquer que le système fonctionne
    if not logs_data:
        from datetime import datetime, timedelta
        import logging
        
        # Obtenir le temps de démarrage du serveur depuis request_stats
        with stats_lock:
            start_time = datetime.fromtimestamp(request_stats.get("start_time", datetime.now().timestamp()))
        
        # Créer un log indiquant le démarrage du serveur
        logs_data = [{
            "timestamp": start_time.isoformat(),
            "level": "INFO",
            "component": "osync.proxy.app",
            "message": "Serveur proxy démarré",
            "details": {
                "pid": os.getpid() if hasattr(os, "getpid") else None,
                "host": request.host if request else "localhost"
            }
        }]
        
        # Ajouter des logs pour indiquer l'initialisation des composants
        components = [
            ("osync.proxy.cluster.manager", "Gestionnaire de cluster initialisé"),
            ("osync.proxy.db.database", "Base de données connectée"),
            ("osync.proxy.web", "Interface web démarrée")
        ]
        
        for i, (component, message) in enumerate(components):
            timestamp = start_time + timedelta(seconds=i+1)
            logs_data.append({
                "timestamp": timestamp.isoformat(),
                "level": "INFO",
                "component": component,
                "message": message,
                "details": {}
            })
    
    # Filtrer par niveau si spécifié
    if level:
        logs_data = [log for log in logs_data if log.get("level", "").upper() == level.upper()]
    
    # Trier les logs par timestamp décroissant (plus récent en premier)
    logs_data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Appliquer la pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_logs = logs_data[start_idx:end_idx] if start_idx < len(logs_data) else []
    
    # Formater les logs pour l'affichage
    formatted_logs = []
    for log in paginated_logs:
        # Formater le timestamp pour l'affichage
        timestamp = log.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            formatted_time = timestamp
        
        # Construire l'entrée de log formatée
        formatted_log = {
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "level": log.get("level", "INFO"),
            "component": log.get("component", "system"),
            "message": log.get("message", ""),
            "details": log.get("details", {})
        }
        formatted_logs.append(formatted_log)
    
    return formatted_logs

@app.route('/terminal')
def terminal():
    """Render the terminal page."""
    return render_template('terminal.html')

@app.route('/swagger')
def swagger_ui():
    """Render the Swagger UI integrated page."""
    return render_template('swagger.html')

# Helper functions for the web UI
def get_cluster_stats():
    """Get cluster statistics for the dashboard."""
    stats = {
        'servers': {'total': 0, 'healthy': 0},
        'models': {'total': 0, 'available': 0},
        'load': {'average': 0.0, 'max': 0.0},
        'latency': {'average': 0.0, 'min': 0.0},
        'uptime': format_uptime(time.time() - request_stats["start_time"]),
        'version': '1.0.0'
    }
    
    # Get cluster manager data
    cm = get_cluster_manager()
    if cm:
        servers = cm.get_server_addresses()
        stats['servers']['total'] = len(servers)
        stats['servers']['healthy'] = len(cm.get_healthy_servers())
        
        # Get models data
        models_dict = cm.get_all_models()
        stats['models']['total'] = len(models_dict)
        stats['models']['available'] = len([m for m in models_dict if any(models_dict[m])])
        
        # Calculate load statistics
        if servers:
            loads = [cm.get_server_load(server) for server in servers]
            stats['load']['average'] = sum(loads) / len(loads) if loads else 0
            stats['load']['max'] = max(loads) if loads else 0
    
    return stats

def get_servers_list():
    """Get server list data for UI."""
    servers = []
    cm = get_cluster_manager()
    
    if cm:
        for addr in cm.get_server_addresses():
            server_info = {
                'address': addr,
                'healthy': cm.get_server_health(addr),
                'load': cm.get_server_load(addr),
                'latency_ms': 0.0,  # Will be set from health monitor if available
                'models': cm.get_all_models()
            }
            
            # Try to get latency from health monitor
            health_monitor = get_health_monitor()
            server_latency = 0.0
            try:
                report = health_monitor.get_health_report()
                if addr in report['servers']:
                    server_latency = report['servers'][addr].get('latency_ms', 0.0)
            except Exception:
                pass
                
            server_info['latency_ms'] = server_latency
            
            # Get models available on this server
            models = []
            all_models = cm.get_all_models()
            for model_name, servers_list in all_models.items():
                if addr in servers_list:
                    models.append(model_name)
            server_info['models'] = models
            
            servers.append(server_info)
    
    return servers

def get_models_list():
    """Get model list data for UI."""
    models = []
    cm = get_cluster_manager()
    
    if cm:
        all_models = cm.get_all_models()
        for model_name, servers_list in all_models.items():
            model_info = {
                'name': model_name,
                'servers': list(servers_list),
                'available': len(servers_list) > 0,
                'size': 0,
                'version': 'unknown',
                'modified_at': None
            }
            models.append(model_info)
    
    return models

def get_popular_models(limit=5):
    """Get list of popular models for dashboard."""
    models = get_models_list()
    # Sort by number of servers (more servers = more popular)
    models.sort(key=lambda m: len(m['servers']), reverse=True)
    return models[:limit]

def get_load_chart_data():
    """Generate load chart data for dashboard."""
    # Default empty chart data
    chart_data = {
        'labels': [],
        'datasets': []
    }
    
    cm = get_cluster_manager()
    if not cm:
        return chart_data
    
    # Get servers
    servers = cm.get_server_addresses()
    if not servers:
        return chart_data
    
    # Create a sample dataset for demonstration
    # In a real implementation, this would use historical data from the health monitor
    chart_data['labels'] = [f"{i}m ago" for i in range(60, 0, -5)]
    datasets = []
    
    # Generate random-like data for each server
    # Convert set to list before slicing
    servers_list = list(servers)
    for i, server in enumerate(servers_list[:5]):  # Limit to 5 servers for clarity
        base_load = cm.get_server_load(server)
        
        # Generate data points with small variations
        import random
        data_points = []
        for _ in range(len(chart_data['labels'])):
            variation = random.uniform(-0.05, 0.05)
            load = max(0, min(1, base_load + variation))
            data_points.append(load)
        
        # Generate a color based on index
        colors = ['rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)', 
                 'rgba(255, 206, 86, 0.5)', 'rgba(75, 192, 192, 0.5)', 
                 'rgba(153, 102, 255, 0.5)']
        color = colors[i % len(colors)]
        
        datasets.append({
            'label': server,
            'data': data_points,
            'backgroundColor': color,
            'borderColor': color.replace('0.5', '1'),
            'borderWidth': 1,
            'fill': False
        })
    
    chart_data['datasets'] = datasets
    return chart_data

def get_health_chart_data():
    """Generate health chart data for dashboard."""
    # Default empty chart data
    chart_data = {
        'labels': [],
        'datasets': []
    }
    
    cm = get_cluster_manager()
    if not cm:
        return chart_data
    
    # Get servers
    servers = cm.get_server_addresses()
    if not servers:
        return chart_data
    
    # Create a sample dataset for demonstration
    # In a real implementation, this would use historical data from the health monitor
    chart_data['labels'] = [f"{i}m ago" for i in range(60, 0, -5)]
    datasets = []
    
    # Generate health data for each server
    # Convert set to list before slicing
    servers_list = list(servers)
    for i, server in enumerate(servers_list[:5]):  # Limit to 5 servers for clarity
        current_health = 1 if cm.get_server_health(server) else 0
        
        # Generate data points with occasional health changes
        import random
        data_points = []
        for _ in range(len(chart_data['labels'])):
            # 90% chance to keep same status, 10% to change
            if random.random() < 0.1:
                current_health = 1 - current_health  # Flip between 0 and 1
            data_points.append(current_health)
        
        # Generate a color based on index
        colors = ['rgba(54, 162, 235, 0.5)', 'rgba(255, 99, 132, 0.5)', 
                 'rgba(255, 206, 86, 0.5)', 'rgba(75, 192, 192, 0.5)', 
                 'rgba(153, 102, 255, 0.5)']
        color = colors[i % len(colors)]
        
        datasets.append({
            'label': server,
            'data': data_points,
            'backgroundColor': color,
            'borderColor': color.replace('0.5', '1'),
            'borderWidth': 1,
            'fill': False,
            'stepped': True
        })
    
    chart_data['datasets'] = datasets
    return chart_data

def format_uptime(seconds):
    """Format uptime in seconds to human readable format."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{int(days)}j {int(hours)}h {int(minutes)}m"
    elif hours > 0:
        return f"{int(hours)}h {int(minutes)}m"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

def run_proxy(host: str = "0.0.0.0", port: int = 8000, 
           server_addresses: List[str] = None,
           enable_distributed: bool = False,
           auto_distribute_large: bool = True,
           rpc_servers: Optional[List[str]] = None,
           enable_discovery: bool = True,
           preferred_interface: Optional[str] = None,
           enable_ui: bool = True,
           enable_web: bool = True,
           verbose: bool = False,
           debug: bool = False) -> None:
    """Start the proxy server.
    
    Args:
        host: Host to bind the proxy server to
        port: Port to bind the proxy server to
        server_addresses: List of gRPC server addresses in "host:port" format
        enable_distributed: Whether to enable distributed inference mode
        auto_distribute_large: Whether to automatically use distributed inference for large models
        rpc_servers: List of RPC servers for distributed inference
        enable_discovery: Enable auto-discovery of RPC servers
        preferred_interface: Preferred network interface IP address for connections
        enable_ui: Whether to enable the console UI
        enable_web: Whether to enable the web interface and Swagger API
        verbose: Enable verbose logging and detailed UI status updates 
        debug: Enable debug mode with maximum verbosity
    """
    global cluster, cluster_manager, coordinator, use_distributed_inference, ui_thread, ui_active
    
    if server_addresses is None or not server_addresses:
        server_addresses = ["localhost:50051"]
    
    # Set up logging based on verbosity settings
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - maximum verbosity")
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.info("Verbose mode enabled")
    else:
        # Default logging
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        
    logger.info(f"Starting proxy with server addresses: {server_addresses}")
    
    # Set distributed inference flag
    use_distributed_inference = enable_distributed and DISTRIBUTED_INFERENCE_AVAILABLE
    
    if use_distributed_inference and not rpc_servers:
        # Default to using the same servers for RPC
        rpc_servers = server_addresses
        # Switch port from 50051 to 50052 if needed
        rpc_servers = [addr.replace(':50051', ':50052') for addr in rpc_servers]
        logger.info(f"Using RPC servers for distributed inference: {rpc_servers}")
    
    # Initialize the cluster for regular load balancing
    cluster = OllamaCluster(server_addresses)
    
    # Initialize the cluster manager (refactored version)
    cluster_manager = get_cluster_manager()
    cluster_manager.initialize({"server_addresses": server_addresses})
    
    # Start health monitoring
    start_health_monitoring()
    
    # Initialize coordinator for distributed inference if enabled
    if use_distributed_inference:
        try:
            coordinator = InferenceCoordinator(rpc_servers)
            logger.info("Distributed inference coordinator initialized")
            
            if auto_distribute_large:
                logger.info("Auto-distribution enabled for large models")
        except Exception as e:
            logger.error(f"Failed to initialize distributed inference: {e}")
            use_distributed_inference = False
    
    # Create health checker thread but don't start immediately
    # We'll start it after discovery service has had a chance to find servers
    health_thread = threading.Thread(target=health_checker, daemon=True)
    
    # Start discovery service for auto-discovery with servers
    discovery_service = None
    if enable_discovery:
        try:
            # Import the discovery service
            from osync.utils.discovery import DiscoveryService
            
            # Create and start the discovery service
            discovery_service = DiscoveryService(
                service_type="proxy",
                service_port=port,
                extra_info={
                    "service_type": "proxy",
                    "distributed_enabled": use_distributed_inference,
                    "auto_distribute_large": auto_distribute_large
                },
                preferred_interface=preferred_interface
            )
            
            # Define a callback function to handle newly discovered servers
            def on_server_discovered(service_id: str, service_info: Dict[str, Any]) -> None:
                # Extract information about the server
                if service_info.get("service_type") != "server":
                    return
                    
                # Get the server address, using best interface if available
                ip = service_info.get("best_ip") or service_info.get("ip")
                port = service_info.get("port", 50052)
                
                # Format differently for IPv6
                if ':' in ip and not ip.startswith('localhost'):
                    server_address = f"[{ip}]:{port}"
                else:
                    server_address = f"{ip}:{port}"
                
                # Get connection endpoints if available - these include properly formatted host:port strings
                connection_endpoints = service_info.get("connection_endpoints", [])
                
                # Also get any reachable IPs from discovery
                reachable_ips = service_info.get("reachable_ips", [])
                if ip not in reachable_ips and ip:
                    reachable_ips.append(ip)
                
                # Get capabilities
                capabilities = service_info.get("capabilities", {})
                device_type = capabilities.get("device_type", "unknown")
                
                # Log the discovery
                logger.info(f"Discovered server: {server_address} (type: {device_type}, ID: {service_id})")
                
                # Prepare connection details dictionary
                connection_details = {
                    "service_id": service_id,
                    "best_ip": ip,
                    "reachable_ips": reachable_ips,
                    "connection_endpoints": connection_endpoints,
                    "source_port": service_info.get("source_port", port),
                    "discovered_at": time.time(),
                    "capabilities": capabilities
                }
                
                # Check if this is a new server
                if server_address not in cluster.server_addresses:
                    logger.info(f"Adding newly discovered server: {server_address}")
                    # Update the cluster with the new server and its connection details
                    cluster.add_server(server_address, connection_details)
                    
                    # Also update the cluster manager
                    cluster_manager.initialize({"server_addresses": list(cluster.server_addresses)})
                else:
                    # Update connection details for existing server
                    cluster.register_connection_details(server_address, connection_details)
                    
                # If this is an RPC server and distributed inference is enabled,
                # add it to the RPC servers list
                if (use_distributed_inference and
                    capabilities.get("service_type") == "rpc-server" and
                    server_address not in rpc_servers):
                    logger.info(f"Adding newly discovered RPC server: {server_address}")
                    rpc_servers.append(server_address)
                    
                    # Reinitialize coordinator if needed
                    if coordinator:
                        try:
                            # Update the coordinator with the new server
                            coordinator.client.server_addresses.append(server_address)
                            logger.info(f"Updated coordinator with new RPC server: {server_address}")
                        except Exception as e:
                            logger.error(f"Failed to update coordinator with new server: {e}")
            
            # Register the callback
            discovery_service.register_discovery_callback(on_server_discovered)
            
            # Start the discovery service
            discovery_service.start()
            logger.info("Auto-discovery service started")
            
            # Wait a short time for initial discoveries before starting health checker
            # This helps avoid health errors on servers we're about to discover
            time.sleep(2)
            
            # Now start the health checker thread after discovery has begun
            health_thread.start()
            logger.info("Health checker started after discovery")
        except ImportError as e:
            logger.warning(f"Auto-discovery not available: {e}")
            # Start health checker anyway since discovery isn't available
            health_thread.start()
        except Exception as e:
            logger.warning(f"Failed to start discovery service: {e}")
            # Start health checker anyway since discovery failed
            health_thread.start()
    else:
        # If discovery is disabled, start health checker immediately
        health_thread.start()
    
    # Start the console UI if enabled
    if enable_ui:
        # Set up parameters to control UI verbosity
        ui_params = {
            "verbose": verbose,
            "debug": debug
        }
        ui_active = True
        ui_exit_event.clear()
        ui_thread = threading.Thread(target=run_console_ui, args=(ui_params,), daemon=True)
        ui_thread.start()
        logger.info("Console UI started")
    else:
        # When UI is disabled, increase log level to reduce noise if not in verbose/debug mode
        if not verbose and not debug:
            logger.info("Console UI disabled, setting log level to WARNING")
            logging.getLogger().setLevel(logging.WARNING)
            # But keep our own logger at INFO for important messages
            logger.setLevel(logging.INFO)
    
    # Register API routes and Swagger if web interface is enabled
    if enable_web:
        # Import the OllamaProxyService
        from .api.services import OllamaProxyService
        
        # Create an instance of OllamaProxyService
        api_service = OllamaProxyService(cluster_manager)
        
        # Register API routes with the api_service
        register_api_routes(app, api_service)
        
        # Register Swagger blueprint
        swagger_bp = init_swagger()
        app.register_blueprint(swagger_bp, url_prefix='/api/v1')
        
        logger.info("Web interface and API enabled")
    
    # Start the Flask app
    logger.info(f"Starting proxy server on {host}:{port}")
    logger.info(f"Distributed inference: {'ENABLED' if use_distributed_inference else 'DISABLED'}")
    try:
        app.run(host=host, port=port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down proxy server...")
    finally:
        # Stop the UI if it's running
        if ui_active:
            ui_exit_event.set()
            if ui_thread and ui_thread.is_alive():
                ui_thread.join(timeout=2)
            ui_active = False
            
        # Stop the discovery service if it's running
        if discovery_service:
            discovery_service.stop()