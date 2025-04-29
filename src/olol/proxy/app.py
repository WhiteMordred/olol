import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from flask import Flask, Response, jsonify, request, render_template, stream_with_context, url_for

from olol.sync.client import OllamaClient
from olol.utils.cluster import OllamaCluster

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

try:
    from olol.rpc.coordinator import InferenceCoordinator
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
    for i, server in enumerate(servers[:5]):  # Limit to 5 servers for clarity
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
    for i, server in enumerate(servers[:5]):  # Limit to 5 servers for clarity
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
            from olol.utils.discovery import DiscoveryService
            
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