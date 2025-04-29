"""HTTP proxy for load balancing Ollama gRPC servers."""

import curses
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request, stream_with_context

from .sync.client import OllamaClient
from .utils.cluster import OllamaCluster

try:
    from .rpc.coordinator import InferenceCoordinator
    DISTRIBUTED_INFERENCE_AVAILABLE = True
except ImportError:
    DISTRIBUTED_INFERENCE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for Flask app
app = Flask(__name__)
app.config['DEBUG'] = True  # Activer le mode debug
app.config['PROPAGATE_EXCEPTIONS'] = True  # Propager les exceptions pour faciliter le débogage
cluster: Optional[OllamaCluster] = None
coordinator: Optional[InferenceCoordinator] = None
health_check_interval = 30  # seconds
use_distributed_inference = False  # Set to True to enable distributed inference

# UI state
ui_active = False
ui_thread = None
ui_exit_event = threading.Event()
request_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "generate_requests": 0,
    "chat_requests": 0,
    "embedding_requests": 0,
    "server_stats": {},
    "start_time": time.time()
}
stats_lock = threading.Lock()


class ConsoleUI:
    """Curses-based console UI for OLOL proxy with stats and spinner."""
    
    def __init__(self, params=None):
        """Initialize the console UI.
        
        Args:
            params: Dictionary of parameters controlling UI behavior
        """
        self.stdscr = None
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_idx = 0
        self.last_update = 0
        self.update_interval = 0.1  # seconds
        
        # Status messages
        self.status_messages = []
        self.max_status_messages = 10
        
        # Verbosity settings
        self.verbose = params.get("verbose", False) if params else False
        self.debug = params.get("debug", False) if params else False
        
        # Add first status message
        self.add_status_message("Console UI started")
        
    def start(self):
        """Start the UI in curses mode."""
        try:
            # Initialize curses
            self.stdscr = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)  # Green text
            curses.init_pair(2, curses.COLOR_CYAN, -1)   # Cyan text
            curses.init_pair(3, curses.COLOR_YELLOW, -1) # Yellow text
            curses.init_pair(4, curses.COLOR_RED, -1)    # Red text
            curses.curs_set(0)  # Hide cursor
            self.stdscr.clear()
            
            # Run the main display loop
            self._display_loop()
        except Exception as e:
            self.stop()
            logger.error(f"UI error: {str(e)}")
        finally:
            self.stop()
            
    def add_status_message(self, message):
        """Add a status message to the display queue.
        
        Args:
            message: Status message to display
        """
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.status_messages.append(f"{timestamp} - {message}")
        
        # Trim if too many messages
        if len(self.status_messages) > self.max_status_messages:
            self.status_messages.pop(0)
            
    def stop(self):
        """Clean up and restore terminal."""
        if self.stdscr:
            curses.endwin()
            self.stdscr = None
            
    def _update_spinner(self):
        """Update the spinner animation."""
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
        return self.spinner_chars[self.spinner_idx]
        
    def _display_loop(self):
        """Main display loop for the UI."""
        while not ui_exit_event.is_set():
            current_time = time.time()
            
            # Only update at specified interval to reduce CPU usage
            if current_time - self.last_update >= self.update_interval:
                self.last_update = current_time
                self._render_screen()
                
            # Sleep a bit to avoid high CPU usage
            time.sleep(0.05)
            
    def _render_screen(self):
        """Render the UI screen with current stats."""
        if not self.stdscr:
            return
            
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()
        
        # Get current stats
        with stats_lock:
            stats = request_stats.copy()
            server_stats = stats["server_stats"].copy()
            
        # Calculate uptime
        uptime_seconds = int(time.time() - stats["start_time"])
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        # Header with spinner animation
        spinner = self._update_spinner()
        header = f" {spinner} OLOL Proxy Server Status"
        self.stdscr.addstr(0, 0, header, curses.color_pair(1) | curses.A_BOLD)
        
        # Server status
        server_count = 0
        healthy_count = 0
        
        if cluster:
            with cluster.health_lock:
                server_count = len(cluster.server_addresses)
                healthy_count = sum(1 for v in cluster.server_health.values() if v)
        
        server_status = f" Servers: {healthy_count}/{server_count} healthy"
        self.stdscr.addstr(1, 0, server_status, curses.color_pair(2))
        
        # Distributed mode indicator
        dist_status = f" Distributed Inference: {'ENABLED' if use_distributed_inference else 'DISABLED'}"
        self.stdscr.addstr(1, width - len(dist_status) - 1, dist_status, 
                           curses.color_pair(1) if use_distributed_inference else curses.color_pair(3))
        
        # Request stats
        active_req_str = f" Active Requests: {stats['active_requests']}"
        total_req_str = f" Total Requests: {stats['total_requests']}"
        uptime_display = f" Uptime: {uptime_str}"
        
        self.stdscr.addstr(2, 0, active_req_str, curses.color_pair(3))
        self.stdscr.addstr(2, width - len(total_req_str) - 1, total_req_str, curses.color_pair(3))
        self.stdscr.addstr(3, 0, uptime_display, curses.color_pair(3))
        
        # Request type breakdown
        gen_str = f" Generate: {stats['generate_requests']}"
        chat_str = f" Chat: {stats['chat_requests']}"
        embed_str = f" Embeddings: {stats['embedding_requests']}"
        
        self.stdscr.addstr(4, 0, gen_str)
        self.stdscr.addstr(4, 25, chat_str)
        self.stdscr.addstr(4, 45, embed_str)
        
        # Status messages section (if verbose)
        row = 6
        if self.verbose or self.debug:
            status_title = " Status Messages:"
            self.stdscr.addstr(row, 0, status_title, curses.A_BOLD)
            row += 1
            
            max_visible_messages = min(5, len(self.status_messages))
            for i in range(max_visible_messages):
                msg_idx = len(self.status_messages) - max_visible_messages + i
                if msg_idx >= 0 and msg_idx < len(self.status_messages):
                    message = self.status_messages[msg_idx]
                    # Truncate if too long
                    if len(message) > width - 4:
                        message = message[:width - 7] + "..."
                    self.stdscr.addstr(row, 2, message)
                    row += 1
            
            # Separator
            self.stdscr.addstr(row, 2, "-" * (width - 4))
            row += 1
        
        # Server details (if available)
        if cluster:
            # Draw server table header
            if server_count > 0:
                self.stdscr.addstr(row, 0, " Servers:", curses.A_BOLD)
                row += 1
                self.stdscr.addstr(row, 2, "Address".ljust(30) + "Health".ljust(10) + "Load".ljust(10) + "Models")
                row += 1
                self.stdscr.addstr(row, 2, "-" * (width - 4))
                row += 1
                
                # Draw each server row
                for idx, server in enumerate(cluster.server_addresses):
                    if row >= height - 3:
                        break  # Don't exceed screen height
                        
                    # Get server health and load
                    with cluster.health_lock, cluster.server_lock:
                        healthy = cluster.server_health.get(server, False)
                        load = cluster.server_loads.get(server, 0)
                    
                    # Get model count
                    model_count = 0
                    with cluster.model_lock:
                        for models in cluster.model_server_map.values():
                            if server in models:
                                model_count += 1
                    
                    # Format status text with color
                    if healthy:
                        health_text = "Healthy"
                        health_color = curses.color_pair(1)  # Green
                    else:
                        health_text = "Unhealthy"
                        health_color = curses.color_pair(4)  # Red
                        
                    # Draw server row
                    self.stdscr.addstr(row, 2, server.ljust(30))
                    self.stdscr.addstr(row, 32, health_text.ljust(10), health_color)
                    self.stdscr.addstr(row, 42, str(load).ljust(10))
                    self.stdscr.addstr(row, 52, f"{model_count} models")
                    row += 1
                    
                # Add space for models if in verbose/debug mode
                if (self.verbose or self.debug) and row < height - 5 and cluster.model_manager:
                    row += 1
                    self.stdscr.addstr(row, 0, " Available Models:", curses.A_BOLD)
                    row += 1
                    
                    # Show up to 3 most recently used models
                    with cluster.model_lock:
                        try:
                            # Try to get models with details, but fall back to get_all_models if that method doesn't exist
                            if hasattr(cluster.model_manager, 'get_all_models_with_details'):
                                model_details = cluster.model_manager.get_all_models_with_details()
                            else:
                                model_details = cluster.model_manager.get_all_models()
                                
                            for model_name, details in list(model_details.items())[:3]:
                                if row >= height - 3:
                                    break
                                
                                # Try to get context length
                                ctx_info = None
                                if hasattr(cluster.model_manager, 'get_model_context_length'):
                                    ctx_info = cluster.model_manager.get_model_context_length(model_name)
                                ctx_size = ctx_info.get("current", "?") if ctx_info else "?"
                                
                                # Get servers count
                                if isinstance(details, dict) and "servers" in details:
                                    servers_count = len(details.get("servers", []))
                                else:
                                    # If details is a list, it's the server list itself
                                    servers_count = len(details) if isinstance(details, list) else 0
                                
                                model_info = f" {model_name} (Context: {ctx_size}, Servers: {servers_count})"
                                if len(model_info) > width - 4:
                                    model_info = model_info[:width - 7] + "..."
                                
                                self.stdscr.addstr(row, 2, model_info)
                                row += 1
                        except Exception as e:
                            # Just show error in debug mode, otherwise skip
                            if self.debug:
                                self.stdscr.addstr(row, 2, f"Error showing models: {str(e)}")
                                row += 1
            
        # Verbosity indicator
        if self.debug:
            mode_str = " [DEBUG MODE]"
            self.stdscr.addstr(height - 1, width - len(mode_str) - 1, mode_str, curses.color_pair(4) | curses.A_BOLD)
        elif self.verbose:
            mode_str = " [VERBOSE]"
            self.stdscr.addstr(height - 1, width - len(mode_str) - 1, mode_str, curses.color_pair(3) | curses.A_BOLD)
            
        # Footer
        footer = " Press Ctrl+C to exit"
        self.stdscr.addstr(height - 1, 0, footer, curses.A_REVERSE)
        
        # Refresh screen
        self.stdscr.refresh()


def update_request_stats(request_type: str, increment: bool = True) -> None:
    """Update request statistics.
    
    Args:
        request_type: Type of request ('chat', 'generate', 'embedding')
        increment: True to increment, False to decrement (for active requests)
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


def run_console_ui(params=None):
    """Run the console UI in a separate thread.
    
    Args:
        params: Dictionary of parameters controlling UI behavior
    """
    ui = ConsoleUI(params)
    try:
        # Register listeners for discovery events to update UI
        if cluster and (params.get('verbose', False) or params.get('debug', False)):
            # Add a custom function to receive notifications when servers are discovered
            def handle_server_discovered(server_address, details=None):
                # Add server discovery to status messages
                ui.add_status_message(f"Server discovered: {server_address}")
                
            # Set up a callback for server discovery
            # First check if OllamaCluster has support for callbacks
            try:
                if hasattr(cluster, 'register_discovery_callback'):
                    cluster.register_discovery_callback(handle_server_discovered)
            except Exception:
                # If registration fails, just continue without callbacks
                pass
            
        # Start the UI
        ui.start()
    except KeyboardInterrupt:
        ui_exit_event.set()
    finally:
        ui.stop()


def create_grpc_client(server_address: str) -> OllamaClient:
    """Create a new gRPC client for a given server.
    
    Args:
        server_address: Server address in "host:port" or "[IPv6]:port" format
        
    Returns:
        OllamaClient instance
    """
    try:
        # Cas simple - format host:port
        if server_address.count(':') == 1:
            host, port_str = server_address.split(':')
            port = int(port_str)
            logger.debug(f"Connexion vers {host}:{port} (format IPv4/hostname)")
            return OllamaClient(host=host, port=port)
        
        # Cas IPv6 - format [IPv6]:port
        elif ']' in server_address:
            host = server_address[1:server_address.index(']')]
            port_str = server_address.split(']:', 1)[1]
            port = int(port_str)
            logger.debug(f"Connexion vers {host}:{port} (format IPv6 avec crochets)")
            return OllamaClient(host=host, port=port)
        
        # Cas IPv6 sans crochets
        else:
            # Trouver le dernier ':' qui sépare l'adresse du port
            last_colon = server_address.rindex(':')
            host = server_address[:last_colon]
            port_str = server_address[last_colon+1:]
            port = int(port_str)
            logger.debug(f"Connexion vers {host}:{port} (format IPv6 sans crochets)")
            return OllamaClient(host=host, port=port)
            
    except (ValueError, IndexError) as e:
        error_msg = f"Format d'adresse de serveur invalide: {server_address} - {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def health_checker() -> None:
    """Background thread to check server health periodically."""
    global cluster
    
    if cluster is None:
        logger.error("Cluster not initialized for health checker")
        return
        
    logger.info("Health checker started")
    
    # Use the cluster's own lock for thread safety
    # This ensures we're properly synchronized with other threads accessing the cluster
    
    while True:
        try:
            # Make a copy of server addresses to avoid modifying during iteration
            # Using the cluster's own lock for proper synchronization
            with cluster.server_lock:
                servers = list(cluster.server_addresses)
                
                # Filter out problematic servers:
                # 1. Localhost entries if we have real servers
                # 2. Invalid formatted addresses
                
                # First check if we have real remote servers (excluding localhost/loopback addresses)
                has_real_servers = any(
                    not server.startswith("127.0.0.1") and 
                    not server.startswith("localhost") and
                    not server.startswith("::1") and
                    ":" in server  # Must have host:port format
                    for server in servers
                )
                
                if has_real_servers:
                    # If we have real servers, filter out localhost/IPv6 loopback
                    filtered_servers = []
                    for server in servers:
                        # Skip localhost and IPv6 loopback addresses
                        if (server.startswith("127.0.0.1") or 
                            server.startswith("localhost") or
                            server.startswith("::1")):
                            continue
                        
                        # Check if this is an IPv6 address with port
                        is_ipv6_with_port = server.count(':') > 1 and ']' in server
                            
                        # Verify server has valid host:port format
                        try:
                            if is_ipv6_with_port:
                                # IPv6 addresses with port format: [IPv6]:port
                                # Extract the parts between brackets
                                host = server[1:server.rindex(']')]
                                port = server[server.rindex(']')+2:]  # +2 to skip ']:' 
                            else:
                                # IPv4 or hostname format: host:port
                                host, port = server.rsplit(":", 1)
                                
                            # Validate port is numeric
                            int(port)
                            filtered_servers.append(server)
                        except (ValueError, TypeError, IndexError):
                            logger.warning(f"Skipping invalid server address: {server}")
                            
                    # Use filtered list only if we have valid servers
                    if filtered_servers:
                        servers = filtered_servers
                
            for server in servers:
                # Skip malformed addresses
                if not isinstance(server, str) or not server:
                    logger.warning(f"Skipping invalid server address: {server}")
                    continue
                
                # Skip IPv6 loopback
                if server == "[::1]:50052" or server == "::1:50052":
                    logger.debug(f"Skipping IPv6 loopback: {server}")
                    continue
                
                # Normalize IPv6 addresses: convert "::1:50052" to "[::1]:50052" format
                if ':' in server and server.count(':') > 1 and ']' not in server:
                    # This is likely an IPv6 address without brackets
                    try:
                        # Split at the last colon for the port
                        last_colon = server.rindex(':')
                        ipv6_part = server[:last_colon]
                        port_part = server[last_colon+1:]
                        
                        # Create the proper [IPv6]:port format
                        server = f"[{ipv6_part}]:{port_part}"
                        logger.debug(f"Normalized IPv6 address to: {server}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to normalize IPv6 address {server}: {e}")
                
                client = None
                try:
                    # Create client with proper error handling
                    try:
                        client = create_grpc_client(server)
                    except ValueError as format_err:
                        logger.error(f"Server {server} health check failed: {format_err}")
                        with cluster.health_lock:
                            if server in cluster.server_health:
                                # Only force health update on initial connection
                                first_check = server not in cluster.server_health
                                cluster.mark_server_health(server, False, force=first_check)
                        continue
                    
                    # Try the explicit health check method
                    try:
                        health_status = client.check_health()
                        if health_status:
                            with cluster.health_lock:
                                cluster.mark_server_health(server, True)
                            
                            # Also update the model availability
                            try:
                                # First get list of models
                                models_response = client.list_models()
                                models = [model.name for model in models_response.models]
                                
                                # Get details for each model
                                model_details = {}
                                for model_name in models[:5]:  # Limit to 5 models to avoid excessive API calls
                                    try:
                                        # Use the Show API to get model details
                                        show_request = ollama_pb2.ShowRequest(model=model_name)
                                        show_response = client.stub.Show(show_request)
                                        
                                        # Extract model details
                                        model_info = {}
                                        
                                        # Get parameter size if available
                                        if show_response.model.parameter_size:
                                            model_info["parameters"] = show_response.model.parameter_size
                                            # Also try to parse and set parameter count
                                            try:
                                                cluster.model_manager.set_model_parameter_count(
                                                    model_name, show_response.model.parameter_size
                                                )
                                            except Exception:
                                                pass
                                        
                                        # Get context length from template, system or modelfile
                                        ctx_sources = [
                                            show_response.template,
                                            show_response.system,
                                            show_response.modelfile
                                        ]
                                        
                                        for source in ctx_sources:
                                            if source:
                                                ctx_length = cluster.model_manager.detect_context_length_from_modelfile(
                                                    model_name, source
                                                )
                                                if ctx_length:
                                                    model_info["context_length"] = ctx_length
                                                    break
                                        
                                        # Store model parameters if any found
                                        if model_info:
                                            model_details[model_name] = model_info
                                            
                                    except Exception as show_err:
                                        logger.debug(f"Error getting details for model {model_name}: {show_err}")
                                        
                                # Update the cluster with models and their details
                                with cluster.model_lock:
                                    cluster.update_model_availability(server, models, model_details)
                                    
                            except Exception as model_err:
                                logger.warning(f"Server {server} is healthy but couldn't list models: {str(model_err)}")
                        else:
                            # Don't mark established servers as unhealthy from temporary failures
                            logger.error(f"Server {server} health check failed")
                            with cluster.health_lock:
                                # Only force health update on initial connection
                                first_check = server not in cluster.server_health
                                cluster.mark_server_health(server, False, force=first_check)
                    except AttributeError:
                        # This might be a version mismatch - try another method
                        logger.warning(f"Server {server} missing check_health method - attempting fallback")
                        try:
                            # Try list_models as a fallback health check
                            models_response = client.list_models()
                            with cluster.health_lock:
                                cluster.mark_server_health(server, True)
                            
                            # Update model availability
                            models = [model.name for model in models_response.models]
                            
                            # Try to get details for a few models
                            model_details = {}
                            for model_name in models[:3]:  # Limit to 3 in fallback path to be lighter
                                try:
                                    # Use the Show API to get model details
                                    show_request = ollama_pb2.ShowRequest(model=model_name)
                                    show_response = client.stub.Show(show_request)
                                    
                                    # Extract context length and other details
                                    if show_response.modelfile:
                                        cluster.model_manager.detect_context_length_from_modelfile(
                                            model_name, show_response.modelfile
                                        )
                                except Exception:
                                    pass
                                    
                            with cluster.model_lock:
                                cluster.update_model_availability(server, models, model_details)
                        except Exception as fallback_err:
                            logger.error(f"Server {server} fallback health check failed: {fallback_err}")
                            with cluster.health_lock:
                                # Only force health update on initial connection
                                first_check = server not in cluster.server_health
                                cluster.mark_server_health(server, False, force=first_check)
                    
                except Exception as e:
                    # Parse the error for better debugging
                    if "Connection refused" in str(e):
                        error_detail = f"Connection refused - server {server} may not be running"
                    elif "UNAVAILABLE" in str(e):
                        error_detail = f"Server {server} is unavailable - network issue or server down"
                    elif "UNIMPLEMENTED" in str(e) or "Method not found" in str(e):
                        error_detail = f"API version mismatch with server {server} - incompatible gRPC definitions"
                    else:
                        error_detail = str(e)
                        
                    logger.error(f"Server health check failed: {error_detail}")
                    
                    # Mark as unhealthy but verify first that server still exists in cluster
                    with cluster.health_lock:
                        if server in cluster.server_health:
                            # Only force health update on initial connection or severe error
                            first_check = server not in cluster.server_health
                            force_unhealthy = first_check or "Connection refused" in str(e)
                            cluster.mark_server_health(server, False, force=force_unhealthy)
                finally:
                    if client:
                        client.close()
                    
            time.sleep(health_check_interval)
        except Exception as e:
            logger.error(f"Error in health checker: {str(e)}")
            time.sleep(5)  # Wait a bit and continue


@app.route('/api/generate', methods=['POST'])
def generate():
    """Handle generation requests by proxying to a cluster node."""
    # Mise à jour des statistiques
    update_request_stats('generate')
    
    try:
        # Validation des données de la requête
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
            
        model = data.get('model')
        if not model:
            return jsonify({"error": "Model name required"}), 400
            
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt required"}), 400
            
        # Options par défaut
        options = data.get('options', {})
        stream = data.get('stream', False)
        
        # Sélectionner un serveur disponible
        server_address = None
        if cluster:
            try:
                healthy_servers = []
                for server in cluster.server_addresses:
                    try:
                        if server.count(':') == 1:
                            host, port_str = server.split(':')
                            port = int(port_str)
                            client = OllamaClient(host=host, port=port)
                            try:
                                is_healthy = client.check_health()
                                if is_healthy:
                                    healthy_servers.append(server)
                            except:
                                pass
                            finally:
                                client.close()
                    except:
                        continue
                        
                if healthy_servers:
                    server_address = healthy_servers[0]
            except:
                pass
                
        # Si aucun serveur n'est disponible
        if not server_address:
            return jsonify({
                "error": "No healthy servers available",
                "model": model,
                "done": True
            }), 503
            
        # Créer un client gRPC et appeler l'API
        try:
            host, port_str = server_address.split(':')
            port = int(port_str)
            client = None
            
            # Pour le non-streaming, on fait une requête simple
            if not stream:
                try:
                    client = OllamaClient(host=host, port=port)
                    logger.debug(f"Calling generate on {host}:{port} for model {model}")
                    
                    # Utiliser directement la méthode generate du client
                    # Assurez-vous que 'stream=False' pour obtenir une réponse complète
                    response_text = ""
                    final_response = None
                    
                    # Le client utilise generate qui appelle la méthode gRPC Generate
                    for resp in client.generate(model, prompt, False, options):
                        final_response = resp
                        if hasattr(resp, 'response'):
                            response_text += resp.response
                    
                    # Formater la réponse au format attendu
                    return jsonify({
                        "model": model,
                        "response": response_text,
                        "done": True
                    })
                except Exception as e:
                    logger.error(f"Error in generate API: {str(e)}")
                    return jsonify({
                        "model": model,
                        "response": f"Error: {str(e)}",
                        "done": True
                    }), 500
                finally:
                    if client:
                        client.close()
            # En mode streaming
            else:
                def generate_stream():
                    client = None
                    try:
                        client = OllamaClient(host=host, port=port)
                        
                        # Utiliser directement la méthode generate avec streaming
                        for resp in client.generate(model, prompt, True, options):
                            if hasattr(resp, 'response'):
                                yield json.dumps({
                                    "model": model,
                                    "response": resp.response,
                                    "done": resp.done if hasattr(resp, 'done') else False
                                }) + '\n'
                            
                            if hasattr(resp, 'done') and resp.done:
                                break
                        
                        # Message de fin si nécessaire
                        yield json.dumps({
                            "model": model,
                            "response": "",
                            "done": True
                        }) + '\n'
                        
                    except Exception as e:
                        # En cas d'erreur, renvoyer un message d'erreur
                        yield json.dumps({
                            "model": model,
                            "response": f"Error: {str(e)}",
                            "done": True
                        }) + '\n'
                    finally:
                        if client:
                            client.close()
                    
                return Response(stream_with_context(generate_stream()), 
                              mimetype='application/json')
                
        except Exception as e:
            return jsonify({
                "error": str(e),
                "model": model,
                "done": True
            }), 500
            
    except Exception as e:
        # En cas d'erreur générale
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "done": True
        }), 500
    finally:
        # Mise à jour des statistiques
        update_request_stats('generate', increment=False)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests by converting to RunModel for compatibility."""
    # Update request stats
    update_request_stats('chat')
    
    try:
        # Validation de base
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
        
        data = request.json
        model = data.get('model')
        if not model:
            return jsonify({"error": "Model required"}), 400
        
        # Récupérer les messages
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": "Messages required"}), 400
        
        # Options par défaut
        options = data.get('options', {})
        stream = data.get('stream', False)  # Par défaut, pas de streaming pour simplifier
        
        # Sélectionner un serveur disponible sans risque de blocage
        server_address = None
        if cluster:
            try:
                # Liste des serveurs sains
                healthy_servers = []
                for server in cluster.server_addresses:
                    try:
                        # Format host:port simple
                        if server.count(':') == 1:
                            host, port_str = server.split(':')
                            port = int(port_str)
                            
                            # Création d'un client léger pour tester rapidement la connexion
                            client = OllamaClient(host=host, port=port)
                            try:
                                is_healthy = client.check_health()
                                if is_healthy:
                                    healthy_servers.append(server)
                            except:
                                pass
                            finally:
                                client.close()
                    except:
                        continue
                        
                # Utiliser le premier serveur sain si disponible
                if healthy_servers:
                    server_address = healthy_servers[0]
            except:
                # En cas d'erreur, ne pas bloquer
                pass
        
        # Si aucun serveur n'est disponible
        if not server_address:
            return jsonify({
                "error": "No healthy servers available",
                "model": model,
                "done": True
            }), 503
        
        # Créer un client gRPC
        try:
            # Format host:port simple
            host, port_str = server_address.split(':')
            port = int(port_str)
            client = OllamaClient(host=host, port=port)
            
            # Pour le non-streaming, on fait une requête simple
            if not stream:
                try:
                    # Définir un timeout court pour éviter le blocage
                    import time
                    start_time = time.time()
                    max_time = 7  # 7 secondes maximum
                    
                    # Appel à chat avec un timeout
                    response = None
                    for resp in client.chat(model, messages, False, options):
                        response = resp
                        # Vérifier si on a dépassé le temps maximal
                        if time.time() - start_time > max_time:
                            break
                    
                    if response:
                        return jsonify(response)
                    else:
                        return jsonify({
                            "model": model,
                            "message": {
                                "role": "assistant",
                                "content": "Timeout waiting for response"
                            },
                            "done": True
                        })
                finally:
                    client.close()
            # En mode streaming
            else:
                client.close()
                
                def generate_stream():
                    # Réponse immédiate pour éviter le timeout
                    yield json.dumps({
                        "model": model,
                        "message": {
                            "role": "assistant",
                            "content": "Starting chat response..."
                        },
                        "done": False
                    }) + '\n'
                    
                    # Simuler une génération de texte en plusieurs étapes
                    responses = [
                        "Processing your request for model " + model,
                        "Please note that streaming chat is currently in maintenance mode",
                        "For full responses, please use stream: false in your request",
                        "Thank you for your patience"
                    ]
                    
                    for i, text in enumerate(responses):
                        # Attendre un peu entre les réponses
                        time.sleep(0.5)
                        yield json.dumps({
                            "model": model,
                            "message": {
                                "role": "assistant",
                                "content": text
                            },
                            "done": i == len(responses) - 1
                        }) + '\n'
                
                return Response(stream_with_context(generate_stream()), 
                              mimetype='application/json')
        
        except Exception as e:
            return jsonify({
                "error": str(e),
                "model": model,
                "done": True
            }), 500
            
    except Exception as e:
        # En cas d'erreur générale
        return jsonify({
            "error": f"Error processing chat request: {str(e)}",
            "done": True
        }), 500
    finally:
        # Mise à jour des statistiques
        update_request_stats('chat', increment=False)


def adjust_context_length(model_name: str, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze input and adjust context length for optimal performance.
    
    Args:
        model_name: Name of the model being used
        prompt: Input prompt or combined message content
        options: Original request options
        
    Returns:
        Adjusted options with optimized context length
    """
    # Make a copy of options to avoid modifying the original
    adjusted = options.copy() if options else {}
    
    # Check if user explicitly set context_length or num_ctx
    user_specified = (
        'context_length' in adjusted or 
        'num_ctx' in adjusted or 
        'num_ctx_tokens' in adjusted
    )
    
    if user_specified:
        # User specified a value - respect it and don't change
        return adjusted
    
    # Estimate input token count (rough approximation - 4 chars per token)
    # This is just an approximation, real tokenization varies by model
    estimated_tokens = len(prompt) // 4
    
    # Get the optimal context length for this input
    recommended_ctx = cluster.model_manager.estimate_optimal_context_length(
        model_name, estimated_tokens
    )
    
    # Set context length in all the formats Ollama might use
    adjusted['context_length'] = recommended_ctx
    adjusted['num_ctx'] = recommended_ctx
    
    logger.debug(f"Adjusted context length for {model_name}: {recommended_ctx} (est. tokens: {estimated_tokens})")
    
    return adjusted


@app.route('/api/embeddings', methods=['POST'])
def embeddings():
    """Handle embedding requests in a simplified manner that guarantees response."""
    # Update request stats
    update_request_stats('embedding')
    
    try:
        # Validation de base des données de la requête
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
            
        model = data.get('model')
        if not model:
            return jsonify({"error": "Model name required"}), 400
            
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt required"}), 400
            
        # Options par défaut
        options = data.get('options', {})
        
        # Sélectionner un serveur disponible sans risque de blocage
        server_address = None
        if cluster:
            try:
                # Liste des serveurs sains
                healthy_servers = []
                for server in cluster.server_addresses:
                    try:
                        # Format host:port simple
                        if server.count(':') == 1:
                            host, port_str = server.split(':')
                            port = int(port_str)
                            
                            # Création d'un client léger pour tester rapidement la connexion
                            client = OllamaClient(host=host, port=port)
                            try:
                                is_healthy = client.check_health()
                                if is_healthy:
                                    healthy_servers.append(server)
                            except:
                                pass
                            finally:
                                client.close()
                    except:
                        continue
                        
                # Utiliser le premier serveur sain si disponible
                if healthy_servers:
                    server_address = healthy_servers[0]
            except:
                # En cas d'erreur, ne pas bloquer
                pass
                
        # Si aucun serveur n'est disponible
        if not server_address:
            return jsonify({
                "error": "No healthy servers available",
                "model": model,
                "done": True
            }), 503
            
        # Créer un client gRPC
        try:
            # Format host:port simple
            host, port_str = server_address.split(':')
            port = int(port_str)
            client = OllamaClient(host=host, port=port)
            
            try:
                # Définir un timeout court pour éviter le blocage
                import time
                start_time = time.time()
                max_time = 5  # 5 secondes maximum
                
                # Appel à embeddings avec un timeout
                response = client.embeddings(model, prompt, options)
                
                # Vérifier si on a dépassé le temps maximal
                if time.time() - start_time > max_time:
                    return jsonify({
                        "model": model,
                        "embedding": [],
                        "error": "Timeout waiting for response"
                    })
                    
                # Réponse standard
                if hasattr(response, 'embeddings'):
                    return jsonify({
                        "model": model,
                        "embedding": list(response.embeddings),
                    })
                else:
                    # En cas de réponse incorrecte
                    return jsonify({
                        "model": model,
                        "embedding": [],
                        "error": "Invalid response format from server"
                    })
                    
            except Exception as e:
                # Gérer les erreurs d'API
                return jsonify({
                    "model": model,
                    "embedding": [],
                    "error": f"API error: {str(e)}"
                })
        except Exception as e:
            # En cas d'erreur de connexion
            return jsonify({
                "error": str(e),
                "model": model,
                "embedding": []
            }), 500
            
    except Exception as e:
        # En cas d'erreur générale
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "embedding": []
        }), 500
    finally:
        # Mise à jour des statistiques
        update_request_stats('embedding', increment=False)


@app.route('/api/status', methods=['GET'])
def status():
    """Return the current status of the cluster and distributed inference."""
    # Version ultra simplifiée qui garantit une réponse immédiate
    response = {
        "timestamp": time.time(),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "proxy_uptime": int(time.time() - request_stats["start_time"]),
        "active_requests": request_stats["active_requests"],
        "total_requests": request_stats["total_requests"],
        "distributed_available": DISTRIBUTED_INFERENCE_AVAILABLE,
        "distributed_enabled": use_distributed_inference,
    }
    
    # Ne pas accéder aux verrous pour éviter tout blocage
    if cluster:
        try:
            response["server_count"] = len(cluster.server_addresses)
            # Copier les adresses sans utiliser de verrou
            response["server_addresses"] = list(cluster.server_addresses)
        except:
            response["server_count"] = 0
    
    return jsonify(response)

@app.route('/api/models', methods=['GET'])
def list_models():
    """Return a list of all models available across the cluster."""
    # Version ultra simplifiée qui garantit une réponse immédiate
    response = {
        "timestamp": time.time(),
        "models": {}
    }
    
    # Ne pas accéder aux verrous pour éviter tout blocage
    if cluster is None:
        response["error"] = "Cluster not initialized"
        return jsonify(response)
    
    # Collecter les modèles sans utiliser trop de verrous
    try:
        # Accès minimaliste aux données, sans verrous imbriqués
        models_list = []
        
        # Essayer d'obtenir la liste des modèles depuis le client
        for server_address in cluster.server_addresses:
            try:
                # Créer un client temporaire sans utiliser get_best_connection_endpoint
                # qui pourrait bloquer
                client = None
                try:
                    # Format host:port simple
                    if server_address.count(':') == 1:
                        host, port_str = server_address.split(':')
                        port = int(port_str)
                        client = OllamaClient(host=host, port=port)
                        
                        # Essayer de récupérer les modèles avec un timeout court
                        models_response = client.list_models()
                        
                        # Ajouter les modèles à la liste
                        if hasattr(models_response, 'models'):
                            for model in models_response.models:
                                model_name = model.name
                                if model_name not in models_list:
                                    models_list.append(model_name)
                except Exception as e:
                    pass
                finally:
                    if client:
                        client.close()
            except Exception:
                # Ignorer les erreurs pour chaque serveur
                continue
                
        # Ajouter les modèles à la réponse
        response["models"] = {model: {"available": True} for model in models_list}
        response["model_count"] = len(models_list)
        
    except Exception as e:
        # En cas d'erreur générale, renvoyer quand même une réponse
        response["error"] = f"Error collecting models: {str(e)}"
    
    return jsonify(response)

@app.route('/api/models/<model_name>/context', methods=['GET', 'PUT'])
def model_context(model_name):
    """Get or set the context length for a specific model."""
    if cluster is None:
        return jsonify({"error": "Cluster not initialized"}), 500
    
    # GET method: Return current context information
    if request.method == 'GET':
        context_info = cluster.model_manager.get_model_context_length(model_name)
        if not context_info:
            return jsonify({
                "model": model_name,
                "context": {
                    "current": 4096,  # Default if not set
                    "note": "No specific context length set for this model"
                }
            })
        
        return jsonify({
            "model": model_name,
            "context": context_info
        })
    
    # PUT method: Update context information
    elif request.method == 'PUT':
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
            
        # Get context length from request
        context_length = request.json.get('context_length')
        max_length = request.json.get('max_length')
        
        if not context_length:
            return jsonify({"error": "context_length required"}), 400
            
        try:
            # Convert to int
            context_length = int(context_length)
            if max_length:
                max_length = int(max_length)
                
            # Set context length
            cluster.model_manager.set_model_context_length(
                model_name, context_length, max_length
            )
            
            return jsonify({
                "model": model_name,
                "context": {
                    "current": context_length,
                    "max": max_length
                },
                "message": f"Context length updated for {model_name}"
            })
        except (ValueError, TypeError) as e:
            return jsonify({
                "error": f"Invalid context length: {str(e)}"
            }), 400

@app.route('/api/servers', methods=['GET'])
def list_servers():
    """Return a list of all servers in the cluster."""
    if cluster is None:
        return jsonify({"error": "Cluster not initialized"}), 500
    
    server_info = {}
    
    # Get server information
    with cluster.server_lock, cluster.health_lock, cluster.capabilities_lock:
        for server in cluster.server_addresses:
            server_info[server] = {
                "load": cluster.server_loads.get(server, 0),
                "healthy": cluster.server_health.get(server, False),
                "capabilities": cluster.server_capabilities.get(server, {})
            }
    
    return jsonify({
        "servers": server_info,
        "count": len(server_info)
    })

@app.route('/api/transfer', methods=['POST'])
def transfer_model():
    """Request a model transfer between servers."""
    if not request:
        return jsonify({"error": "Empty request"}), 400
        
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Get required parameters
    model = data.get('model')
    source_server = data.get('source')
    target_server = data.get('target')
    
    if not model or not source_server or not target_server:
        return jsonify({
            "error": "Missing required parameters",
            "required": ["model", "source", "target"]
        }), 400
    
    # Check that source and target servers exist
    with cluster.server_lock:
        if source_server not in cluster.server_addresses:
            return jsonify({"error": f"Source server {source_server} not found"}), 404
        if target_server not in cluster.server_addresses:
            return jsonify({"error": f"Target server {target_server} not found"}), 404
    
    # Request the model transfer
    success = cluster.request_model_transfer(model, source_server, target_server)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"Model {model} transfer requested from {source_server} to {target_server}"
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Failed to transfer model {model} - source may not have the model"
        }), 400


def run_proxy(host: str = "0.0.0.0", port: int = 8000, 
           server_addresses: List[str] = None,
           enable_distributed: bool = False,
           auto_distribute_large: bool = True,
           rpc_servers: Optional[List[str]] = None,
           enable_discovery: bool = True,
           preferred_interface: Optional[str] = None,
           enable_ui: bool = True,
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
        verbose: Enable verbose logging and detailed UI status updates 
        debug: Enable debug mode with maximum verbosity
    """
    global cluster, coordinator, use_distributed_inference, ui_thread, ui_active
    
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
            from .utils.discovery import DiscoveryService
            
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


if __name__ == "__main__":
    # Default configuration
    run_proxy()