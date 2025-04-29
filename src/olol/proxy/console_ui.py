"""Console UI for the Ollama proxy server."""

import curses
import logging
import threading
import time
from typing import Dict, Any, Optional

from .stats import stats_lock, request_stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UI state
ui_exit_event = threading.Event()

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
        
        # Import needed only here to avoid circular imports
        from proxy import cluster, use_distributed_inference
        
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


def run_console_ui(params=None):
    """Run the console UI in a separate thread.
    
    Args:
        params: Dictionary of parameters controlling UI behavior
    """
    ui = ConsoleUI(params)
    try:
        # Import needed only here to avoid circular imports
        from proxy import cluster
        
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