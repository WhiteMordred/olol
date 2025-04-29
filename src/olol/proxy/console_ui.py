"""Console UI for the Ollama proxy server using Rich."""

import threading
import time
import logging
from typing import Dict, Any, Optional, List

# Import Rich components
from rich.live import Live
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

from .stats import stats_lock, request_stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UI state
ui_exit_event = threading.Event()

class RichUI:
    """Rich-based console UI for OLOL proxy with stats and tables."""
    
    def __init__(self, params=None):
        """Initialize the Rich UI.
        
        Args:
            params: Dictionary of parameters controlling UI behavior
        """
        # Create Rich console
        self.console = Console()
        
        # Status messages
        self.status_messages = []
        self.max_status_messages = 15
        
        # Last refresh time
        self.last_update = 0
        self.update_interval = 0.2  # seconds
        
        # Layout elements
        self.layout = self._make_layout()
        
        # Verbosity settings
        self.verbose = params.get("verbose", False) if params else False
        self.debug = params.get("debug", False) if params else False
        
        # Add first status message
        self.add_status_message("Rich Console UI started")
        
    def _make_layout(self) -> Layout:
        """Create the layout structure for the UI."""
        layout = Layout(name="root")
        
        # Split the screen into main sections
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=1)
        )
        
        # Split the main area into sections
        layout["main"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2)
        )
        
        # Split left column
        layout["left"].split(
            Layout(name="stats", size=7),
            Layout(name="servers", ratio=1)
        )
        
        # Split right column
        layout["right"].split(
            Layout(name="status_messages", ratio=1),
            Layout(name="models", ratio=1)
        )
        
        return layout
        
    def add_status_message(self, message: str) -> None:
        """Add a status message to the display queue.
        
        Args:
            message: Status message to display
        """
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.status_messages.append(f"[cyan]{timestamp}[/cyan] - {message}")
        
        # Trim if too many messages
        if len(self.status_messages) > self.max_status_messages:
            self.status_messages.pop(0)
    
    def _create_header(self, stats: Dict[str, Any]) -> Panel:
        """Create header panel with basic info."""
        # Calculate uptime
        uptime_seconds = int(time.time() - stats["start_time"])
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        spinner = Spinner("dots", text="OLOL Proxy Server")
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            f"[bold green]{spinner} OLOL Proxy Server Status[/bold green]",
            f"[yellow]Uptime: {uptime_str}[/yellow]"
        )
        
        from . import app
        use_distributed_inference = getattr(app, 'use_distributed_inference', False)
        dist_status = f"[green]ENABLED[/green]" if use_distributed_inference else "[yellow]DISABLED[/yellow]"
        
        grid.add_row(f"Distributed Inference: {dist_status}", "")
        
        return Panel(grid, border_style="blue", box=box.ROUNDED)
    
    def _create_stats_panel(self, stats: Dict[str, Any]) -> Panel:
        """Create stats panel with request statistics."""
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        
        grid.add_row(
            "[bold]Request Statistics[/bold]", 
            f"[yellow]Active: {stats['active_requests']} | Total: {stats['total_requests']}[/yellow]"
        )
        
        stats_table = Table(expand=True, box=box.SIMPLE)
        stats_table.add_column("Type", justify="left", style="cyan")
        stats_table.add_column("Count", justify="right")
        stats_table.add_column("Last", justify="right")
        
        stats_table.add_row(
            "Generate", 
            str(stats['generate_requests']),
            f"{stats['last_generate_time']:.2f}s" if stats.get('last_generate_time') else "N/A"
        )
        stats_table.add_row(
            "Chat", 
            str(stats['chat_requests']),
            f"{stats['last_chat_time']:.2f}s" if stats.get('last_chat_time') else "N/A"
        )
        stats_table.add_row(
            "Embeddings", 
            str(stats['embedding_requests']),
            f"{stats['last_embedding_time']:.2f}s" if stats.get('last_embedding_time') else "N/A"
        )
        
        return Panel(
            stats_table,
            title="[bold]Request Stats[/bold]",
            border_style="green",
            box=box.ROUNDED
        )
    
    def _create_servers_panel(self, cluster=None) -> Panel:
        """Create servers panel with server status."""
        if not cluster:
            return Panel("No cluster data available", title="Servers", border_style="yellow")
        
        server_count = 0
        healthy_count = 0
        server_data = []
        
        try:
            with cluster.health_lock:
                server_count = len(cluster.server_addresses)
                healthy_count = sum(1 for v in cluster.server_health.values() if v)
                
                # Get data for each server
                for server in cluster.server_addresses:
                    healthy = cluster.server_health.get(server, False)
                    with cluster.server_lock:
                        load = cluster.server_loads.get(server, 0)
                    
                    # Get model count
                    model_count = 0
                    with cluster.model_lock:
                        for models in cluster.model_server_map.values():
                            if server in models:
                                model_count += 1
                                
                    server_data.append({
                        "address": server,
                        "healthy": healthy,
                        "load": load,
                        "model_count": model_count
                    })
        except Exception as e:
            if self.debug:
                self.add_status_message(f"Error getting server data: {str(e)}")
        
        # Create server table
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Server", style="dim")
        table.add_column("Health", justify="center")
        table.add_column("Load", justify="right")
        table.add_column("Models", justify="right")
        
        for server in server_data:
            health_text = "[green]✓ Healthy[/green]" if server["healthy"] else "[red]✗ Unhealthy[/red]"
            table.add_row(
                server["address"], 
                health_text,
                str(server["load"]), 
                str(server["model_count"])
            )
        
        title = f"[bold]Servers[/bold] [green]{healthy_count}[/green]/[blue]{server_count}[/blue] healthy"
        return Panel(table, title=title, border_style="blue", box=box.ROUNDED)
    
    def _create_models_panel(self, cluster=None) -> Panel:
        """Create models panel with model info."""
        if not cluster or not hasattr(cluster, 'model_manager'):
            return Panel("No model data available", title="Models", border_style="yellow")
        
        # Create models table
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Model", style="cyan")
        table.add_column("Context", justify="right")
        table.add_column("Servers", justify="right")
        
        try:
            with cluster.model_lock:
                # Try to get models with details
                if hasattr(cluster.model_manager, 'get_all_models_with_details'):
                    model_details = cluster.model_manager.get_all_models_with_details()
                else:
                    model_details = cluster.model_manager.get_all_models()
                
                # Show models (limit to 10 to avoid crowding)
                for model_name, details in list(model_details.items())[:10]:
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
                    
                    table.add_row(model_name, str(ctx_size), str(servers_count))
        except Exception as e:
            if self.debug:
                self.add_status_message(f"Error getting model data: {str(e)}")
            table.add_row("[red]Error loading models[/red]", "", "")
        
        return Panel(table, title="[bold]Available Models[/bold]", border_style="cyan", box=box.ROUNDED)
    
    def _create_status_panel(self) -> Panel:
        """Create status messages panel."""
        if not self.status_messages:
            return Panel("No status messages", title="Status", border_style="green")
        
        # Show the most recent messages
        # Reverse order to show newest at the bottom
        visible_messages = self.status_messages[-10:] if len(self.status_messages) > 10 else self.status_messages
        
        text = "\n".join(visible_messages)
        return Panel(Text.from_markup(text), title="[bold]Status Messages[/bold]", border_style="green", box=box.ROUNDED)
    
    def _create_footer(self) -> Panel:
        """Create footer panel with help text."""
        footer_text = "[bold]Press Ctrl+C to exit[/bold]"
        if self.debug:
            footer_text += " [red][DEBUG MODE][/red]"
        elif self.verbose:
            footer_text += " [yellow][VERBOSE][/yellow]"
            
        return Panel(Text.from_markup(footer_text), border_style="blue", box=box.ROUNDED)
    
    def _generate_layout(self) -> Layout:
        """Generate the complete layout with current data."""
        try:
            # Get stats safely with a lock
            with stats_lock:
                stats = request_stats.copy()
            
            # Import needed here to avoid circular imports
            from . import app
            cluster = getattr(app, 'cluster', None)
            
            # Update layout components
            self.layout["header"].update(self._create_header(stats))
            self.layout["stats"].update(self._create_stats_panel(stats))
            self.layout["servers"].update(self._create_servers_panel(cluster))
            self.layout["status_messages"].update(self._create_status_panel())
            self.layout["models"].update(self._create_models_panel(cluster))
            self.layout["footer"].update(self._create_footer())
            
            return self.layout
        
        except Exception as e:
            # In case of error, return a simple error panel
            logger.error(f"Error generating layout: {str(e)}")
            layout = Layout(Panel(f"[bold red]Error rendering UI: {str(e)}[/bold red]"))
            return layout
            
    def start(self):
        """Start the Rich UI."""
        try:
            # Run the UI with live updates
            with Live(self._generate_layout(), refresh_per_second=4, screen=True) as live:
                while not ui_exit_event.is_set():
                    # Update at specified interval
                    current_time = time.time()
                    if current_time - self.last_update >= self.update_interval:
                        self.last_update = current_time
                        live.update(self._generate_layout())
                    
                    # Avoid high CPU usage
                    time.sleep(0.1)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            ui_exit_event.set()
        except Exception as e:
            logger.error(f"UI error: {str(e)}")
            ui_exit_event.set()


def run_console_ui(params=None):
    """Run the Rich console UI in a separate thread.
    
    Args:
        params: Dictionary of parameters controlling UI behavior
    """
    try:
        # Start with Rich UI instead of curses UI
        ui = RichUI(params)
        
        # Import needed only here to avoid circular imports
        from . import app
        cluster = getattr(app, 'cluster', None)
        
        # Register listeners for discovery events to update UI
        if cluster and (params.get('verbose', False) or params.get('debug', False)):
            # Add a custom function to receive notifications when servers are discovered
            def handle_server_discovered(server_address, details=None):
                # Add server discovery to status messages
                ui.add_status_message(f"Server discovered: {server_address}")
                
            # Set up a callback for server discovery
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
    except Exception as e:
        logger.error(f"Failed to start UI: {str(e)}")
        ui_exit_event.set()