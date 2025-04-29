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
        
        # Utiliser un caractère fixe au lieu de spinner pour éviter l'erreur de render()
        spinner_text = "•"
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            f"[bold green]{spinner_text} OLOL Proxy Server Status[/bold green]",
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
        if not cluster or not hasattr(cluster, 'server_addresses'):
            return Panel("Initialisation du cluster en cours...", title="Servers", border_style="yellow")
        
        try:
            # Récupération des informations sans verrous qui pourraient bloquer
            server_addresses = getattr(cluster, 'server_addresses', [])[:]  # Copie pour éviter les problèmes de concurrence
            
            if not server_addresses:
                return Panel("Aucun serveur trouvé.\nVérifiez que vos serveurs Ollama sont bien accessibles.", 
                            title="Servers", border_style="yellow")
                
            # Création du tableau des serveurs
            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Serveur", style="dim")
            table.add_column("État", justify="center")
            
            # Récupérer l'état de santé de manière sécurisée
            server_health = {}
            try:
                # Tentative d'accès à l'état de santé sans verrou
                server_health = {s: cluster.server_health.get(s, False) for s in server_addresses}
            except:
                # En cas d'erreur, considérer tous les serveurs comme disponibles pour l'affichage
                server_health = {s: True for s in server_addresses}
            
            # Afficher chaque serveur
            for server in server_addresses:
                is_healthy = server_health.get(server, True)  # Par défaut considéré comme sain
                table.add_row(server, "[green]✓ Disponible[/green]" if is_healthy else "[red]✗ Indisponible[/red]")
                
            return Panel(table, title=f"[bold]Serveurs[/bold] ({len(server_addresses)} détectés)", 
                        border_style="blue", box=box.ROUNDED)
        except Exception as e:
            logger.exception("Erreur lors de la création du panneau des serveurs")
            return Panel(f"Erreur d'accès aux serveurs: {str(e)}", title="Serveurs", border_style="red")
    
    def _create_models_panel(self, cluster=None) -> Panel:
        """Create models panel with model availability info."""
        if not cluster:
            return Panel("Initialisation du cluster en cours...", title="Modèles", border_style="yellow")
            
        try:
            # Récupération des modèles de manière sécurisée sans verrous
            models_info = []
            
            # Essayer d'abord d'accéder à model_server_map directement
            try:
                server_map = getattr(cluster, 'model_server_map', {})
                
                # Créer une liste de modèles à partir de la carte
                for model_name, servers in server_map.items():
                    if servers:  # Si au moins un serveur a ce modèle
                        models_info.append({
                            'name': model_name,
                            'server': servers[0]  # Utiliser le premier serveur
                        })
            except Exception:
                # Si problème d'accès à model_server_map, utiliser l'approche directe
                pass
                
            # Si aucun modèle n'est trouvé dans la carte, essayer l'approche directe
            if not models_info:
                for server in getattr(cluster, 'server_addresses', []):
                    # Essayer de contacter directement chaque serveur, sans utiliser les verrous du cluster
                    try:
                        if server.count(':') == 1:
                            host, port_str = server.split(':')
                            port = int(port_str)
                            
                            # Créer un client temporaire
                            from olol.sync.client import OllamaClient
                            client = OllamaClient(host=host, port=port)
                            
                            try:
                                models_response = client.list_models()
                                
                                if hasattr(models_response, 'models'):
                                    for model in models_response.models:
                                        model_name = model.name
                                        if model_name not in [m['name'] for m in models_info]:
                                            models_info.append({
                                                'name': model_name,
                                                'server': server
                                            })
                            finally:
                                client.close()
                    except Exception:
                        # Ignorer les erreurs pour ce serveur et continuer
                        continue
            
            # Si toujours aucun modèle trouvé
            if not models_info:
                # Vérifier combien de serveurs sont disponibles
                try:
                    healthy_servers = sum(1 for h in getattr(cluster, 'server_health', {}).values() if h)
                    if healthy_servers == 0:
                        return Panel("Aucun serveur n'est disponible.\nVérifiez la connexion avec vos serveurs Ollama.", 
                                    title="Modèles", border_style="yellow")
                except:
                    pass
                
                return Panel("Aucun modèle disponible actuellement.\nVérifiez vos serveurs Ollama.", 
                            title="Modèles", border_style="yellow")
            
            # Créer le tableau des modèles
            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Modèle", style="cyan")
            table.add_column("Serveur", justify="right")
                
            # Afficher les modèles (limité pour éviter un tableau trop grand)
            for model_info in models_info[:10]:
                table.add_row(model_info['name'], model_info['server'])
                
            # Indiquer s'il y a plus de modèles que ceux affichés
            if len(models_info) > 10:
                table.add_row(f"...et {len(models_info) - 10} autres", "")
                
            return Panel(table, title=f"[bold]Modèles disponibles[/bold] ({len(models_info)})", 
                        border_style="cyan", box=box.ROUNDED)
                
        except Exception as e:
            logger.exception("Erreur lors de la création du panneau des modèles")
            return Panel(f"Erreur d'accès aux modèles: {str(e)}", title="Modèles", border_style="red")
    
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
            # Ajouter un délai initial pour permettre au cluster de s'initialiser
            time.sleep(1.0)
            
            # Run the UI with live updates - réduction du taux de rafraîchissement pour éviter les effets de glitch
            with Live(self._generate_layout(), refresh_per_second=2, screen=True) as live:
                while not ui_exit_event.is_set():
                    try:
                        # Update at specified interval
                        current_time = time.time()
                        if current_time - self.last_update >= self.update_interval:
                            self.last_update = current_time
                            # Utiliser un délai entre chaque mise à jour du layout
                            layout = self._generate_layout()
                            live.update(layout)
                    except Exception as e:
                        # Isoler les erreurs de mise à jour pour éviter que l'interface ne se bloque
                        logger.error(f"Error updating UI: {str(e)}")
                        self.add_status_message(f"UI update error: {str(e)}")
                    
                    # Augmenter le délai pour réduire l'utilisation du CPU et les glitches
                    time.sleep(0.25)
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