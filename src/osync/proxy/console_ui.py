"""Console UI for the Ollama proxy server using Rich."""

import threading
import time
import logging
from typing import Dict, Any, Optional, List, Deque
from collections import deque
import os

# Import Rich components
from rich.live import Live
from rich.console import Console, RenderableType
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.logging import RichHandler
from rich.box import Box
from rich import box
from rich.align import Align

from .stats import stats_lock, request_stats
from .db.database import get_db
from .db.sync_manager import get_sync_manager
from .cluster.registry import get_model_registry
from .queue.queue import get_queue_manager

# Set up logging
# Créer un logger specifique pour Rich
console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# UI state
ui_exit_event = threading.Event()

# Créer un gestionnaire de logs pour capturer tous les messages
class LogCapture(logging.Handler):
    def __init__(self, level=logging.NOTSET, max_lines=200):
        super().__init__(level)
        self.log_messages = deque(maxlen=max_lines)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    def emit(self, record):
        # Formatter le message et l'ajouter à notre deque
        try:
            log_entry = self.formatter.format(record)
            # Ajouter la couleur selon le niveau de log
            if record.levelno >= logging.ERROR:
                log_entry = f"[bold red]{log_entry}[/bold red]"
            elif record.levelno >= logging.WARNING:
                log_entry = f"[yellow]{log_entry}[/yellow]"
            elif record.levelno >= logging.DEBUG:
                log_entry = f"[dim]{log_entry}[/dim]"
            
            self.log_messages.append(log_entry)
        except Exception:
            self.handleError(record)

# Installer le gestionnaire de logs
log_capture = LogCapture(max_lines=200)
root_logger = logging.getLogger()
root_logger.addHandler(log_capture)

class RichUI:
    """Rich-based console UI for OllamaSync proxy with stats and tables."""
    
    def __init__(self, params=None):
        """Initialize the Rich UI.
        
        Args:
            params: Dictionary of parameters controlling UI behavior
        """
        # Create Rich console
        self.console = Console()
        
        # Status messages
        self.status_messages = []
        self.max_status_messages = 30  # Augmenté de 15 à 30
        
        # Requests log
        self.request_logs = deque(maxlen=50)  # Pour stocker les requêtes récentes
        
        # Interface mode
        self.show_logs = False  # Par défaut, on montre l'interface normale
        self.show_requests = False  # Par défaut, on ne montre pas la liste des requêtes
        
        # Last refresh time
        self.last_update = 0
        self.update_interval = 0.5  # Augmenté de 0.2 à 0.5 secondes
        
        # Layout elements
        self.layout = self._make_layout()
        
        # Verbosity settings
        self.verbose = params.get("verbose", False) if params else False
        self.debug = params.get("debug", False) if params else False
        
        # Access to persistent storage
        self.db = get_db()
        self.sync_manager = get_sync_manager()
        self.model_registry = get_model_registry()
        self.queue_manager = get_queue_manager()
        
        # Add first status message
        self.add_status_message("Rich Console UI started")
        self.add_status_message("Connected to TinyDB database")
        self.add_status_message("Press 'l' to toggle logs display, 'r' for requests, 'q' to exit")
        
        # Keyboard handler thread
        self.keyboard_thread = None
        self.keyboard_monitoring = False
        
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
            Layout(name="logs_or_models", ratio=1)  # Cette zone affichera soit les logs soit les modèles
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
    
    def add_request_log(self, request_data: Dict[str, Any]) -> None:
        """Add a request log to the requests display queue.
        
        Args:
            request_data: Request data to log
        """
        self.request_logs.append(request_data)
    
    def _create_header(self, stats: Dict[str, Any]) -> Panel:
        """Create header panel with basic info."""
        # Calculate uptime
        uptime_seconds = int(time.time() - stats["start_time"])
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        # Utiliser un caractère fixe au lieu de spinner pour éviter l'erreur de render()
        spinner_text = "•"
        
        # Ajouter une indication sur le mode d'affichage actuel
        mode_indicator = "[bold yellow]MODE LOGS[/bold yellow]" if self.show_logs else \
                       "[bold cyan]MODE REQUÊTES[/bold cyan]" if self.show_requests else \
                       "[bold green]MODE NORMAL[/bold green]"
        
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            f"[bold green]{spinner_text} Ollama Sync Server Status[/bold green] - {mode_indicator}",
            f"[yellow]Uptime: {uptime_str}[/yellow]"
        )
        
        # Récupérer les paramètres depuis la base de données
        config = self.db.search_one("config", lambda q: q.key == "distributed_inference")
        use_distributed_inference = config.get("value", False) if config else False
        
        dist_status = f"[green]ENABLED[/green]" if use_distributed_inference else "[yellow]DISABLED[/yellow]"
        
        grid.add_row(f"Distributed Inference: {dist_status}", "")
        
        # Ajouter des statistiques de la file d'attente
        try:
            queue_stats = self.queue_manager.get_queue_stats()
            pending_count = queue_stats.get("pending", 0)
            processing_count = queue_stats.get("processing", 0)
            grid.add_row(f"Queue: [cyan]{pending_count}[/cyan] pending, [cyan]{processing_count}[/cyan] processing", "")
        except Exception as e:
            logger.debug(f"Error retrieving queue statistics: {str(e)}")
        
        return Panel(grid, border_style="blue", box=box.ROUNDED)
    
    def _create_stats_panel(self, stats: Dict[str, Any]) -> Panel:
        """Create stats panel with request statistics."""
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        
        # Récupérer les statistiques de la base de données
        try:
            db_stats = self.db.search_one("stats", lambda q: q.key == "request_stats") or {}
            if isinstance(db_stats, dict) and "value" in db_stats:
                db_stats = db_stats["value"]
            else:
                db_stats = {}
            
            # Fusionner avec les stats en mémoire
            for key, value in db_stats.items():
                if key not in stats:
                    stats[key] = value
        except Exception as e:
            logger.debug(f"Error retrieving statistics from DB: {str(e)}")
        
        # Obtenir les stats de la file d'attente
        queue_stats = {}
        try:
            queue_stats = self.queue_manager.get_queue_stats()
        except Exception as e:
            logger.debug(f"Error retrieving queue statistics: {str(e)}")
        
        active_requests = stats.get('active_requests', 0)
        total_requests = stats.get('total_requests', 0)
        
        # Intégrer les données de la file d'attente si disponibles
        if queue_stats:
            queue_total = queue_stats.get("total_requests", 0)
            total_requests = max(total_requests, queue_total)  # Prendre le plus grand
        
        grid.add_row(
            "[bold]Request Statistics[/bold]", 
            f"[yellow]Active: {active_requests} | Total: {total_requests}[/yellow]"
        )
        
        stats_table = Table(expand=True, box=box.SIMPLE)
        stats_table.add_column("Type", justify="left", style="cyan")
        stats_table.add_column("Count", justify="right")
        stats_table.add_column("Last", justify="right")
        
        stats_table.add_row(
            "Generate", 
            str(stats.get('generate_requests', 0)),
            f"{stats.get('last_generate_time', 0):.2f}s" if stats.get('last_generate_time') else "N/A"
        )
        stats_table.add_row(
            "Chat", 
            str(stats.get('chat_requests', 0)),
            f"{stats.get('last_chat_time', 0):.2f}s" if stats.get('last_chat_time') else "N/A"
        )
        stats_table.add_row(
            "Embeddings", 
            str(stats.get('embedding_requests', 0)),
            f"{stats.get('last_embedding_time', 0):.2f}s" if stats.get('last_embedding_time') else "N/A"
        )
        
        # Ajouter des données du système de file d'attente
        if queue_stats:
            stats_table.add_row(
                "Completed", 
                str(queue_stats.get("completed", 0)),
                ""
            )
            stats_table.add_row(
                "Failed/Canceled", 
                str(queue_stats.get("failed", 0) + queue_stats.get("canceled", 0)),
                ""
            )
        
        return Panel(
            stats_table,
            title="[bold]Request Stats[/bold]",
            border_style="green",
            box=box.ROUNDED
        )
    
    def _create_servers_panel(self) -> Panel:
        """Create servers panel with server status from database."""
        try:
            # Récupérer les serveurs depuis la base de données
            servers = self.sync_manager.read_from_ram("servers")
            
            if not servers:
                return Panel("No server found.\nCheck that your Ollama servers are accessible.", 
                            title="Servers", border_style="yellow")
                
            # Création du tableau des serveurs
            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Address", style="dim")
            table.add_column("Status", justify="center")
            table.add_column("Load", justify="right")
            table.add_column("Models", justify="right")
            table.add_column("Backend", justify="right")
            
            # Afficher chaque serveur
            for server in servers:
                address = server.get("address", "Unknown")
                is_healthy = server.get("healthy", False)
                load = server.get("load", 0.0)
                models_count = len(server.get("models", []))
                backend = server.get("backend", "Unknown")
                
                # Formater la charge
                load_str = f"{int(load * 100)}%" if load <= 1.0 else f"{load:.2f}"
                load_color = "green" if load < 0.7 else "yellow" if load < 0.9 else "red"
                
                table.add_row(
                    address, 
                    "[green]Online[/green]" if is_healthy else "[red]Offline[/red]",
                    f"[{load_color}]{load_str}[/{load_color}]",
                    str(models_count) if is_healthy else "-",
                    str(backend) if is_healthy else "Unknown"
                )
            
            # Calculer le nombre de serveurs en ligne
            healthy_servers = sum(1 for server in servers if server.get("healthy", False))
            return Panel(table, title=f"[bold]Servers[/bold] ({healthy_servers}/{len(servers)} online)", 
                        border_style="blue", box=box.ROUNDED)
        except Exception as e:
            logger.exception("Error creating servers panel")
            return Panel(f"Error accessing servers: {str(e)}", title="Servers", border_style="red")
    
    def _create_models_panel(self) -> Panel:
        """Create models panel with model availability from registry."""
        try:
            # Utiliser le registre de modèles pour obtenir les informations à jour
            models_status = self.model_registry.get_model_status()
            
            models_list = []
            if isinstance(models_status, dict) and "models" in models_status:
                models_list = models_status["models"]
            elif not isinstance(models_status, dict) or "error" in models_status:
                # En cas d'erreur spécifique du registre
                error_msg = models_status.get("error", "Unknown error") if isinstance(models_status, dict) else str(models_status)
                return Panel(f"Registry error: {error_msg}", title="Models", border_style="red")
            
            if not models_list:
                # Vérifier s'il y a des serveurs actifs
                servers = self.sync_manager.read_from_ram("servers")
                healthy_servers = sum(1 for server in servers if server.get("healthy", False))
                
                if healthy_servers == 0:
                    return Panel("No server is available.\nCheck connection with your Ollama servers.", 
                                title="Models", border_style="yellow")
                
                return Panel("No model available.\nCheck your Ollama servers.", 
                            title="Models", border_style="yellow")
            
            # Créer le tableau des modèles
            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Name", style="cyan")
            table.add_column("Size", justify="right")
            table.add_column("Servers", justify="right")
            table.add_column("Status", justify="right")
                
            # Afficher les modèles (limité pour éviter un tableau trop grand)
            for model_info in models_list[:10]:
                model_name = model_info.get("name", "Unknown")
                model_size = model_info.get("size", "?")
                servers = model_info.get("servers", [])
                status = model_info.get("status", "Ready")
                
                # Formater la taille du modèle
                size_str = f"{model_size} GB" if isinstance(model_size, (int, float)) else str(model_size)
                
                # Formater le statut avec couleur
                if status.lower() in ["prêt", "ready"]:
                    status_str = f"[green]{status}[/green]"
                elif status.lower() in ["en cours", "loading", "downloading"]:
                    status_str = f"[yellow]{status}[/yellow]"
                else:
                    status_str = status
                
                table.add_row(
                    model_name, 
                    size_str,
                    f"{len(servers)} server{'s' if len(servers) > 1 else ''}",
                    status_str
                )
                
            # Indiquer s'il y a plus de modèles que ceux affichés
            if len(models_list) > 10:
                table.add_row(f"...and {len(models_list) - 10} more", "", "", "")
                
            # Compter les modèles prêts vs en cours de chargement
            ready_models = sum(1 for m in models_list if m.get("status", "").lower() in ["prêt", "ready"])
            return Panel(table, title=f"[bold]Available Models[/bold] ({ready_models}/{len(models_list)} ready)", 
                        border_style="cyan", box=box.ROUNDED)
                
        except Exception as e:
            logger.exception("Error creating models panel")
            return Panel(f"Error accessing models: {str(e)}", title="Models", border_style="red")
    
    def _create_status_panel(self) -> Panel:
        """Create status messages panel."""
        if not self.status_messages:
            return Panel("No status messages", title="Status", border_style="green")
        
        # Show the most recent messages
        # Afficher plus de messages quand on est en mode logs (20 au lieu de 10)
        visible_count = 20 if self.show_logs else 10
        visible_messages = self.status_messages[-visible_count:] if len(self.status_messages) > visible_count else self.status_messages
        
        text = "\n".join(visible_messages)
        return Panel(Text.from_markup(text), title="[bold]Status Messages[/bold]", border_style="green", box=box.ROUNDED)
    
    def _create_logs_panel(self) -> Panel:
        """Create logs panel with captured log messages."""
        if not log_capture.log_messages:
            return Panel("No log messages captured yet.", title="Logs", border_style="yellow")
        
        # Calculer combien de logs on peut afficher en fonction de la hauteur disponible
        # Estimation de 20 lignes disponibles, mais cela peut varier
        visible_logs = list(log_capture.log_messages)[-20:]
        
        text = "\n".join(visible_logs)
        return Panel(
            Text.from_markup(text),
            title=f"[bold]Logs[/bold] ({len(log_capture.log_messages)} captured)",
            border_style="yellow",
            box=box.ROUNDED
        )
        
    def _create_requests_panel(self) -> Panel:
        """Create requests panel with recent requests."""
        if not self.request_logs:
            return Panel("No requests captured yet.", title="Recent Requests", border_style="magenta")
        
        # Créer un tableau pour les requêtes récentes
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("ID", style="dim", width=8)
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Model", width=12)
        table.add_column("Time", justify="right", width=8)
        table.add_column("Status", justify="right", width=8)
        
        # Afficher les requêtes les plus récentes (15 max)
        for req in list(self.request_logs)[-15:]:
            req_id = req.get("id", "?")[:7]  # Tronquer l'ID
            req_type = req.get("type", "Unknown")
            model = req.get("model", "Unknown")
            duration = req.get("duration", 0)
            status = req.get("status", "Unknown")
            
            # Formater le statut avec couleur
            if status.lower() in ["completed", "success"]:
                status_str = f"[green]{status}[/green]"
            elif status.lower() in ["processing", "pending"]:
                status_str = f"[yellow]{status}[/yellow]"
            elif status.lower() in ["failed", "error"]:
                status_str = f"[red]{status}[/red]"
            else:
                status_str = status
                
            # Formater la durée
            duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else str(duration)
            
            table.add_row(
                req_id,
                req_type,
                model,
                duration_str,
                status_str
            )
        
        return Panel(
            table,
            title=f"[bold]Recent Requests[/bold] ({len(self.request_logs)} captured)",
            border_style="magenta",
            box=box.ROUNDED
        )
    
    def _create_footer(self) -> Panel:
        """Create footer panel with help text."""
        footer_text = "[bold]Press 'l' to toggle logs | 'r' to toggle requests | 'q' to exit[/bold]"
        if self.debug:
            footer_text += " [red][DEBUG MODE][/red]"
        elif self.verbose:
            footer_text += " [yellow][VERBOSE][/yellow]"
            
        # Ajouter des statistiques de la file d'attente
        try:
            queue_stats = self.queue_manager.get_queue_stats()
            if "batches" in queue_stats:
                batch_info = queue_stats["batches"]
                footer_text += f" | Batches: {batch_info.get('total', 0)} ({batch_info.get('pending', 0)} pending)"
        except Exception:
            pass
            
        return Panel(Text.from_markup(footer_text), border_style="blue", box=box.ROUNDED)
    
    def _generate_layout(self) -> Layout:
        """Generate the complete layout with current data."""
        try:
            # Get stats safely with a lock
            with stats_lock:
                stats = request_stats.copy()
            
            # Update layout components
            self.layout["header"].update(self._create_header(stats))
            self.layout["stats"].update(self._create_stats_panel(stats))
            self.layout["servers"].update(self._create_servers_panel())
            self.layout["status_messages"].update(self._create_status_panel())
            
            # Choisir ce qu'on affiche dans la zone logs_or_models en fonction du mode
            if self.show_logs:
                # Mode logs : afficher les logs de debug
                self.layout["logs_or_models"].update(self._create_logs_panel())
            elif self.show_requests:
                # Mode requêtes : afficher les requêtes récentes
                self.layout["logs_or_models"].update(self._create_requests_panel())
            else:
                # Mode normal : afficher les modèles
                self.layout["logs_or_models"].update(self._create_models_panel())
                
            self.layout["footer"].update(self._create_footer())
            
            return self.layout
        
        except Exception as e:
            # In case of error, return a simple error panel
            logger.error(f"Error generating layout: {str(e)}")
            layout = Layout(Panel(f"[bold red]Error rendering UI: {str(e)}[/bold red]"))
            return layout
    
    def _keyboard_monitor(self):
        """Monitor keyboard input for UI control."""
        try:
            import tty
            import sys
            import termios
            
            self.keyboard_monitoring = True
            # Save terminal settings
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            
            try:
                # Set terminal to raw mode
                tty.setraw(fd)
                
                while not ui_exit_event.is_set() and self.keyboard_monitoring:
                    # Wait for a keypress (with timeout)
                    ready_to_read = select.select([sys.stdin], [], [], 0.1)[0]
                    if ready_to_read:
                        key = sys.stdin.read(1)
                        
                        # Process the key
                        if key == 'q':  # Exit
                            ui_exit_event.set()
                        elif key == 'l':  # Toggle logs display
                            self.show_logs = not self.show_logs
                            if self.show_logs:
                                self.show_requests = False  # Désactiver mode requêtes
                                self.add_status_message("Switched to LOGS mode")
                            else:
                                self.add_status_message("Switched to NORMAL mode")
                        elif key == 'r':  # Toggle requests display
                            self.show_requests = not self.show_requests
                            if self.show_requests:
                                self.show_logs = False  # Désactiver mode logs
                                self.add_status_message("Switched to REQUESTS mode")
                            else:
                                self.add_status_message("Switched to NORMAL mode")
            finally:
                # Restore terminal settings
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                self.keyboard_monitoring = False
        except ImportError:
            # If we can't import tty or termios, we're probably on Windows
            logger.warning("Keyboard monitoring not available (requires Unix-like OS)")
            self.keyboard_monitoring = False
        except Exception as e:
            logger.error(f"Error in keyboard monitor: {e}")
            self.keyboard_monitoring = False
            
    def start(self):
        """Start the Rich UI."""
        try:
            # Ajouter un délai initial pour permettre au cluster de s'initialiser
            time.sleep(1.0)
            
            # Import select ici pour éviter des erreurs d'import sur Windows
            import select
            
            # Démarrer le thread de surveillance du clavier
            self.keyboard_thread = threading.Thread(target=self._keyboard_monitor, daemon=True)
            self.keyboard_thread.start()
            
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
                    time.sleep(0.5)  # Augmenté de 0.25 à 0.5
        except ImportError:
            logger.error("Required module 'select' not available")
            # Fallback to a simpler UI without keyboard control
            self._start_simple()
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            ui_exit_event.set()
        except Exception as e:
            logger.error(f"UI error: {str(e)}")
            ui_exit_event.set()
            
    def _start_simple(self):
        """Start a simpler version of the UI without keyboard monitoring."""
        try:
            with Live(self._generate_layout(), refresh_per_second=1, screen=True) as live:
                while not ui_exit_event.is_set():
                    try:
                        layout = self._generate_layout()
                        live.update(layout)
                        time.sleep(1.0)  # Rafraîchissement plus lent
                    except Exception as e:
                        logger.error(f"Error updating UI: {str(e)}")
                    
            time.sleep(0.5)
        except KeyboardInterrupt:
            ui_exit_event.set()


def run_console_ui(params=None):
    """Run the Rich console UI in a separate thread.
    
    Args:
        params: Dictionary of parameters controlling UI behavior
    """
    try:
        # Start with Rich UI
        ui = RichUI(params)
        
        # Monitorer les changements dans la base de données pour les refléter dans l'UI
        ui.add_status_message("Monitoring database changes")
        
        # Start the UI
        ui.start()
    except KeyboardInterrupt:
        ui_exit_event.set()
    except Exception as e:
        logger.error(f"Failed to start UI: {str(e)}")
        ui_exit_event.set()