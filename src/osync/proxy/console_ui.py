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
from .db.database import get_db
from .db.sync_manager import get_sync_manager
from .cluster.registry import get_model_registry
from .queue.queue import get_queue_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UI state
ui_exit_event = threading.Event()

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
        self.max_status_messages = 15
        
        # Last refresh time
        self.last_update = 0
        self.update_interval = 0.2  # seconds
        
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
        self.add_status_message("Connecté à la base de données TinyDB")
        
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
            f"[bold green]{spinner_text} Ollama Sync Server Status[/bold green]",
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
            logger.debug(f"Erreur lors de la récupération des statistiques de file d'attente: {str(e)}")
        
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
            logger.debug(f"Erreur lors de la récupération des statistiques depuis la DB: {str(e)}")
        
        # Obtenir les stats de la file d'attente
        queue_stats = {}
        try:
            queue_stats = self.queue_manager.get_queue_stats()
        except Exception as e:
            logger.debug(f"Erreur lors de la récupération des statistiques de la file: {str(e)}")
        
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
                return Panel("Aucun serveur trouvé.\nVérifiez que vos serveurs Ollama sont bien accessibles.", 
                            title="Serveurs", border_style="yellow")
                
            # Création du tableau des serveurs
            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Adresse", style="dim")
            table.add_column("État", justify="center")
            table.add_column("Charge", justify="right")
            table.add_column("Modèles", justify="right")
            table.add_column("Backend", justify="right")
            
            # Afficher chaque serveur
            for server in servers:
                address = server.get("address", "Unknown")
                is_healthy = server.get("healthy", False)
                load = server.get("load", 0.0)
                models_count = len(server.get("models", []))
                backend = server.get("backend", "Inconnu")
                
                # Formater la charge
                load_str = f"{int(load * 100)}%" if load <= 1.0 else f"{load:.2f}"
                load_color = "green" if load < 0.7 else "yellow" if load < 0.9 else "red"
                
                table.add_row(
                    address, 
                    "[green]En ligne[/green]" if is_healthy else "[red]Hors ligne[/red]",
                    f"[{load_color}]{load_str}[/{load_color}]",
                    str(models_count) if is_healthy else "-",
                    str(backend) if is_healthy else "Inconnu"
                )
            
            # Calculer le nombre de serveurs en ligne
            healthy_servers = sum(1 for server in servers if server.get("healthy", False))
            return Panel(table, title=f"[bold]Serveurs[/bold] ({healthy_servers}/{len(servers)} en ligne)", 
                        border_style="blue", box=box.ROUNDED)
        except Exception as e:
            logger.exception("Erreur lors de la création du panneau des serveurs")
            return Panel(f"Erreur d'accès aux serveurs: {str(e)}", title="Serveurs", border_style="red")
    
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
                error_msg = models_status.get("error", "Erreur inconnue") if isinstance(models_status, dict) else str(models_status)
                return Panel(f"Erreur du registre: {error_msg}", title="Modèles", border_style="red")
            
            if not models_list:
                # Vérifier s'il y a des serveurs actifs
                servers = self.sync_manager.read_from_ram("servers")
                healthy_servers = sum(1 for server in servers if server.get("healthy", False))
                
                if healthy_servers == 0:
                    return Panel("Aucun serveur n'est disponible.\nVérifiez la connexion avec vos serveurs Ollama.", 
                                title="Modèles", border_style="yellow")
                
                return Panel("Aucun modèle disponible actuellement.\nVérifiez vos serveurs Ollama.", 
                            title="Modèles", border_style="yellow")
            
            # Créer le tableau des modèles
            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Nom", style="cyan")
            table.add_column("Taille", justify="right")
            table.add_column("Serveurs", justify="right")
            table.add_column("Statut", justify="right")
                
            # Afficher les modèles (limité pour éviter un tableau trop grand)
            for model_info in models_list[:10]:
                model_name = model_info.get("name", "Unknown")
                model_size = model_info.get("size", "?")
                servers = model_info.get("servers", [])
                status = model_info.get("status", "Prêt")
                
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
                    f"{len(servers)} serveur{'s' if len(servers) > 1 else ''}",
                    status_str
                )
                
            # Indiquer s'il y a plus de modèles que ceux affichés
            if len(models_list) > 10:
                table.add_row(f"...et {len(models_list) - 10} autres", "", "", "")
                
            # Compter les modèles prêts vs en cours de chargement
            ready_models = sum(1 for m in models_list if m.get("status", "").lower() in ["prêt", "ready"])
            return Panel(table, title=f"[bold]Modèles disponibles[/bold] ({ready_models}/{len(models_list)} prêts)", 
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
            self.layout["models"].update(self._create_models_panel())
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
        
        # Monitorer les changements dans la base de données pour les refléter dans l'UI
        ui.add_status_message("Surveillance active des changements de la base de données")
        
        # Start the UI
        ui.start()
    except KeyboardInterrupt:
        ui_exit_event.set()
    except Exception as e:
        logger.error(f"Failed to start UI: {str(e)}")
        ui_exit_event.set()