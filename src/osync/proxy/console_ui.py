"""Console UI for the Ollama proxy server using Rich."""

import threading
import time
import logging
import sys
import os
from typing import Dict, Any, Optional, List, Deque
from collections import deque
import select
from io import StringIO

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

# UI state
ui_exit_event = threading.Event()

# Stream for capturing stderr and stdout
class StreamCapture:
    def __init__(self, maxlen=500):
        self.buffer = deque(maxlen=maxlen)
        self.old_stderr = None
        self.old_stdout = None
        self.active = False
    
    def write(self, text):
        # Forward to original stream
        if self.old_stderr:
            self.old_stderr.write(text)
            self.old_stderr.flush()
        
        # Don't capture empty lines
        if text and not text.isspace():
            self.buffer.append(text)
    
    def flush(self):
        if self.old_stderr:
            self.old_stderr.flush()
    
    # Ajouter des méthodes nécessaires pour un objet "file-like"
    def isatty(self):
        return self.old_stderr.isatty() if self.old_stderr else False
        
    def fileno(self):
        return self.old_stderr.fileno() if self.old_stderr else None
    
    # Cette méthode est cruciale pour Flask/Click
    def __getattr__(self, name):
        # Déléguer tous les autres attributs au flux d'origine
        if self.old_stderr:
            return getattr(self.old_stderr, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def start_capture(self):
        if not self.active:
            self.old_stderr = sys.stderr
            self.old_stdout = sys.stdout
            sys.stderr = self
            sys.stdout = self
            self.active = True
    
    def stop_capture(self):
        if self.active:
            sys.stderr = self.old_stderr
            sys.stdout = self.old_stdout
            self.active = False
    
    def get_recent_output(self, n=20):
        return list(self.buffer)[-n:]

# Set up stream capture
stream_capture = StreamCapture(maxlen=1000)

# Créer un gestionnaire de logs pour capturer tous les messages
class LogCapture(logging.Handler):
    def __init__(self, level=logging.NOTSET, max_lines=500):
        super().__init__(level)
        self.log_messages = deque(maxlen=max_lines)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        self.debug_logs = deque(maxlen=max_lines)  # Logs de niveau DEBUG uniquement
        self.info_logs = deque(maxlen=max_lines)   # Logs de niveau INFO uniquement
        self.error_logs = deque(maxlen=max_lines)  # Logs de niveau WARNING, ERROR, CRITICAL
    
    def emit(self, record):
        # Formatter le message et l'ajouter à notre deque
        try:
            log_entry = self.formatter.format(record)
            
            # Ajouter la couleur selon le niveau de log
            formatted_entry = log_entry
            if record.levelno >= logging.ERROR:
                formatted_entry = f"[bold red]{log_entry}[/bold red]"
            elif record.levelno >= logging.WARNING:
                formatted_entry = f"[yellow]{log_entry}[/yellow]"
            elif record.levelno >= logging.DEBUG:
                formatted_entry = f"[dim]{log_entry}[/dim]"
                
            # Ajouter à la liste générale
            self.log_messages.append(formatted_entry)
            
            # Ajouter aussi à la liste spécifique en fonction du niveau
            if record.levelno >= logging.ERROR:
                self.error_logs.append(formatted_entry)
            elif record.levelno >= logging.INFO:
                self.info_logs.append(formatted_entry)
            elif record.levelno >= logging.DEBUG:
                self.debug_logs.append(formatted_entry)
                
        except Exception:
            self.handleError(record)

# Configurer le logging pour capturer tous les logs
log_capture = LogCapture(max_lines=500)

# Console Rich pour les logs
rich_console = Console(stderr=True)

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,  # Capturer tous les niveaux de logs
    format="%(message)s",
    handlers=[
        RichHandler(console=rich_console, rich_tracebacks=True, markup=True),
        log_capture  # Notre gestionnaire personnalisé
    ]
)
logger = logging.getLogger(__name__)

# S'assurer que les handlers sont bien configurés pour le root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Capturer tous les logs
if not any(isinstance(h, LogCapture) for h in root_logger.handlers):
    root_logger.addHandler(log_capture)

class RichUI:
    """Rich-based console UI for OllamaSync proxy with stats and tables."""
    
    def __init__(self, params=None):
        """Initialize the Rich UI.
        
        Args:
            params: Dictionary of parameters controlling UI behavior
        """
        # Commencer à capturer les flux de sortie standard
        stream_capture.start_capture()
        
        # Create Rich console
        self.console = Console()
        
        # Status messages
        self.status_messages = []
        self.max_status_messages = 50  # Augmenté pour conserver plus de messages
        
        # Requests log
        self.request_logs = deque(maxlen=100)  # Pour stocker les requêtes récentes
        
        # Interface mode
        self.show_logs = False  # Par défaut, on montre l'interface normale
        self.show_requests = False  # Par défaut, on ne montre pas la liste des requêtes
        self.log_level = "all"  # Options: "all", "debug", "info", "error"
        
        # Last refresh time
        self.last_update = 0
        self.update_interval = 0.5  # secondes
        
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
        self.add_status_message("Connected to database")
        self.add_status_message("Press 'l' to toggle logs | 'd' for debug logs | 'r' for requests | 'q' to exit")
        
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
            Layout(name="servers", ratio=1),
            Layout(name="db_sync", size=8 if self.debug else 0)  # Zone dédiée aux infos de sync DB
        )
        
        # Split right column
        layout["right"].split(
            Layout(name="status_messages", ratio=1),
            Layout(name="logs_area", ratio=2)  # Zone principale pour les logs
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
        mode_indicator = ""
        if self.show_logs:
            if self.log_level == "debug":
                mode_indicator = "[bold yellow]MODE DEBUG LOGS[/bold yellow]"
            elif self.log_level == "error":
                mode_indicator = "[bold red]MODE ERROR LOGS[/bold red]"
            else:
                mode_indicator = "[bold blue]MODE LOGS[/bold blue]"
        elif self.show_requests:
            mode_indicator = "[bold cyan]MODE REQUÊTES[/bold cyan]"
        else:
            mode_indicator = "[bold green]MODE NORMAL[/bold green]"
        
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
    
    def _create_db_sync_panel(self) -> Panel:
        """Create panel with database synchronization information."""
        try:
            # Récupérer des infos de synchronisation de la base de données
            sync_stats = {}
            try:
                sync_stats = self.sync_manager.get_sync_stats()
            except:
                sync_stats = {
                    "last_sync": time.time(),
                    "ram_items": 0,
                    "db_items": 0,
                    "sync_interval": 0
                }
            
            # Création du tableau de synchronisation
            table = Table(box=box.SIMPLE, expand=True)
            table.add_column("Collection", style="cyan")
            table.add_column("RAM", justify="right")
            table.add_column("DB", justify="right")
            table.add_column("Status", justify="center")
            
            # Ajouter des lignes pour différentes collections
            for collection_name, stats in sync_stats.get("collections", {}).items():
                ram_count = stats.get("ram_count", 0)
                db_count = stats.get("db_count", 0)
                status = "[green]✓[/green]" if ram_count == db_count else "[yellow]⚠[/yellow]"
                
                table.add_row(
                    collection_name,
                    str(ram_count),
                    str(db_count),
                    status
                )
            
            # Si pas de collections spécifiques, ajouter des statistiques générales
            if not sync_stats.get("collections"):
                ram_items = sync_stats.get("ram_items", 0)
                db_items = sync_stats.get("db_items", 0)
                
                table.add_row(
                    "Total",
                    str(ram_items),
                    str(db_items),
                    "[green]✓[/green]" if ram_items == db_items else "[yellow]⚠[/yellow]"
                )
            
            # Ajouter des infos sur la dernière synchronisation
            last_sync = sync_stats.get("last_sync", 0)
            sync_interval = sync_stats.get("sync_interval", 0)
            
            # Formater le temps écoulé depuis la dernière synchronisation
            time_since_sync = time.time() - last_sync
            if time_since_sync < 60:
                time_str = f"{int(time_since_sync)}s ago"
            else:
                time_str = f"{int(time_since_sync / 60)}m ago"
            
            footer = Text.from_markup(f"Last sync: [cyan]{time_str}[/cyan] | Interval: {sync_interval}s")
            
            return Panel(
                Table.grid(expand=True)
                .add_column()
                .add_row(table)
                .add_row(footer),
                title="[bold]Database Sync[/bold]",
                border_style="yellow",
                box=box.ROUNDED
            )
            
        except Exception as e:
            logger.debug(f"Error creating DB sync panel: {str(e)}")
            return Panel("Error fetching sync info", title="DB Sync", border_style="red")
    
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
        visible_count = 15
        visible_messages = self.status_messages[-visible_count:] if len(self.status_messages) > visible_count else self.status_messages
        
        text = "\n".join(visible_messages)
        return Panel(Text.from_markup(text), title=f"[bold]Status Messages[/bold] ({len(self.status_messages)} total)", border_style="green", box=box.ROUNDED)
    
    def _create_logs_panel(self) -> Panel:
        """Create logs panel with captured log messages."""
        # Choisir les logs en fonction du niveau sélectionné
        logs_to_show = []
        
        if self.log_level == "debug":
            logs_to_show = list(log_capture.debug_logs)
            title = "[bold yellow]DEBUG Logs[/bold yellow]"
            style = "yellow"
        elif self.log_level == "error":
            logs_to_show = list(log_capture.error_logs)
            title = "[bold red]ERROR Logs[/bold red]"
            style = "red"
        else:
            # Tous les logs (mais limités aux 30 plus récents)
            logs_to_show = list(log_capture.log_messages)
            title = "[bold]All Logs[/bold]"
            style = "blue"
        
        if not logs_to_show:
            return Panel(f"No {self.log_level} logs captured yet.", title=title, border_style=style)
        
        # Obtenir les logs récents (basé sur la taille d'écran estimée)
        visible_logs = logs_to_show[-25:] if len(logs_to_show) > 25 else logs_to_show
        
        # Ajouter les sorties capturées de stdout/stderr
        stdout_logs = stream_capture.get_recent_output(5)
        if stdout_logs:
            if visible_logs:
                visible_logs.append("")
                visible_logs.append("[bold magenta]--- Standard Output ---[/bold magenta]")
            for line in stdout_logs:
                visible_logs.append(line.rstrip())
        
        text = "\n".join(visible_logs)
        return Panel(
            Text.from_markup(text),
            title=f"{title} ({len(logs_to_show)} captured)",
            border_style=style,
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
        footer_text = "[bold]Press 'l' to toggle logs | 'd' for debug logs | 'e' for error logs | 'r' for requests | 'q' to exit[/bold]"
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
            
        # Ajouter le nombre de logs capturés
        footer_text += f" | Logs: {len(log_capture.log_messages)} (Debug: {len(log_capture.debug_logs)}, Error: {len(log_capture.error_logs)})"
            
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
            
            # Ajouter le panneau de sync DB en mode debug
            if self.debug:
                self.layout["db_sync"].update(self._create_db_sync_panel())
                self.layout["db_sync"].visible = True
            else:
                self.layout["db_sync"].visible = False
            
            self.layout["status_messages"].update(self._create_status_panel())
            
            # Choisir ce qu'on affiche dans la zone logs_area en fonction du mode
            if self.show_logs or self.debug:
                # Mode logs : afficher les logs
                self.layout["logs_area"].update(self._create_logs_panel())
            elif self.show_requests:
                # Mode requêtes : afficher les requêtes récentes
                self.layout["logs_area"].update(self._create_requests_panel())
            else:
                # Mode normal : afficher les modèles
                self.layout["logs_area"].update(self._create_models_panel())
                
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
                                self.log_level = "all"  # Afficher tous les logs par défaut
                                self.show_requests = False  # Désactiver mode requêtes
                                self.add_status_message("Switched to LOGS mode (all levels)")
                            else:
                                self.add_status_message("Switched to NORMAL mode")
                        elif key == 'd':  # Activer les logs de debug
                            self.show_logs = True
                            self.log_level = "debug"
                            self.show_requests = False
                            self.add_status_message("Switched to DEBUG logs mode")
                        elif key == 'e':  # Activer les logs d'erreur
                            self.show_logs = True
                            self.log_level = "error"
                            self.show_requests = False
                            self.add_status_message("Switched to ERROR logs mode")
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
            
            # Démarrer le thread de surveillance du clavier
            self.keyboard_thread = threading.Thread(target=self._keyboard_monitor, daemon=True)
            self.keyboard_thread.start()
            
            # Si on est en mode debug, activer automatiquement l'affichage des logs
            if self.debug:
                self.show_logs = True
                self.log_level = "debug"
                self.add_status_message("Debug mode: logs display activated automatically")
            
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
                    time.sleep(0.5)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            ui_exit_event.set()
        except Exception as e:
            logger.error(f"UI error: {str(e)}")
            ui_exit_event.set()
        finally:
            # Arrêter la capture de flux
            stream_capture.stop_capture()
            
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
        finally:
            # Arrêter la capture de flux
            stream_capture.stop_capture()


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