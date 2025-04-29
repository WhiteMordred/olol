"""HTTP proxy for load balancing Ollama gRPC servers."""

# Ce fichier est le point d'entrée principal qui 
# importe tout depuis le sous-package proxy

from olol.proxy.app import run_proxy, app
from olol.proxy.console_ui import ConsoleUI, run_console_ui
from olol.proxy.stats import update_request_stats, request_stats, stats_lock
from olol.proxy.health import health_checker
from olol.proxy.utils import create_grpc_client, adjust_context_length

# Variables globales nécessaires pour la compatibilité
cluster = None
coordinator = None
ui_thread = None
ui_active = False
use_distributed_inference = False

# Point d'entrée pour l'exécution directe
if __name__ == "__main__":
    # Default configuration
    run_proxy()