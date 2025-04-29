"""HTTP proxy for load balancing Ollama gRPC servers."""

# Ce fichier est le point d'entrée principal qui 
# importe tout depuis le sous-package proxy

from olol.proxy.app import run_proxy

cluster = None
coordinator = None
ui_thread = None
ui_active = False
use_distributed_inference = False

# Point d'entrée pour l'exécution directe
if __name__ == "__main__":
    # Default configuration
    run_proxy()