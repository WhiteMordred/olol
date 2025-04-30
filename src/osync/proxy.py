"""HTTP proxy for load balancing Ollama gRPC servers."""

# Ce fichier est le point d'entrée principal qui 
# importe tout depuis le sous-package proxy

import argparse
import sys
from typing import List, Optional

from osync.proxy.app import run_proxy

# Points d'accès globaux pour les tests et l'introspection
cluster = None
cluster_manager = None
coordinator = None
ui_thread = None
ui_active = False
use_distributed_inference = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ollama Sync - Load balancer pour serveurs Ollama")
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Adresse IP du serveur proxy (défaut: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port du serveur proxy (défaut: 8000)')
    parser.add_argument('--servers', type=str, nargs='+', default=None,
                        help='Liste des adresses de serveurs Ollama (format: host:port)')
    
    # Options d'interface
    parser.add_argument('--console', action='store_true', default=True,
                        help='Activer l\'interface console (par défaut)')
    parser.add_argument('--no-console', action='store_false', dest='console',
                        help='Désactiver l\'interface console')
    parser.add_argument('--web', action='store_true', default=True,
                        help='Activer l\'interface web (par défaut)')
    parser.add_argument('--no-web', action='store_false', dest='web',
                        help='Désactiver l\'interface web')
    
    # Options de découverte et de clustering
    parser.add_argument('--discovery', action='store_true', default=True,
                        help='Activer la découverte automatique des serveurs (par défaut)')
    parser.add_argument('--no-discovery', action='store_false', dest='discovery',
                        help='Désactiver la découverte automatique des serveurs')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Activer l\'inférence distribuée')
    parser.add_argument('--auto-distribute', action='store_true', default=True,
                        help='Activer automatiquement l\'inférence distribuée pour les grands modèles (par défaut)')
    parser.add_argument('--no-auto-distribute', action='store_false', dest='auto_distribute',
                        help='Désactiver l\'inférence distribuée automatique pour les grands modèles')
    parser.add_argument('--rpc-servers', type=str, nargs='+', default=None,
                        help='Liste des adresses de serveurs RPC (format: host:port)')
    parser.add_argument('--interface', type=str, default=None,
                        help='Interface réseau préférée pour les connexions')
    
    # Options de verbosité
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Activer le mode verbeux')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Activer le mode debug (verbosité maximale)')
    
    return parser.parse_args()

def main():
    """Point d'entrée principal du proxy."""
    args = parse_args()
    
    run_proxy(
        host=args.host,
        port=args.port,
        server_addresses=args.servers,
        enable_distributed=args.distributed,
        auto_distribute_large=args.auto_distribute,
        rpc_servers=args.rpc_servers,
        enable_discovery=args.discovery,
        preferred_interface=args.interface,
        enable_ui=args.console,
        enable_web=args.web,
        verbose=args.verbose,
        debug=args.debug
    )

# Point d'entrée pour l'exécution directe
if __name__ == "__main__":
    main()