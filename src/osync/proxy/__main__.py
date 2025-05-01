"""Point d'entrée pour l'exécution du proxy en tant que module."""

import sys
import argparse
import logging
from .app import run_proxy

def main():
    """
    Point d'entrée principal pour le proxy Ollama.
    Parse les arguments de ligne de commande et démarre le proxy.
    """
    parser = argparse.ArgumentParser(description='Démarre le proxy Ollama')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Adresse de l\'hôte pour le serveur (défaut: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port pour le serveur (défaut: 8000)')
    parser.add_argument('--servers', type=str, nargs='+', default=['localhost:50051'],
                        help='Liste des adresses des serveurs Ollama (défaut: localhost:50051)')
    parser.add_argument('--distributed', action='store_true',
                        help='Activer le mode d\'inférence distribuée')
    parser.add_argument('--auto-distribute-large', action='store_true', default=True,
                        help='Distribuer automatiquement les modèles larges (défaut: activé)')
    parser.add_argument('--rpc-servers', type=str, nargs='+',
                        help='Liste des serveurs RPC pour l\'inférence distribuée')
    parser.add_argument('--enable-discovery', action='store_true', default=True,
                        help='Activer la découverte automatique des serveurs (défaut: activé)')
    parser.add_argument('--preferred-interface', type=str,
                        help='Interface réseau préférée pour les connexions')
    parser.add_argument('--enable-ui', action='store_true', default=True,
                        help='Activer l\'interface console (défaut: activé)')
    parser.add_argument('--enable-web', action='store_true', default=True,
                        help='Activer l\'interface web et l\'API Swagger (défaut: activé)')
    parser.add_argument('--verbose', action='store_true',
                        help='Activer la journalisation détaillée')
    parser.add_argument('--debug', action='store_true',
                        help='Activer le mode débogage avec journalisation maximale')

    args = parser.parse_args()

    # Configurer la journalisation
    log_level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Démarrage du proxy Ollama...")
    
    # Lancer le proxy avec les arguments parsés
    run_proxy(
        host=args.host,
        port=args.port,
        server_addresses=args.servers,
        enable_distributed=args.distributed,
        auto_distribute_large=args.auto_distribute_large,
        rpc_servers=args.rpc_servers,
        enable_discovery=args.enable_discovery,
        preferred_interface=args.preferred_interface,
        enable_ui=args.enable_ui,
        enable_web=args.enable_web,
        verbose=args.verbose,
        debug=args.debug
    )

if __name__ == "__main__":
    sys.exit(main())