"""Health checking functionality for Ollama proxy server."""

import logging
import time
from typing import Dict, List, Optional

from olol.sync.client import OllamaClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global settings
health_check_interval = 30  # seconds

def health_checker() -> None:
    """Background thread to check server health periodically."""
    # Import here to avoid circular imports
    import olol.proxy.app as proxy_app
    cluster = proxy_app.cluster
    
    if cluster is None:
        logger.error("Cluster not initialized for health checker")
        return
        
    logger.info("Health checker started")
    
    # Map pour stocker les modèles sur chaque serveur
    server_model_map = {}
    
    while True:
        try:
            # Faire une copie des adresses de serveurs pour éviter les modifications pendant l'itération
            with cluster.server_lock:
                servers = list(cluster.server_addresses)
                
            for server in servers:
                # Ignorer les adresses invalides
                if not isinstance(server, str) or not server:
                    logger.warning(f"Skipping invalid server address: {server}")
                    continue
                
                # Pour l'affichage dans les logs
                short_server = server.split(":")[-1]
                
                client = None
                try:
                    # Format host:port simple
                    if server.count(':') == 1:
                        host, port_str = server.split(':')
                        port = int(port_str)
                        client = OllamaClient(host=host, port=port)
                        
                        # Vérifier l'état de santé
                        is_healthy = client.check_health()
                        
                        # Mise à jour de l'état de santé
                        with cluster.health_lock:
                            cluster.mark_server_health(server, is_healthy)
                            
                        # Si serveur en bonne santé, récupérer les modèles
                        if is_healthy:
                            # Récupérer les modèles
                            try:
                                models_response = client.list_models()
                                models = []
                                
                                # Extraire les noms des modèles
                                if hasattr(models_response, 'models'):
                                    models = [model.name for model in models_response.models]
                                    logger.info(f"Server {short_server} has {len(models)} models: {', '.join(models) if models else 'none'}")
                                    
                                    # Update the cluster with this server's models
                                    # Add server -> models mapping 
                                    server_model_map[server] = models
                                    
                                    # Update model -> server mapping in cluster
                                    with cluster.model_lock:
                                        for model_name in models:
                                            if model_name not in cluster.model_server_map:
                                                cluster.model_server_map[model_name] = []
                                            
                                            if server not in cluster.model_server_map[model_name]:
                                                cluster.model_server_map[model_name].append(server)
                                                
                                    # Créer un dictionnaire de détails pour les modèles
                                    model_details = {}
                                    
                                    # Pour les 3 premiers modèles, récupérer des détails supplémentaires
                                    for model_name in models[:3]:  # Limite pour éviter trop d'appels API
                                        try:
                                            # Créer un dictionnaire pour ce modèle
                                            model_details[model_name] = {
                                                "parameters": "Not detected"
                                            }
                                            
                                            # Pour des détails supplémentaires, on pourrait appeler d'autres méthodes ici
                                            # Mais pour l'instant, on se contente des noms de modèles
                                        except Exception as e:
                                            logger.debug(f"Error getting details for model {model_name}: {e}")
                                    
                                    # Mise à jour de la disponibilité des modèles
                                    cluster.model_manager.update_server_models(server, models)
                                else:
                                    logger.warning(f"Server {short_server} returned invalid model list")
                            except Exception as model_err:
                                logger.warning(f"Failed to list models on server {short_server}: {model_err}")
                    else:
                        # Format IPv6 non géré pour cette fonction simplifiée
                        logger.debug(f"Skipping IPv6 address: {server}")
                except Exception as e:
                    logger.error(f"Health check failed for server {short_server}: {e}")
                    with cluster.health_lock:
                        cluster.mark_server_health(server, False)
                finally:
                    if client:
                        client.close()
            
            # Pause entre les vérifications
            time.sleep(health_check_interval)
        except Exception as e:
            logger.error(f"Error in health checker: {e}")
            time.sleep(5)  # Wait a bit and try again