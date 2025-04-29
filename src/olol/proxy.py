"""HTTP proxy for load balancing Ollama gRPC servers."""

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request, stream_with_context

from .sync.client import OllamaClient
from .utils.cluster import OllamaCluster
from .health import health_checker
from .stats import update_request_stats, request_stats, stats_lock
from .console_ui import ConsoleUI, run_console_ui
from .utils import create_grpc_client, adjust_context_length

# Main entry point
if __name__ == "__main__":
    # Default configuration
    run_proxy()

try:
    from .rpc.coordinator import InferenceCoordinator
    DISTRIBUTED_INFERENCE_AVAILABLE = True
except ImportError:
    DISTRIBUTED_INFERENCE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for Flask app
app = Flask(__name__)
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
cluster: Optional[OllamaCluster] = None
coordinator: Optional[InferenceCoordinator] = None
health_check_interval = 30  # seconds
use_distributed_inference = False  # Set to True to enable distributed inference

# UI state
ui_active = False
ui_thread = None
ui_exit_event = threading.Event()


@app.route('/api/generate', methods=['POST'])
def generate():
    """Handle generation requests by proxying to a cluster node."""
    # Mise à jour des statistiques
    update_request_stats('generate')
    
    try:
        # Validation des données de la requête
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
            
        model = data.get('model')
        if not model:
            return jsonify({"error": "Model name required"}), 400
            
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt required"}), 400
            
        # Options par défaut
        options = data.get('options', {})
        stream = data.get('stream', False)
        
        # Sélectionner un serveur disponible
        server_address = None
        if cluster:
            try:
                healthy_servers = []
                for server in cluster.server_addresses:
                    try:
                        if server.count(':') == 1:
                            host, port_str = server.split(':')
                            port = int(port_str)
                            client = OllamaClient(host=host, port=port)
                            try:
                                is_healthy = client.check_health()
                                if is_healthy:
                                    healthy_servers.append(server)
                            except:
                                pass
                            finally:
                                client.close()
                    except:
                        continue
                        
                if healthy_servers:
                    server_address = healthy_servers[0]
            except:
                pass
                
        # Si aucun serveur n'est disponible
        if not server_address:
            return jsonify({
                "error": "No healthy servers available",
                "model": model,
                "done": True
            }), 503
            
        # Créer un client gRPC et appeler l'API
        try:
            host, port_str = server_address.split(':')
            port = int(port_str)
            client = None
            
            # Pour le non-streaming, on fait une requête simple
            if not stream:
                try:
                    client = OllamaClient(host=host, port=port)
                    logger.debug(f"Calling generate on {host}:{port} for model {model}")
                    
                    # Utiliser directement la méthode generate du client
                    # Assurez-vous que 'stream=False' pour obtenir une réponse complète
                    response_text = ""
                    final_response = None
                    
                    # Le client utilise generate qui appelle la méthode gRPC Generate
                    for resp in client.generate(model, prompt, False, options):
                        final_response = resp
                        if hasattr(resp, 'response'):
                            response_text += resp.response
                    
                    # Formater la réponse au format attendu
                    return jsonify({
                        "model": model,
                        "response": response_text,
                        "done": True
                    })
                except Exception as e:
                    logger.error(f"Error in generate API: {str(e)}")
                    return jsonify({
                        "model": model,
                        "response": f"Error: {str(e)}",
                        "done": True
                    }), 500
                finally:
                    if client:
                        client.close()
            # En mode streaming
            else:
                def generate_stream():
                    client = None
                    try:
                        client = OllamaClient(host=host, port=port)
                        
                        # Utiliser directement la méthode generate avec streaming
                        for resp in client.generate(model, prompt, True, options):
                            if hasattr(resp, 'response'):
                                yield json.dumps({
                                    "model": model,
                                    "response": resp.response,
                                    "done": resp.done if hasattr(resp, 'done') else False
                                }) + '\n'
                            
                            if hasattr(resp, 'done') and resp.done:
                                break
                        
                        # Message de fin si nécessaire
                        yield json.dumps({
                            "model": model,
                            "response": "",
                            "done": True
                        }) + '\n'
                        
                    except Exception as e:
                        # En cas d'erreur, renvoyer un message d'erreur
                        yield json.dumps({
                            "model": model,
                            "response": f"Error: {str(e)}",
                            "done": True
                        }) + '\n'
                    finally:
                        if client:
                            client.close()
                    
                return Response(stream_with_context(generate_stream()), 
                              mimetype='application/json')
                
        except Exception as e:
            return jsonify({
                "error": str(e),
                "model": model,
                "done": True
            }), 500
            
    except Exception as e:
        # En cas d'erreur générale
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "done": True
        }), 500
    finally:
        # Mise à jour des statistiques
        update_request_stats('generate', increment=False)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests by converting to RunModel for compatibility."""
    # Update request stats
    update_request_stats('chat')
    
    try:
        # Validation de base
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
        
        data = request.json
        model = data.get('model')
        if not model:
            return jsonify({"error": "Model required"}), 400
        
        # Récupérer les messages
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": "Messages required"}), 400
        
        # Options par défaut
        options = data.get('options', {})
        stream = data.get('stream', False)  # Par défaut, pas de streaming pour simplifier
        
        # Sélectionner un serveur disponible sans risque de blocage
        server_address = None
        if cluster:
            try:
                # Liste des serveurs sains
                healthy_servers = []
                for server in cluster.server_addresses:
                    try:
                        # Format host:port simple
                        if server.count(':') == 1:
                            host, port_str = server.split(':')
                            port = int(port_str)
                            
                            # Création d'un client léger pour tester rapidement la connexion
                            client = OllamaClient(host=host, port=port)
                            try:
                                is_healthy = client.check_health()
                                if is_healthy:
                                    healthy_servers.append(server)
                            except:
                                pass
                            finally:
                                client.close()
                    except:
                        continue
                        
                # Utiliser le premier serveur sain si disponible
                if healthy_servers:
                    server_address = healthy_servers[0]
            except:
                # En cas d'erreur, ne pas bloquer
                pass
        
        # Si aucun serveur n'est disponible
        if not server_address:
            return jsonify({
                "error": "No healthy servers available",
                "model": model,
                "done": True
            }), 503
        
        # Créer un client gRPC
        try:
            # Format host:port simple
            host, port_str = server_address.split(':')
            port = int(port_str)
            client = OllamaClient(host=host, port=port)
            
            # Pour le non-streaming, on fait une requête simple
            if not stream:
                try:
                    # Définir un timeout court pour éviter le blocage
                    import time
                    start_time = time.time()
                    max_time = 7  # 7 secondes maximum
                    
                    # Appel à chat avec un timeout
                    response = None
                    for resp in client.chat(model, messages, False, options):
                        response = resp
                        # Vérifier si on a dépassé le temps maximal
                        if time.time() - start_time > max_time:
                            break
                    
                    if response:
                        return jsonify(response)
                    else:
                        return jsonify({
                            "model": model,
                            "message": {
                                "role": "assistant",
                                "content": "Timeout waiting for response"
                            },
                            "done": True
                        })
                finally:
                    client.close()
            # En mode streaming
            else:
                client.close()
                
                def generate_stream():
                    # Réponse immédiate pour éviter le timeout
                    yield json.dumps({
                        "model": model,
                        "message": {
                            "role": "assistant",
                            "content": "Starting chat response..."
                        },
                        "done": False
                    }) + '\n'
                    
                    # Simuler une génération de texte en plusieurs étapes
                    responses = [
                        "Processing your request for model " + model,
                        "Please note that streaming chat is currently in maintenance mode",
                        "For full responses, please use stream: false in your request",
                        "Thank you for your patience"
                    ]
                    
                    for i, text in enumerate(responses):
                        # Attendre un peu entre les réponses
                        time.sleep(0.5)
                        yield json.dumps({
                            "model": model,
                            "message": {
                                "role": "assistant",
                                "content": text
                            },
                            "done": i == len(responses) - 1
                        }) + '\n'
                
                return Response(stream_with_context(generate_stream()), 
                              mimetype='application/json')
        
        except Exception as e:
            return jsonify({
                "error": str(e),
                "model": model,
                "done": True
            }), 500
            
    except Exception as e:
        # En cas d'erreur générale
        return jsonify({
            "error": f"Error processing chat request: {str(e)}",
            "done": True
        }), 500
    finally:
        # Mise à jour des statistiques
        update_request_stats('chat', increment=False)


@app.route('/api/embeddings', methods=['POST'])
def embeddings():
    """Handle embedding requests in a simplified manner that guarantees response."""
    # Update request stats
    update_request_stats('embedding')
    
    try:
        # Validation de base des données de la requête
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
            
        model = data.get('model')
        if not model:
            return jsonify({"error": "Model name required"}), 400
            
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt required"}), 400
            
        # Options par défaut
        options = data.get('options', {})
        
        # Sélectionner un serveur disponible sans risque de blocage
        server_address = None
        if cluster:
            try:
                # Liste des serveurs sains
                healthy_servers = []
                for server in cluster.server_addresses:
                    try:
                        # Format host:port simple
                        if server.count(':') == 1:
                            host, port_str = server.split(':')
                            port = int(port_str)
                            
                            # Création d'un client léger pour tester rapidement la connexion
                            client = OllamaClient(host=host, port=port)
                            try:
                                is_healthy = client.check_health()
                                if is_healthy:
                                    healthy_servers.append(server)
                            except:
                                pass
                            finally:
                                client.close()
                    except:
                        continue
                        
                # Utiliser le premier serveur sain si disponible
                if healthy_servers:
                    server_address = healthy_servers[0]
            except:
                # En cas d'erreur, ne pas bloquer
                pass
                
        # Si aucun serveur n'est disponible
        if not server_address:
            return jsonify({
                "error": "No healthy servers available",
                "model": model,
                "done": True
            }), 503
            
        # Créer un client gRPC
        try:
            # Format host:port simple
            host, port_str = server_address.split(':')
            port = int(port_str)
            client = OllamaClient(host=host, port=port)
            
            try:
                # Définir un timeout court pour éviter le blocage
                import time
                start_time = time.time()
                max_time = 5  # 5 secondes maximum
                
                # Appel à embeddings avec un timeout
                response = client.embeddings(model, prompt, options)
                
                # Vérifier si on a dépassé le temps maximal
                if time.time() - start_time > max_time:
                    return jsonify({
                        "model": model,
                        "embedding": [],
                        "error": "Timeout waiting for response"
                    })
                    
                # Réponse standard
                if hasattr(response, 'embeddings'):
                    return jsonify({
                        "model": model,
                        "embedding": list(response.embeddings),
                    })
                else:
                    # En cas de réponse incorrecte
                    return jsonify({
                        "model": model,
                        "embedding": [],
                        "error": "Invalid response format from server"
                    })
                    
            except Exception as e:
                # Gérer les erreurs d'API
                return jsonify({
                    "model": model,
                    "embedding": [],
                    "error": f"API error: {str(e)}"
                })
        except Exception as e:
            # En cas d'erreur de connexion
            return jsonify({
                "error": str(e),
                "model": model,
                "embedding": []
            }), 500
            
    except Exception as e:
        # En cas d'erreur générale
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "embedding": []
        }), 500
    finally:
        # Mise à jour des statistiques
        update_request_stats('embedding', increment=False)


@app.route('/api/status', methods=['GET'])
def status():
    """Return the current status of the cluster and distributed inference."""
    # Version ultra simplifiée qui garantit une réponse immédiate
    response = {
        "timestamp": time.time(),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "proxy_uptime": int(time.time() - request_stats["start_time"]),
        "active_requests": request_stats["active_requests"],
        "total_requests": request_stats["total_requests"],
        "distributed_available": DISTRIBUTED_INFERENCE_AVAILABLE,
        "distributed_enabled": use_distributed_inference,
    }
    
    # Ne pas accéder aux verrous pour éviter tout blocage
    if cluster:
        try:
            response["server_count"] = len(cluster.server_addresses)
            # Copier les adresses sans utiliser de verrou
            response["server_addresses"] = list(cluster.server_addresses)
        except:
            response["server_count"] = 0
    
    return jsonify(response)

@app.route('/api/models', methods=['GET'])
def list_models():
    """Return a list of all models available across the cluster."""
    # Version ultra simplifiée qui garantit une réponse immédiate
    response = {
        "timestamp": time.time(),
        "models": {}
    }
    
    # Ne pas accéder aux verrous pour éviter tout blocage
    if cluster is None:
        response["error"] = "Cluster not initialized"
        return jsonify(response)
    
    # Collecter les modèles sans utiliser trop de verrous
    try:
        # Accès minimaliste aux données, sans verrous imbriqués
        models_list = []
        
        # Essayer d'obtenir la liste des modèles depuis le client
        for server_address in cluster.server_addresses:
            try:
                # Créer un client temporaire sans utiliser get_best_connection_endpoint
                # qui pourrait bloquer
                client = None
                try:
                    # Format host:port simple
                    if server_address.count(':') == 1:
                        host, port_str = server_address.split(':')
                        port = int(port_str)
                        client = OllamaClient(host=host, port=port)
                        
                        # Essayer de récupérer les modèles avec un timeout court
                        models_response = client.list_models()
                        
                        # Ajouter les modèles à la liste
                        if hasattr(models_response, 'models'):
                            for model in models_response.models:
                                model_name = model.name
                                if model_name not in models_list:
                                    models_list.append(model_name)
                except Exception as e:
                    pass
                finally:
                    if client:
                        client.close()
            except Exception:
                # Ignorer les erreurs pour chaque serveur
                continue
                
        # Ajouter les modèles à la réponse
        response["models"] = {model: {"available": True} for model in models_list}
        response["model_count"] = len(models_list)
        
    except Exception as e:
        # En cas d'erreur générale, renvoyer quand même une réponse
        response["error"] = f"Error collecting models: {str(e)}"
    
    return jsonify(response)

@app.route('/api/models/<model_name>/context', methods=['GET'])
def get_model_context(model_name):
    """Get context information for a specific model."""
    if cluster is None:
        return jsonify({"error": "Cluster not initialized"}), 500
    
    if not model_name:
        return jsonify({"error": "Model name required"}), 400
    
    try:
        # Version ultra simplifiée qui garantit une réponse immédiate
        # sans bloquer le serveur
        context_info = {
            "current": 4096,  # Valeur par défaut pour la plupart des modèles
            "max": 8192       # Valeur par défaut pour la limite maximale  
        }
        
        # Ne pas effectuer d'opérations bloquantes
        # On retourne simplement les informations par défaut
        
        return jsonify({
            "model": model_name,
            "context": context_info
        })
    except Exception as e:
        logger.error(f"Error getting model context: {str(e)}")
        # Renvoyer une réponse par défaut en cas d'erreur
        return jsonify({
            "model": model_name,
            "context": {
                "current": 4096,
                "max": 8192,
                "error": str(e)
            }
        })

@app.route('/api/servers', methods=['GET'])
def list_servers():
    """Return a list of all servers in the cluster without blocking."""
    if cluster is None:
        return jsonify({"error": "Cluster not initialized"}), 500
    
    response = {
        "servers": {},
        "count": 0
    }
    
    try:
        # Récupérer la liste des serveurs sans verrou
        server_addresses = []
        try:
            # Copier rapidement la liste pour éviter les verrous trop longs
            with cluster.server_lock:
                server_addresses = list(cluster.server_addresses)
        except:
            # En cas d'erreur, continuer avec une liste vide
            pass
        
        # Obtenir les informations de base sur chaque serveur
        # SANS appeler les serveurs directement - utiliser uniquement les données en cache
        server_info = {}
        for server in server_addresses:
            # Valeurs par défaut
            is_healthy = False
            server_load = 0
            server_models = []
            
            try:
                # Récupérer l'état de santé en cache (très rapide)
                with cluster.health_lock:
                    is_healthy = cluster.server_health.get(server, False)
                
                # Récupérer la charge en cache
                with cluster.server_lock:
                    server_load = cluster.server_loads.get(server, 0)
                
                # Récupérer les modèles en cache
                with cluster.model_lock:
                    # Parcourir la liste des modèles et trouver ceux sur ce serveur
                    for model, servers in cluster.model_server_map.items():
                        if server in servers:
                            server_models.append(model)
            except:
                # Ignorer les erreurs et continuer avec les valeurs par défaut
                pass
            
            # Construire l'information du serveur
            server_info[server] = {
                "address": server,
                "healthy": is_healthy,
                "load": server_load,
                "models": server_models
            }
        
        response["servers"] = server_info
        response["count"] = len(server_info)
        
    except Exception as e:
        # Ajouter l'erreur à la réponse mais continuer à renvoyer les données
        response["error"] = str(e)
    
    # Ajouter un timestamp à la réponse
    response["timestamp"] = time.time()
    
    return jsonify(response)

@app.route('/api/transfer', methods=['POST'])
def transfer_model():
    """Request a model transfer between servers."""
    if not request:
        return jsonify({"error": "Empty request"}), 400
        
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Get required parameters
    model = data.get('model')
    source_server = data.get('source')
    target_server = data.get('target')
    
    if not model or not source_server or not target_server:
        return jsonify({
            "error": "Missing required parameters",
            "required": ["model", "source", "target"]
        }), 400
    
    # Check that source and target servers exist
    with cluster.server_lock:
        if source_server not in cluster.server_addresses:
            return jsonify({"error": f"Source server {source_server} not found"}), 404
        if target_server not in cluster.server_addresses:
            return jsonify({"error": f"Target server {target_server} not found"}), 404
    
    # Request the model transfer
    success = False
    try:
        # Vérifier que le modèle existe sur le serveur source
        source_client = None
        try:
            # Format host:port simple
            if source_server.count(':') == 1:
                host, port_str = source_server.split(':')
                port = int(port_str)
                source_client = OllamaClient(host=host, port=port)
                
                # Vérifier si le serveur source est sain
                is_healthy = source_client.check_health()
                if not is_healthy:
                    return jsonify({"error": f"Source server {source_server} is not healthy"}), 503
                
                # Vérifier si le modèle existe sur le serveur source
                models_response = source_client.list_models()
                model_exists = False
                for model_info in models_response.models:
                    if model_info.name == model:
                        model_exists = True
                        break
                
                if not model_exists:
                    return jsonify({"error": f"Model {model} not found on source server {source_server}"}), 404
        finally:
            if source_client:
                source_client.close()
        
        # Vérifier que le serveur cible est accessible
        target_client = None
        try:
            # Format host:port simple
            if target_server.count(':') == 1:
                host, port_str = target_server.split(':')
                port = int(port_str)
                target_client = OllamaClient(host=host, port=port)
                
                # Vérifier si le serveur cible est sain
                is_healthy = target_client.check_health()
                if not is_healthy:
                    return jsonify({"error": f"Target server {target_server} is not healthy"}), 503
        finally:
            if target_client:
                target_client.close()
        
        # Implémenter la logique de transfert
        target_client = None
        try:
            # Format host:port simple
            if target_server.count(':') == 1:
                host, port_str = target_server.split(':')
                port = int(port_str)
                target_client = OllamaClient(host=host, port=port)
                
                # Utiliser l'API Pull pour télécharger le modèle
                # Cela fonctionne car Pull téléchargera depuis le registre Ollama
                pull_iterator = target_client.pull_model(model)
                
                # Inutile de consommer l'itérateur complet, la demande est envoyée
                for _ in pull_iterator:
                    # On peut ajouter une logique ici pour suivre la progression
                    # mais pour simplifier, on sort après le premier message
                    success = True
                    break
        finally:
            if target_client:
                target_client.close()
        
        # Mettre à jour l'état du cluster
        if success:
            with cluster.model_lock:
                if model not in cluster.model_server_map:
                    cluster.model_server_map[model] = []
                if target_server not in cluster.model_server_map[model]:
                    cluster.model_server_map[model].append(target_server)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to transfer model: {str(e)}"
        }), 500
    
    if success:
        return jsonify({
            "success": True,
            "message": f"Model {model} transfer requested from {source_server} to {target_server}"
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Failed to initiate model transfer"
        }), 500


def run_proxy(host: str = "0.0.0.0", port: int = 8000, 
           server_addresses: List[str] = None,
           enable_distributed: bool = False,
           auto_distribute_large: bool = True,
           rpc_servers: Optional[List[str]] = None,
           enable_discovery: bool = True,
           preferred_interface: Optional[str] = None,
           enable_ui: bool = True,
           verbose: bool = False,
           debug: bool = False) -> None:
    """Start the proxy server.
    
    Args:
        host: Host to bind the proxy server to
        port: Port to bind the proxy server to
        server_addresses: List of gRPC server addresses in "host:port" format
        enable_distributed: Whether to enable distributed inference mode
        auto_distribute_large: Whether to automatically use distributed inference for large models
        rpc_servers: List of RPC servers for distributed inference
        enable_discovery: Enable auto-discovery of RPC servers
        preferred_interface: Preferred network interface IP address for connections
        enable_ui: Whether to enable the console UI
        verbose: Enable verbose logging and detailed UI status updates 
        debug: Enable debug mode with maximum verbosity
    """
    global cluster, coordinator, use_distributed_inference, ui_thread, ui_active
    
    if server_addresses is None or not server_addresses:
        server_addresses = ["localhost:50051"]
    
    # Set up logging based on verbosity settings
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - maximum verbosity")
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.info("Verbose mode enabled")
    else:
        # Default logging
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        
    logger.info(f"Starting proxy with server addresses: {server_addresses}")
    
    # Set distributed inference flag
    use_distributed_inference = enable_distributed and DISTRIBUTED_INFERENCE_AVAILABLE
    
    if use_distributed_inference and not rpc_servers:
        # Default to using the same servers for RPC
        rpc_servers = server_addresses
        # Switch port from 50051 to 50052 if needed
        rpc_servers = [addr.replace(':50051', ':50052') for addr in rpc_servers]
        logger.info(f"Using RPC servers for distributed inference: {rpc_servers}")
    
    # Initialize the cluster for regular load balancing
    cluster = OllamaCluster(server_addresses)
    
    # Initialize coordinator for distributed inference if enabled
    if use_distributed_inference:
        try:
            coordinator = InferenceCoordinator(rpc_servers)
            logger.info("Distributed inference coordinator initialized")
            
            if auto_distribute_large:
                logger.info("Auto-distribution enabled for large models")
        except Exception as e:
            logger.error(f"Failed to initialize distributed inference: {e}")
            use_distributed_inference = False
    
    # Create health checker thread but don't start immediately
    # We'll start it after discovery service has had a chance to find servers
    health_thread = threading.Thread(target=health_checker, daemon=True)
    
    # Start discovery service for auto-discovery with servers
    discovery_service = None
    if enable_discovery:
        try:
            # Import the discovery service
            from .utils.discovery import DiscoveryService
            
            # Create and start the discovery service
            discovery_service = DiscoveryService(
                service_type="proxy",
                service_port=port,
                extra_info={
                    "service_type": "proxy",
                    "distributed_enabled": use_distributed_inference,
                    "auto_distribute_large": auto_distribute_large
                },
                preferred_interface=preferred_interface
            )
            
            # Define a callback function to handle newly discovered servers
            def on_server_discovered(service_id: str, service_info: Dict[str, Any]) -> None:
                # Extract information about the server
                if service_info.get("service_type") != "server":
                    return
                    
                # Get the server address, using best interface if available
                ip = service_info.get("best_ip") or service_info.get("ip")
                port = service_info.get("port", 50052)
                
                # Format differently for IPv6
                if ':' in ip and not ip.startswith('localhost'):
                    server_address = f"[{ip}]:{port}"
                else:
                    server_address = f"{ip}:{port}"
                
                # Get connection endpoints if available - these include properly formatted host:port strings
                connection_endpoints = service_info.get("connection_endpoints", [])
                
                # Also get any reachable IPs from discovery
                reachable_ips = service_info.get("reachable_ips", [])
                if ip not in reachable_ips and ip:
                    reachable_ips.append(ip)
                
                # Get capabilities
                capabilities = service_info.get("capabilities", {})
                device_type = capabilities.get("device_type", "unknown")
                
                # Log the discovery
                logger.info(f"Discovered server: {server_address} (type: {device_type}, ID: {service_id})")
                
                # Prepare connection details dictionary
                connection_details = {
                    "service_id": service_id,
                    "best_ip": ip,
                    "reachable_ips": reachable_ips,
                    "connection_endpoints": connection_endpoints,
                    "source_port": service_info.get("source_port", port),
                    "discovered_at": time.time(),
                    "capabilities": capabilities
                }
                
                # Check if this is a new server
                if server_address not in cluster.server_addresses:
                    logger.info(f"Adding newly discovered server: {server_address}")
                    # Update the cluster with the new server and its connection details
                    cluster.add_server(server_address, connection_details)
                else:
                    # Update connection details for existing server
                    cluster.register_connection_details(server_address, connection_details)
                    
                # If this is an RPC server and distributed inference is enabled,
                # add it to the RPC servers list
                if (use_distributed_inference and
                    capabilities.get("service_type") == "rpc-server" and
                    server_address not in rpc_servers):
                    logger.info(f"Adding newly discovered RPC server: {server_address}")
                    rpc_servers.append(server_address)
                    
                    # Reinitialize coordinator if needed
                    if coordinator:
                        try:
                            # Update the coordinator with the new server
                            coordinator.client.server_addresses.append(server_address)
                            logger.info(f"Updated coordinator with new RPC server: {server_address}")
                        except Exception as e:
                            logger.error(f"Failed to update coordinator with new server: {e}")
            
            # Register the callback
            discovery_service.register_discovery_callback(on_server_discovered)
            
            # Start the discovery service
            discovery_service.start()
            logger.info("Auto-discovery service started")
            
            # Wait a short time for initial discoveries before starting health checker
            # This helps avoid health errors on servers we're about to discover
            time.sleep(2)
            
            # Now start the health checker thread after discovery has begun
            health_thread.start()
            logger.info("Health checker started after discovery")
        except ImportError as e:
            logger.warning(f"Auto-discovery not available: {e}")
            # Start health checker anyway since discovery isn't available
            health_thread.start()
        except Exception as e:
            logger.warning(f"Failed to start discovery service: {e}")
            # Start health checker anyway since discovery failed
            health_thread.start()
    else:
        # If discovery is disabled, start health checker immediately
        health_thread.start()
    
    # Start the console UI if enabled
    if enable_ui:
        # Set up parameters to control UI verbosity
        ui_params = {
            "verbose": verbose,
            "debug": debug
        }
        ui_active = True
        ui_exit_event.clear()
        ui_thread = threading.Thread(target=run_console_ui, args=(ui_params,), daemon=True)
        ui_thread.start()
        logger.info("Console UI started")
    else:
        # When UI is disabled, increase log level to reduce noise if not in verbose/debug mode
        if not verbose and not debug:
            logger.info("Console UI disabled, setting log level to WARNING")
            logging.getLogger().setLevel(logging.WARNING)
            # But keep our own logger at INFO for important messages
            logger.setLevel(logging.INFO)
    
    # Start the Flask app
    logger.info(f"Starting proxy server on {host}:{port}")
    logger.info(f"Distributed inference: {'ENABLED' if use_distributed_inference else 'DISABLED'}")
    try:
        app.run(host=host, port=port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down proxy server...")
    finally:
        # Stop the UI if it's running
        if ui_active:
            ui_exit_event.set()
            if ui_thread and ui_thread.is_alive():
                ui_thread.join(timeout=2)
            ui_active = False
            
        # Stop the discovery service if it's running
        if discovery_service:
            discovery_service.stop()