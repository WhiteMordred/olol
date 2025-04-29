"""
Routes pour l'API du proxy Ollama.

Ce module définit les points de terminaison RESTful pour interagir avec le cluster de serveurs Ollama.
"""

import json
import logging
from flask import Blueprint, request, Response, stream_with_context, jsonify

from olol.proxy.stats import update_request_stats
from .models import (
    GenerateRequest, ChatRequest, EmbeddingsRequest,
    ValidationError
)
from .services import OllamaProxyService

# Configuration du logging
logger = logging.getLogger(__name__)

# Création du blueprint
api_bp = Blueprint('api', __name__)

# Service partagé
ollama_service = None

def init_routes(cluster_manager):
    """
    Initialise les routes avec un gestionnaire de cluster.
    
    Args:
        cluster_manager: Le gestionnaire de cluster Ollama
    """
    global ollama_service
    ollama_service = OllamaProxyService(cluster_manager)


@api_bp.route('/api/generate', methods=['POST'])
def generate():
    """
    Point de terminaison pour la génération de texte.
    Implémente la même API que Ollama pour une compatibilité transparente.
    """
    # Mise à jour des statistiques
    update_request_stats()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Requête JSON invalide"}), 400
            
        # Parsing de la requête
        try:
            generate_request = GenerateRequest.from_dict(data)
        except ValidationError as e:
            return jsonify({"error": str(e)}), 400
        
        # Traitement de la requête en fonction du mode (streaming ou non)
        if generate_request.stream:
            # Mode streaming
            def generate_stream():
                for chunk in ollama_service.generate_stream(generate_request):
                    yield f"data: {json.dumps(chunk)}\n\n"
                
            response = Response(
                stream_with_context(generate_stream()),
                mimetype='text/event-stream'
            )
            response.headers['X-Accel-Buffering'] = 'no'
            response.headers['Cache-Control'] = 'no-cache'
            return response
        else:
            # Mode non-streaming
            result = ollama_service.generate(generate_request)
            if "error" in result:
                if result.get("error").startswith("Aucun serveur disponible"):
                    return jsonify(result), 503  # Service indisponible
                return jsonify(result), 400  # Bad request
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/chat', methods=['POST'])
def chat():
    """
    Point de terminaison pour les conversations de chat.
    Implémente la même API que Ollama pour une compatibilité transparente.
    """
    # Mise à jour des statistiques
    update_request_stats()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Requête JSON invalide"}), 400
            
        # Parsing de la requête
        try:
            chat_request = ChatRequest.from_dict(data)
        except ValidationError as e:
            return jsonify({"error": str(e)}), 400
        
        # Traitement de la requête en fonction du mode (streaming ou non)
        if chat_request.stream:
            # Mode streaming
            def chat_stream():
                for chunk in ollama_service.chat_stream(chat_request):
                    yield f"data: {json.dumps(chunk)}\n\n"
                
            response = Response(
                stream_with_context(chat_stream()),
                mimetype='text/event-stream'
            )
            response.headers['X-Accel-Buffering'] = 'no'
            response.headers['Cache-Control'] = 'no-cache'
            return response
        else:
            # Mode non-streaming
            result = ollama_service.chat(chat_request)
            if "error" in result:
                if result.get("error").startswith("Aucun serveur disponible"):
                    return jsonify(result), 503  # Service indisponible
                return jsonify(result), 400  # Bad request
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Erreur lors du chat: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/embeddings', methods=['POST'])
def embeddings():
    """
    Point de terminaison pour la génération d'embeddings.
    Implémente la même API que Ollama pour une compatibilité transparente.
    """
    # Mise à jour des statistiques
    update_request_stats()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Requête JSON invalide"}), 400
            
        # Parsing de la requête
        try:
            embeddings_request = EmbeddingsRequest.from_dict(data)
        except ValidationError as e:
            return jsonify({"error": str(e)}), 400
        
        # Exécution de la requête
        result = ollama_service.embeddings(embeddings_request)
        if "error" in result:
            if result.get("error").startswith("Aucun serveur disponible"):
                return jsonify(result), 503  # Service indisponible
            return jsonify(result), 400  # Bad request
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération des embeddings: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/status', methods=['GET'])
def status():
    """
    Point de terminaison pour obtenir le statut du proxy.
    """
    try:
        result = ollama_service.get_status()
        return jsonify(result.to_dict())
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/models', methods=['GET'])
def list_models():
    """
    Point de terminaison pour lister tous les modèles disponibles.
    Compatible avec l'API Ollama mais avec des informations supplémentaires sur les serveurs.
    """
    try:
        result = ollama_service.list_models()
        return jsonify(result.to_dict())
    except Exception as e:
        logger.error(f"Erreur lors du listage des modèles: {e}")
        return jsonify({"error": str(e)}), 500
        

@api_bp.route('/api/servers', methods=['GET'])
def list_servers():
    """
    Point de terminaison pour lister tous les serveurs du cluster.
    C'est une API spécifique au proxy qui n'existe pas dans l'API Ollama originale.
    """
    try:
        result = ollama_service.list_servers()
        return jsonify(result.to_dict())
    except Exception as e:
        logger.error(f"Erreur lors du listage des serveurs: {e}")
        return jsonify({"error": str(e)}), 500