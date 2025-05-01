"""
Routes API pour le proxy Ollama.

Ce module définit les routes HTTP pour interagir avec les serveurs Ollama.
"""

import json
import logging
import os
from typing import Dict, Any

from flask import Blueprint, request, jsonify, Response, stream_with_context, Flask, current_app

from .models import (
    GenerateRequest, GenerateResponse, 
    ChatRequest, ChatResponse, ChatMessage,
    EmbeddingsRequest, EmbeddingsResponse,
    ModelInfo, ServerInfo, StatusResponse, ModelsResponse, ServersResponse,
    dict_to_generate_request, dict_to_chat_request, dict_to_embeddings_request
)
from .services import OllamaProxyService

# Configuration du logging
logger = logging.getLogger(__name__)

# Création du blueprint
api_bp = Blueprint('api', __name__)


def register_api_routes(app: Flask, ollamaproxy_service: OllamaProxyService):
    """
    Enregistre les routes API avec l'application Flask et attache le service proxy.
    
    Args:
        app: L'application Flask
        ollamaproxy_service: Le service proxy Ollama
    """
    # Attacher le service au contexte de l'application
    app.ollamaproxy_service = ollamaproxy_service
    
    # Enregistrer le blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    
    logger.info("Routes API enregistrées avec succès")


@api_bp.route('/v1/generate', methods=['POST'])
def generate():
    """
    Point de terminaison pour générer du texte.
    """
    service = current_app.ollamaproxy_service
    
    try:
        # Analyser la requête
        req_json = request.get_json() if request.is_json else {}
        model = req_json.get('model', '')
        
        logger.debug(f"Requête generate pour le modèle: {model}")
        
        # Créer un objet GenerateRequest en utilisant la fonction utilitaire
        req = dict_to_generate_request(req_json)
        
        # Si streaming est demandé
        if req.stream:
            def generate_stream():
                """Produit un flux de réponses."""
                try:
                    for chunk in service.generate_stream(req):
                        yield json.dumps(chunk) + '\n'
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du stream: {e}")
                    yield json.dumps({"error": str(e), "done": True}) + '\n'
            
            # Retourner la réponse en streaming
            return Response(generate_stream(), content_type='application/x-ndjson')
        else:
            # Obtenir une réponse non-streaming
            resp = service.generate(req)
            return jsonify(resp)
            
    except Exception as e:
        logger.error(f"Erreur dans la route generate: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/v1/chat', methods=['POST'])
def chat():
    """
    Point de terminaison pour le chat.
    """
    try:
        service = current_app.ollamaproxy_service
        
        # Analyser la requête
        req_json = request.get_json() if request.is_json else {}
        model = req_json.get('model', '')
        
        logger.debug(f"Requête chat pour le modèle: {model}")
        
        # Créer un objet ChatRequest en utilisant la fonction utilitaire
        req = dict_to_chat_request(req_json)
        
        # Si streaming est demandé
        if req.stream:
            def generate_stream():
                """Produit un flux de réponses."""
                try:
                    for chunk in service.chat_stream(req):
                        yield json.dumps(chunk) + '\n'
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du stream: {e}")
                    yield json.dumps({"error": str(e), "done": True}) + '\n'
            
            # Retourner la réponse en streaming
            return Response(generate_stream(), content_type='application/x-ndjson')
        else:
            # Obtenir une réponse non-streaming
            resp = service.chat(req)
            return jsonify(resp)
            
    except Exception as e:
        logger.error(f"Erreur dans la route chat: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """
    Point de terminaison pour obtenir des embeddings.
    """
    try:
        service = current_app.ollamaproxy_service
        
        # Analyser la requête
        req_json = request.get_json() if request.is_json else {}
        model = req_json.get('model', '')
        
        logger.debug(f"Requête embeddings pour le modèle: {model}")
        
        # Créer un objet EmbeddingsRequest en utilisant la fonction utilitaire
        req = dict_to_embeddings_request(req_json)
        
        # Obtenir une réponse
        resp_dict = service.embeddings(req)
        
        # Retourner la réponse
        return jsonify(resp_dict)
    except Exception as e:
        logger.error(f"Erreur dans la route embeddings: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/v1/models', methods=['GET'])
def models():
    """
    Point de terminaison pour lister les modèles disponibles.
    """
    try:
        service = current_app.ollamaproxy_service
        
        logger.debug("Requête pour lister les modèles")
        
        # Obtenir les modèles
        resp = service.models()
        
        # Retourner la réponse
        return jsonify(resp)
    except Exception as e:
        logger.error(f"Erreur dans la route models: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/v1/status', methods=['GET'])
def status():
    """
    Point de terminaison pour vérifier le statut du proxy.
    """
    try:
        service = current_app.ollamaproxy_service
        
        logger.debug("Requête pour le statut")
        
        # Obtenir le statut en appelant get_status() au lieu de status()
        resp = service.get_status()
        
        # Retourner la réponse
        return jsonify(resp)
    except Exception as e:
        logger.error(f"Erreur dans la route status: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/v1/servers', methods=['GET'])
def servers():
    """
    Point de terminaison pour lister les serveurs disponibles.
    """
    try:
        service = current_app.ollamaproxy_service
        
        logger.debug("Requête pour lister les serveurs")
        
        # Obtenir les serveurs
        resp = service.servers()
        
        # Retourner la réponse
        return jsonify(resp)
    except Exception as e:
        logger.error(f"Erreur dans la route servers: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/v1/stats/history', methods=['GET'])
def stats_history():
    """
    Point de terminaison pour obtenir l'historique des statistiques de requêtes.
    
    Args:
        period: La période d'agrégation ('hourly', 'daily', 'weekly')
        days: Le nombre de jours d'historique à récupérer
    
    Returns:
        Un JSON contenant l'historique des statistiques
    """
    service = current_app.ollamaproxy_service
    
    period = request.args.get('period', 'daily')
    days = int(request.args.get('days', 7))
    
    logger.debug(f"Requête pour l'historique des statistiques: période={period}, jours={days}")
    
    # Obtenir l'historique des statistiques
    resp = service.get_request_stats_history(period=period, days=days)
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/stats/aggregate', methods=['GET'])
def stats_aggregate():
    """
    Point de terminaison pour obtenir des statistiques agrégées.
    
    Args:
        period: La période d'agrégation ('hourly', 'daily', 'weekly')
        days: Le nombre de jours d'historique à récupérer
    
    Returns:
        Un JSON contenant les statistiques agrégées
    """
    service = current_app.ollamaproxy_service
    
    period = request.args.get('period', 'daily')
    days = int(request.args.get('days', 30))
    
    logger.debug(f"Requête pour les statistiques agrégées: période={period}, jours={days}")
    
    # Obtenir les statistiques agrégées
    resp = service.get_aggregated_stats(period=period, days=days)
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/models/history', methods=['GET'])
def models_history():
    """
    Point de terminaison pour obtenir l'historique d'utilisation des modèles.
    
    Args:
        days: Le nombre de jours d'historique à récupérer
    
    Returns:
        Un JSON contenant l'historique d'utilisation des modèles
    """
    service = current_app.ollamaproxy_service
    
    days = int(request.args.get('days', 30))
    
    logger.debug(f"Requête pour l'historique des modèles: jours={days}")
    
    # Obtenir l'historique d'utilisation des modèles
    resp = service.get_models_history(days=days)
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/models/<model_name>/stats', methods=['GET'])
def model_stats(model_name):
    """
    Point de terminaison pour obtenir des statistiques détaillées pour un modèle spécifique.
    
    Args:
        model_name: Le nom du modèle
        days: Le nombre de jours d'historique à récupérer
    
    Returns:
        Un JSON contenant les statistiques du modèle
    """
    service = current_app.ollamaproxy_service
    
    days = int(request.args.get('days', 30))
    
    logger.debug(f"Requête pour les statistiques du modèle {model_name}: jours={days}")
    
    # Obtenir les statistiques du modèle
    resp = service.get_model_stats(model_name=model_name, days=days)
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/health/history', methods=['GET'])
def health_history():
    """
    Point de terminaison pour obtenir l'historique de santé du cluster.
    
    Args:
        days: Le nombre de jours d'historique à récupérer
        period: La période d'agrégation ('hourly', 'daily', 'weekly')
    
    Returns:
        Un JSON contenant l'historique de santé
    """
    service = current_app.ollamaproxy_service
    
    days = int(request.args.get('days', 30))
    period = request.args.get('period', 'daily')
    
    logger.debug(f"Requête pour l'historique de santé: période={period}, jours={days}")
    
    # Obtenir l'historique de santé
    resp = service.get_health_history_long_term(days=days, period=period)
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/servers/stats', methods=['GET'])
def servers_stats():
    """
    Point de terminaison pour obtenir des statistiques sur tous les serveurs.
    
    Args:
        days: Le nombre de jours d'historique à récupérer
        period: La période d'agrégation ('hourly', 'daily', 'weekly')
    
    Returns:
        Un JSON contenant les statistiques des serveurs
    """
    service = current_app.ollamaproxy_service
    
    days = int(request.args.get('days', 7))
    period = request.args.get('period', 'daily')
    
    logger.debug(f"Requête pour les statistiques des serveurs: période={period}, jours={days}")
    
    # Obtenir les statistiques des serveurs
    resp = service.get_servers_stats(days=days, period=period)
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/admin/cleanup', methods=['POST'])
def cleanup_data():
    """
    Point de terminaison pour nettoyer les anciennes données de la base de données.
    
    Args:
        days_to_keep: Le nombre de jours de données à conserver
    
    Returns:
        Un JSON contenant le résultat du nettoyage
    """
    service = current_app.ollamaproxy_service
    
    req_json = request.get_json() if request.is_json else {}
    days_to_keep = req_json.get('days_to_keep', 90)
    
    logger.debug(f"Requête pour nettoyer les données: jours_à_conserver={days_to_keep}")
    
    # Nettoyer les données
    resp = service.cleanup_old_data(days_to_keep=days_to_keep)
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/admin/aggregate', methods=['POST'])
def force_aggregation():
    """
    Point de terminaison pour forcer l'agrégation des statistiques.
    
    Args:
        period: La période d'agrégation ('hourly', 'daily')
    
    Returns:
        Un JSON contenant le résultat de l'agrégation
    """
    service = current_app.ollamaproxy_service
    
    req_json = request.get_json() if request.is_json else {}
    period = req_json.get('period', 'hourly')
    
    logger.debug(f"Requête pour forcer l'agrégation: période={period}")
    
    # Forcer l'agrégation
    resp = service.force_aggregation(period=period)
    
    # Retourner la réponse
    return jsonify(resp)