"""
Routes API pour le proxy Ollama.

Ce module définit les routes HTTP pour interagir avec les serveurs Ollama.
"""

import json
import logging
import os
from typing import Dict, Any

from flask import Blueprint, request, jsonify, Response, stream_with_context

from .models import (
    GenerateRequest, GenerateResponse, 
    ChatRequest, ChatResponse, ChatMessage,
    EmbeddingsRequest, EmbeddingsResponse,
    ModelInfo, ServerInfo, StatusResponse, ModelsResponse, ServersResponse
)
from .services import OllamaProxyService

# Configuration du logging
logger = logging.getLogger(__name__)

# Création du blueprint
api_bp = Blueprint('api', __name__)


@api_bp.route('/v1/generate', methods=['POST'])
def generate():
    """
    Point de terminaison pour générer du texte.
    """
    service = request.app.ollamaproxy_service
    
    # Analyser la requête
    req_json = request.get_json() if request.is_json else {}
    model = req_json.get('model', '')
    
    logger.debug(f"Requête generate pour le modèle: {model}")
    
    # Créer un objet GenerateRequest
    req = GenerateRequest.from_dict(req_json)
    
    # Obtenir une réponse en streaming
    resp = service.generate(req)
    
    def generate_stream():
        """Produit un flux de réponses."""
        for chunk in resp:
            yield chunk
    
    # Retourner la réponse en streaming
    return Response(generate_stream(), content_type='application/x-ndjson')


@api_bp.route('/v1/chat', methods=['POST'])
def chat():
    """
    Point de terminaison pour le chat.
    """
    service = request.app.ollamaproxy_service
    
    # Analyser la requête
    req_json = request.get_json() if request.is_json else {}
    model = req_json.get('model', '')
    
    logger.debug(f"Requête chat pour le modèle: {model}")
    
    # Créer un objet ChatRequest
    req = ChatRequest.from_dict(req_json)
    
    # Obtenir une réponse en streaming
    resp = service.chat(req)
    
    def generate_stream():
        """Produit un flux de réponses."""
        for chunk in resp:
            yield chunk
    
    # Retourner la réponse en streaming
    return Response(generate_stream(), content_type='application/x-ndjson')


@api_bp.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """
    Point de terminaison pour obtenir des embeddings.
    """
    service = request.app.ollamaproxy_service
    
    # Analyser la requête
    req_json = request.get_json() if request.is_json else {}
    model = req_json.get('model', '')
    
    logger.debug(f"Requête embeddings pour le modèle: {model}")
    
    # Créer un objet EmbeddingsRequest
    req = EmbeddingsRequest.from_dict(req_json)
    
    # Obtenir une réponse
    resp_dict = service.embeddings(req)
    
    # Retourner la réponse
    return jsonify(resp_dict)


@api_bp.route('/v1/models', methods=['GET'])
def models():
    """
    Point de terminaison pour lister les modèles disponibles.
    """
    service = request.app.ollamaproxy_service
    
    logger.debug("Requête pour lister les modèles")
    
    # Obtenir les modèles
    resp = service.models()
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/status', methods=['GET'])
def status():
    """
    Point de terminaison pour vérifier le statut du proxy.
    """
    service = request.app.ollamaproxy_service
    
    logger.debug("Requête pour le statut")
    
    # Obtenir le statut
    resp = service.status()
    
    # Retourner la réponse
    return jsonify(resp)


@api_bp.route('/v1/servers', methods=['GET'])
def servers():
    """
    Point de terminaison pour lister les serveurs disponibles.
    """
    service = request.app.ollamaproxy_service
    
    logger.debug("Requête pour lister les serveurs")
    
    # Obtenir les serveurs
    resp = service.servers()
    
    # Retourner la réponse
    return jsonify(resp)

# Nouvelles routes pour les statistiques persistantes

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
    service = request.app.ollamaproxy_service
    
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
    service = request.app.ollamaproxy_service
    
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
    service = request.app.ollamaproxy_service
    
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
    service = request.app.ollamaproxy_service
    
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
    service = request.app.ollamaproxy_service
    
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
    service = request.app.ollamaproxy_service
    
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
    service = request.app.ollamaproxy_service
    
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
    service = request.app.ollamaproxy_service
    
    req_json = request.get_json() if request.is_json else {}
    period = req_json.get('period', 'hourly')
    
    logger.debug(f"Requête pour forcer l'agrégation: période={period}")
    
    # Forcer l'agrégation
    resp = service.force_aggregation(period=period)
    
    # Retourner la réponse
    return jsonify(resp)