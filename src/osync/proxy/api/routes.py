"""
Module contenant les routes API pour le proxy Ollama.
Ce module redirige les routes API vers le service OllamaAPIService.
"""

import json
import logging
from typing import Any, Dict, Optional
from flask import Blueprint, request, jsonify, Response, stream_with_context
from .models import dict_to_generate_request, dict_to_chat_request, dict_to_embeddings_request
from dataclasses import asdict

# Configuration du logging
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception levée lorsqu'une validation des données échoue."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        

def register_api_routes(app, api_service):
    """
    Enregistre toutes les routes API avec l'application Flask.
    
    Args:
        app: L'application Flask
        api_service: Le service OllamaAPIService pour gérer les requêtes API
    """
    
    # Route API de génération
    @app.route('/api/v1/generate', methods=['POST'])
    def generate():
        """Générer du texte à partir d'un prompt"""
        try:
            data = request.json
            # Convertir les données JSON en objet GenerateRequest
            generate_request = dict_to_generate_request(data)
            
            # Si generate_request.stream est vrai, utiliser le streaming
            if generate_request.stream:
                def generate_stream():
                    try:
                        for chunk in api_service.generate_stream(generate_request):
                            yield f"data: {json.dumps(chunk)}\n\n"
                    except Exception as e:
                        logger.error(f"Erreur lors du streaming: {str(e)}")
                        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
                
                return Response(
                    generate_stream(),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache'}
                )
            else:
                # Génération standard (non-streaming)
                response = api_service.generate(generate_request)
                return jsonify(response)
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}")
            return jsonify({'error': f"Erreur de génération: {str(e)}"}), 500
    
    # Route API de chat
    @app.route('/api/v1/chat', methods=['POST'])
    def chat():
        """Échanger avec un modèle en format conversation"""
        try:
            data = request.json
            # Convertir les données JSON en objet ChatRequest
            chat_request = dict_to_chat_request(data)
            response = api_service.chat(chat_request)
            # Convertir l'objet ChatResponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors du chat: {str(e)}")
            return jsonify({'error': f"Erreur de chat: {str(e)}"}), 500
    
    # Route API d'embeddings
    @app.route('/api/v1/embeddings', methods=['POST'])
    def embeddings():
        """Générer des embeddings à partir d'un texte"""
        try:
            data = request.json
            # Convertir les données JSON en objet EmbeddingsRequest
            embeddings_request = dict_to_embeddings_request(data)
            response = api_service.embeddings(embeddings_request)
            # Convertir l'objet EmbeddingsResponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embeddings: {str(e)}")
            return jsonify({'error': f"Erreur d'embeddings: {str(e)}"}), 500
    
    # Route API pour lister les modèles
    @app.route('/api/v1/models', methods=['GET'])
    def list_models():
        """Lister tous les modèles disponibles"""
        try:
            models_response = api_service.list_models()
            # Convertir l'objet ModelsResponse en dictionnaire
            if hasattr(models_response, '__dict__') and not isinstance(models_response, dict):
                return jsonify(asdict(models_response))
            return jsonify(models_response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des modèles: {str(e)}")
            return jsonify({'error': f"Erreur de récupération des modèles: {str(e)}"}), 500
    
    # Route API pour obtenir le status du proxy
    @app.route('/api/v1/status', methods=['GET'])
    def status():
        """Obtenir le status actuel du proxy"""
        try:
            # Convertir l'objet StatusResponse en dictionnaire avant de le retourner
            status_response = api_service.get_status()
            return asdict(status_response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du status: {str(e)}")
            return {'error': f"Erreur de status: {str(e)}"}, 500
    
    # Route API pour lister les serveurs
    @app.route('/api/v1/servers', methods=['GET'])
    def list_servers():
        """Lister tous les serveurs du cluster"""
        try:
            servers_response = api_service.list_servers()
            # Convertir l'objet ServersResponse en dictionnaire
            if hasattr(servers_response, '__dict__') and not isinstance(servers_response, dict):
                return jsonify(asdict(servers_response))
            return jsonify(servers_response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des serveurs: {str(e)}")
            return jsonify({'error': f"Erreur de récupération des serveurs: {str(e)}"}), 500
    
    # Route API pour ajouter un serveur
    @app.route('/api/v1/servers', methods=['POST'])
    def add_server():
        """Ajouter un nouveau serveur au cluster"""
        try:
            data = request.json
            response = api_service.add_server(
                address=data.get('address'),
                verify_health=data.get('verify_health', True)
            )
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du serveur: {str(e)}")
            return jsonify({'error': f"Erreur d'ajout de serveur: {str(e)}"}), 500
    
    # Route API pour supprimer un serveur
    @app.route('/api/v1/servers/<server>', methods=['DELETE'])
    def remove_server(server):
        """Supprimer un serveur du cluster"""
        try:
            response = api_service.remove_server(server)
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du serveur: {str(e)}")
            return jsonify({'error': f"Erreur de suppression de serveur: {str(e)}"}), 500
    
    # Route API pour obtenir les détails d'un serveur
    @app.route('/api/v1/servers/<server>', methods=['GET'])
    def get_server(server):
        """Obtenir les détails d'un serveur"""
        try:
            response = api_service.get_server_details(server)
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des détails du serveur: {str(e)}")
            return jsonify({'error': f"Erreur de récupération des détails: {str(e)}"}), 500
    
    # Route API pour vérifier la santé d'un serveur
    @app.route('/api/v1/servers/<server>/check_health', methods=['POST'])
    def check_server_health(server):
        """Vérifier la santé d'un serveur"""
        try:
            response = api_service.check_server_health(server)
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de santé du serveur: {str(e)}")
            return jsonify({'error': f"Erreur de vérification de santé: {str(e)}"}), 500
    
    # Route API pour obtenir le rapport de santé global
    @app.route('/api/v1/health', methods=['GET'])
    def get_health():
        """Obtenir le rapport de santé du cluster"""
        try:
            response = api_service.get_health_report()
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du rapport de santé: {str(e)}")
            return jsonify({'error': f"Erreur de rapport de santé: {str(e)}"}), 500
    
    # Route API pour obtenir les statistiques de santé
    @app.route('/api/v1/health/stats', methods=['GET'])
    def get_health_stats():
        """Obtenir des statistiques sur la santé du cluster"""
        try:
            hours = request.args.get('hours', 24, type=int)
            response = api_service.get_health_stats(hours=hours)
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques de santé: {str(e)}")
            return jsonify({'error': f"Erreur de statistiques de santé: {str(e)}"}), 500
    
    # Route API pour obtenir l'historique de santé d'un serveur
    @app.route('/api/v1/health/server/<server>', methods=['GET'])
    def get_server_health_history(server):
        """Obtenir l'historique de santé d'un serveur"""
        try:
            hours = request.args.get('hours', 24, type=int)
            response = api_service.get_server_health_history(server=server, hours=hours)
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique de santé: {str(e)}")
            return jsonify({'error': f"Erreur d'historique de santé: {str(e)}"}), 500
    
    # Routes API pour les graphiques
    @app.route('/api/v1/chart/load', methods=['GET'])
    def get_load_chart_data():
        """Obtenir les données pour le graphique de charge des serveurs"""
        try:
            hours = request.args.get('hours', 1, type=int)
            response = api_service.get_load_chart_data(hours=hours)
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données du graphique: {str(e)}")
            return jsonify({'error': f"Erreur de données de graphique: {str(e)}"}), 500
    
    @app.route('/api/v1/chart/health', methods=['GET'])
    def get_health_chart_data():
        """Obtenir les données pour le graphique de santé des serveurs"""
        try:
            hours = request.args.get('hours', 1, type=int)
            response = api_service.get_health_chart_data(hours=hours)
            # Convertir la réponse en dictionnaire si nécessaire
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return jsonify(asdict(response))
            return jsonify(response)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données du graphique: {str(e)}")
            return jsonify({'error': f"Erreur de données de graphique: {str(e)}"}), 500
    
    logger.info("Routes API enregistrées avec succès")
    return app