"""
Documentation Swagger pour l'API du proxy Ollama.

Ce module crée une interface Swagger pour documenter l'API RESTful du proxy Ollama,
permettant aux utilisateurs de visualiser et de tester les endpoints disponibles.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_restx import Api, Resource, fields, Namespace
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from olol.proxy.api.services import OllamaProxyService
from olol.proxy.cluster.manager import ClusterManager
from olol.proxy.cluster.health import get_health_monitor
from .models import dict_to_generate_request, dict_to_chat_request, dict_to_embeddings_request

# Configuration du logging
logger = logging.getLogger(__name__)

# Création du blueprint pour Swagger
swagger_bp = Blueprint('swagger', __name__)

# Initialisation de l'API
api = Api(
    app=swagger_bp,
    version='1.0',
    title='OLOL - Proxy API pour Ollama',
    description='API de proxy pour interagir avec un cluster de serveurs Ollama',
    doc='/docs',
    validate=True
)

# Namespace pour les endpoints principaux
api_ns = api.namespace('api/v1', description='Opérations de l\'API Ollama')

# Service API Ollama
cluster_manager = ClusterManager()
ollama_service = OllamaProxyService(cluster_manager=cluster_manager)

# Modèles pour la documentation
generate_options = api.model('GenerateOptions', {
    'temperature': fields.Float(description='Température pour la génération', required=False, default=0.8),
    'top_p': fields.Float(description='Top-p pour la génération', required=False),
    'top_k': fields.Integer(description='Top-k pour la génération', required=False),
    'num_ctx': fields.Integer(description='Taille du contexte', required=False),
    'num_predict': fields.Integer(description='Nombre de tokens à prédire', required=False),
    'stop': fields.List(fields.String, description='Séquences d\'arrêt', required=False),
    'frequency_penalty': fields.Float(description='Pénalité de fréquence', required=False),
    'presence_penalty': fields.Float(description='Pénalité de présence', required=False),
    'repeat_penalty': fields.Float(description='Pénalité de répétition', required=False),
    'seed': fields.Integer(description='Graine pour la génération', required=False),
})

generate_request = api.model('GenerateRequest', {
    'model': fields.String(required=True, description='Nom du modèle à utiliser'),
    'prompt': fields.String(required=True, description='Texte à compléter'),
    'stream': fields.Boolean(required=False, default=False, description='Mode streaming'),
    'options': fields.Nested(generate_options, required=False, description='Options de génération')
})

generate_response = api.model('GenerateResponse', {
    'model': fields.String(description='Nom du modèle utilisé'),
    'response': fields.String(description='Texte généré'),
    'done': fields.Boolean(description='Indique si la génération est terminée'),
    'context': fields.List(fields.Integer, description='Contexte utilisé'),
    'total_duration': fields.Integer(description='Durée totale en nanosecondes'),
    'load_duration': fields.Integer(description='Durée de chargement en nanosecondes'),
    'prompt_eval_count': fields.Integer(description='Nombre de tokens évalués dans le prompt'),
    'eval_count': fields.Integer(description='Nombre de tokens générés'),
    'eval_duration': fields.Integer(description='Durée d\'évaluation en nanosecondes')
})

chat_message = api.model('ChatMessage', {
    'role': fields.String(required=True, description='Rôle dans la conversation (system, user, assistant)'),
    'content': fields.String(required=True, description='Contenu du message'),
    'images': fields.List(fields.String, description='Liste d\'URLs d\'images (pour la vision)', required=False)
})

chat_request = api.model('ChatRequest', {
    'model': fields.String(required=True, description='Nom du modèle à utiliser'),
    'messages': fields.List(fields.Nested(chat_message), required=True, description='Liste des messages de la conversation'),
    'stream': fields.Boolean(required=False, default=False, description='Mode streaming'),
    'options': fields.Nested(generate_options, required=False, description='Options de génération')
})

chat_response = api.model('ChatResponse', {
    'model': fields.String(description='Nom du modèle utilisé'),
    'message': fields.Nested(chat_message, description='Message généré par l\'assistant'),
    'done': fields.Boolean(description='Indique si la génération est terminée'),
    'total_duration': fields.Integer(description='Durée totale en nanosecondes'),
    'load_duration': fields.Integer(description='Durée de chargement en nanosecondes'),
    'prompt_eval_count': fields.Integer(description='Nombre de tokens évalués dans le prompt'),
    'eval_count': fields.Integer(description='Nombre de tokens générés'),
    'eval_duration': fields.Integer(description='Durée d\'évaluation en nanosecondes')
})

embeddings_request = api.model('EmbeddingsRequest', {
    'model': fields.String(required=True, description='Nom du modèle à utiliser'),
    'prompt': fields.String(required=True, description='Texte pour lequel générer l\'embedding'),
    'options': fields.Nested(generate_options, required=False, description='Options de génération')
})

embeddings_response = api.model('EmbeddingsResponse', {
    'model': fields.String(description='Nom du modèle utilisé'),
    'embedding': fields.List(fields.Float, description='Vecteur d\'embedding généré')
})

model_context = api.model('ModelContextInfo', {
    'current': fields.Integer(description='Taille du contexte actuel', default=4096),
    'max': fields.Integer(description='Taille maximum du contexte', default=8192)
})

model_info = api.model('ModelInfo', {
    'name': fields.String(description='Nom du modèle'),
    'size': fields.Integer(description='Taille du modèle en octets'),
    'modified_at': fields.DateTime(description='Date de dernière modification'),
    'version': fields.String(description='Version du modèle'),
    'available': fields.Boolean(description='Indique si le modèle est disponible'),
    'servers': fields.List(fields.String, description='Liste des serveurs hébergeant ce modèle'),
    'context': fields.Nested(model_context, description='Informations sur le contexte')
})

models_response = api.model('ModelsResponse', {
    'models': fields.List(fields.Nested(model_info), description='Liste des modèles disponibles')
})

server_info = api.model('ServerInfo', {
    'address': fields.String(description='Adresse du serveur'),
    'healthy': fields.Boolean(description='État de santé du serveur'),
    'load': fields.Float(description='Charge du serveur'),
    'models': fields.List(fields.String, description='Liste des modèles hébergés'),
    'capabilities': fields.Raw(description='Capacités du serveur')
})

servers_response = api.model('ServersResponse', {
    'servers': fields.List(fields.Nested(server_info), description='Liste des serveurs du cluster')
})

status_response = api.model('StatusResponse', {
    'timestamp': fields.Float(description='Timestamp Unix'),
    'server_time': fields.String(description='Heure du serveur au format ISO')
})

health_report = api.model('HealthReport', {
    'timestamp': fields.Float(description='Timestamp Unix'),
    'datetime': fields.String(description='Heure ISO'),
    'cluster_health': fields.Nested(api.model('ClusterHealth', {
        'total_servers': fields.Integer(description='Nombre total de serveurs'),
        'healthy_servers': fields.Integer(description='Nombre de serveurs sains'),
        'unhealthy_servers': fields.Integer(description='Nombre de serveurs défectueux'),
        'average_load': fields.Float(description='Charge moyenne du cluster'),
        'average_latency': fields.Float(description='Latence moyenne en ms')
    })),
    'servers': fields.Raw(description='Détails par serveur')
})

error_response = api.model('ErrorResponse', {
    'error': fields.String(description='Message d\'erreur')
})

@api_ns.route('/generate')
class Generate(Resource):
    @api_ns.doc('generate_text')
    @api_ns.expect(generate_request, validate=True)
    @api_ns.response(200, 'Succès', generate_response)
    @api_ns.response(400, 'Requête invalide', error_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def post(self):
        """Générer du texte à partir d'un prompt"""
        try:
            data = request.json
            generate_request = dict_to_generate_request(data)
            
            if generate_request.stream:
                return {"error": "Le streaming n'est pas supporté dans Swagger UI"}, 400
            
            response = ollama_service.generate(generate_request)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}")
            return {'error': f"Erreur de génération: {str(e)}"}, 500

@api_ns.route('/chat')
class Chat(Resource):
    @api_ns.doc('chat_conversation')
    @api_ns.expect(chat_request, validate=True)
    @api_ns.response(200, 'Succès', chat_response)
    @api_ns.response(400, 'Requête invalide', error_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def post(self):
        """Échanger avec un modèle en format conversation"""
        try:
            data = request.json
            chat_request = dict_to_chat_request(data)
            response = ollama_service.chat(chat_request)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors du chat: {str(e)}")
            return {'error': f"Erreur de chat: {str(e)}"}, 500

@api_ns.route('/embeddings')
class Embeddings(Resource):
    @api_ns.doc('generate_embeddings')
    @api_ns.expect(embeddings_request, validate=True)
    @api_ns.response(200, 'Succès', embeddings_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def post(self):
        """Générer des embeddings à partir d'un texte"""
        try:
            data = request.json
            embeddings_request = dict_to_embeddings_request(data)
            response = ollama_service.embeddings(embeddings_request)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embeddings: {str(e)}")
            return {'error': f"Erreur d'embeddings: {str(e)}"}, 500

@api_ns.route('/models')
class Models(Resource):
    @api_ns.doc('list_models')
    @api_ns.response(200, 'Succès', models_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Lister tous les modèles disponibles"""
        try:
            models_response = ollama_service.list_models()
            if hasattr(models_response, '__dict__') and not isinstance(models_response, dict):
                return asdict(models_response)
            return models_response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des modèles: {str(e)}")
            return {'error': f"Erreur de récupération des modèles: {str(e)}"}, 500

@api_ns.route('/status')
class Status(Resource):
    @api_ns.doc('get_status')
    @api_ns.response(200, 'Succès', status_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Obtenir le statut actuel du proxy"""
        try:
            status_response = ollama_service.get_status()
            if hasattr(status_response, '__dict__') and not isinstance(status_response, dict):
                return asdict(status_response)
            return status_response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut: {str(e)}")
            return {'error': f"Erreur de statut: {str(e)}"}, 500

@api_ns.route('/servers')
class Servers(Resource):
    @api_ns.doc('list_servers')
    @api_ns.response(200, 'Succès', servers_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Lister tous les serveurs du cluster"""
        try:
            servers_response = ollama_service.list_servers()
            if hasattr(servers_response, '__dict__') and not isinstance(servers_response, dict):
                return asdict(servers_response)
            return servers_response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des serveurs: {str(e)}")
            return {'error': f"Erreur de récupération des serveurs: {str(e)}"}, 500
    
    @api_ns.doc('add_server')
    @api_ns.expect(api.model('ServerAddRequest', {
        'address': fields.String(required=True, description='Adresse du serveur (host:port)'),
        'verify_health': fields.Boolean(required=False, default=True, description='Vérifier la santé du serveur')
    }))
    @api_ns.response(200, 'Succès')
    @api_ns.response(400, 'Requête invalide', error_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def post(self):
        """Ajouter un nouveau serveur au cluster"""
        try:
            data = request.json
            response = ollama_service.add_server(
                address=data.get('address'),
                verify_health=data.get('verify_health', True)
            )
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du serveur: {str(e)}")
            return {'error': f"Erreur d'ajout de serveur: {str(e)}"}, 500

@api_ns.route('/servers/<string:server>')
class ServerDetails(Resource):
    @api_ns.doc('get_server')
    @api_ns.response(200, 'Succès')
    @api_ns.response(404, 'Serveur non trouvé', error_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self, server):
        """Obtenir les détails d'un serveur"""
        try:
            response = ollama_service.get_server_details(server)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des détails du serveur: {str(e)}")
            return {'error': f"Erreur de récupération des détails: {str(e)}"}, 500
    
    @api_ns.doc('remove_server')
    @api_ns.response(200, 'Succès')
    @api_ns.response(404, 'Serveur non trouvé', error_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def delete(self, server):
        """Supprimer un serveur du cluster"""
        try:
            response = ollama_service.remove_server(server)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du serveur: {str(e)}")
            return {'error': f"Erreur de suppression de serveur: {str(e)}"}, 500

@api_ns.route('/servers/<string:server>/check_health')
class ServerHealth(Resource):
    @api_ns.doc('check_server_health')
    @api_ns.response(200, 'Succès')
    @api_ns.response(404, 'Serveur non trouvé', error_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def post(self, server):
        """Vérifier la santé d'un serveur"""
        try:
            response = ollama_service.check_server_health(server)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de santé du serveur: {str(e)}")
            return {'error': f"Erreur de vérification de santé: {str(e)}"}, 500

@api_ns.route('/health')
class Health(Resource):
    @api_ns.doc('get_health')
    @api_ns.response(200, 'Succès', health_report)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Obtenir un rapport de santé détaillé du cluster"""
        try:
            response = ollama_service.get_health_report()
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du rapport de santé: {str(e)}")
            return {'error': f"Erreur de rapport de santé: {str(e)}"}, 500

@api_ns.route('/health/stats')
class HealthStats(Resource):
    @api_ns.doc('get_health_stats')
    @api_ns.response(200, 'Succès')
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Obtenir des statistiques agrégées sur la santé du cluster"""
        try:
            hours = request.args.get('hours', 24, type=int)
            response = ollama_service.get_health_stats(hours=hours)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques de santé: {str(e)}")
            return {'error': f"Erreur de statistiques: {str(e)}"}, 500

@api_ns.route('/health/server/<string:server>')
class ServerHealthHistory(Resource):
    @api_ns.doc('get_server_health_history')
    @api_ns.response(200, 'Succès')
    @api_ns.response(404, 'Serveur non trouvé', error_response)
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self, server):
        """Obtenir l'historique de santé d'un serveur"""
        try:
            hours = request.args.get('hours', 24, type=int)
            response = ollama_service.get_server_health_history(server=server, hours=hours)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique de santé: {str(e)}")
            return {'error': f"Erreur d'historique de santé: {str(e)}"}, 500

@api_ns.route('/chart/load')
class LoadChartData(Resource):
    @api_ns.doc('get_load_chart_data')
    @api_ns.response(200, 'Succès')
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Obtenir les données pour le graphique de charge des serveurs"""
        try:
            hours = request.args.get('hours', 1, type=int)
            response = ollama_service.get_load_chart_data(hours=hours)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données du graphique: {str(e)}")
            return {'error': f"Erreur de données de graphique: {str(e)}"}, 500

@api_ns.route('/chart/health')
class HealthChartData(Resource):
    @api_ns.doc('get_health_chart_data')
    @api_ns.response(200, 'Succès')
    @api_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Obtenir les données pour le graphique de santé des serveurs"""
        try:
            hours = request.args.get('hours', 1, type=int)
            response = ollama_service.get_health_chart_data(hours=hours)
            if hasattr(response, '__dict__') and not isinstance(response, dict):
                return asdict(response)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données du graphique: {str(e)}")
            return {'error': f"Erreur de données de graphique: {str(e)}"}, 500

def init_swagger():
    """
    Initialise l'interface Swagger.
    Cette fonction doit être appelée après l'initialisation de l'application Flask.
    """
    logger.info("Initialisation de l'interface Swagger")
    return swagger_bp