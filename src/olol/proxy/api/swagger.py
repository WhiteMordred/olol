"""
Documentation Swagger pour l'API du proxy Ollama.

Ce module crée une interface Swagger pour documenter l'API RESTful du proxy Ollama,
permettant aux utilisateurs de visualiser et de tester les endpoints disponibles.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_restx import Api, Resource, fields, Namespace
from typing import Dict, Any, Optional, List

from olol.proxy.api.services import OllamaProxyService
from olol.proxy.cluster.manager import ClusterManager
from olol.proxy.cluster.health import get_health_monitor

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
    doc='/',  # Modifié de '/docs' à '/' pour éviter la duplication dans l'URL
    validate=True
)

# Namespace pour les endpoints principaux
ollama_ns = api.namespace('api', description='Opérations de l\'API Ollama')

# Namespace pour les endpoints du cluster
cluster_ns = api.namespace('cluster', description='Opérations de gestion du cluster')

# Service API Ollama
cluster_manager = ClusterManager()
ollama_service = OllamaProxyService(cluster_manager=cluster_manager)

# Modèles pour la documentation
# Modèle pour la requête de génération
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

# Modèle pour le chat
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

# Modèle pour les embeddings
embeddings_request = api.model('EmbeddingsRequest', {
    'model': fields.String(required=True, description='Nom du modèle à utiliser'),
    'prompt': fields.String(required=True, description='Texte pour lequel générer l\'embedding'),
    'options': fields.Nested(generate_options, required=False, description='Options de génération')
})

embeddings_response = api.model('EmbeddingsResponse', {
    'model': fields.String(description='Nom du modèle utilisé'),
    'embedding': fields.List(fields.Float, description='Vecteur d\'embedding généré')
})

# Modèles pour le statut et les informations
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

# Erreur
error_response = api.model('ErrorResponse', {
    'error': fields.String(description='Message d\'erreur')
})


@ollama_ns.route('/generate')
class Generate(Resource):
    @ollama_ns.doc('generate_text')
    @ollama_ns.expect(generate_request, validate=True)
    @ollama_ns.response(200, 'Succès', generate_response)
    @ollama_ns.response(400, 'Requête invalide', error_response)
    @ollama_ns.response(500, 'Erreur serveur', error_response)
    @ollama_ns.response(503, 'Service indisponible', error_response)
    def post(self):
        """Générer du texte à partir d'un prompt"""
        try:
            data = request.json
            response = ollama_service.generate(
                model=data.get('model'),
                prompt=data.get('prompt'),
                stream=data.get('stream', False),
                options=data.get('options', {})
            )
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}")
            return {'error': f"Erreur de génération: {str(e)}"}, 500


@ollama_ns.route('/chat')
class Chat(Resource):
    @ollama_ns.doc('chat_conversation')
    @ollama_ns.expect(chat_request, validate=True)
    @ollama_ns.response(200, 'Succès', chat_response)
    @ollama_ns.response(400, 'Requête invalide', error_response)
    @ollama_ns.response(500, 'Erreur serveur', error_response)
    @ollama_ns.response(503, 'Service indisponible', error_response)
    def post(self):
        """Échanger avec un modèle en format conversation"""
        try:
            data = request.json
            response = ollama_service.chat(
                model=data.get('model'),
                messages=data.get('messages'),
                stream=data.get('stream', False),
                options=data.get('options', {})
            )
            return response
        except Exception as e:
            logger.error(f"Erreur lors du chat: {str(e)}")
            return {'error': f"Erreur de chat: {str(e)}"}, 500


@ollama_ns.route('/embeddings')
class Embeddings(Resource):
    @ollama_ns.doc('generate_embeddings')
    @ollama_ns.expect(embeddings_request, validate=True)
    @ollama_ns.response(200, 'Succès', embeddings_response)
    @ollama_ns.response(400, 'Requête invalide', error_response)
    @ollama_ns.response(500, 'Erreur serveur', error_response)
    @ollama_ns.response(503, 'Service indisponible', error_response)
    def post(self):
        """Générer des embeddings à partir d'un texte"""
        try:
            data = request.json
            response = ollama_service.embeddings(
                model=data.get('model'),
                prompt=data.get('prompt'),
                options=data.get('options', {})
            )
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embeddings: {str(e)}")
            return {'error': f"Erreur d'embeddings: {str(e)}"}, 500


@ollama_ns.route('/models')
class Models(Resource):
    @ollama_ns.doc('list_models')
    @ollama_ns.response(200, 'Succès', models_response)
    @ollama_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Lister tous les modèles disponibles sur le cluster"""
        try:
            return ollama_service.list_models()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des modèles: {str(e)}")
            return {'error': f"Erreur de récupération des modèles: {str(e)}"}, 500


@cluster_ns.route('/servers')
class Servers(Resource):
    @cluster_ns.doc('list_servers')
    @cluster_ns.response(200, 'Succès', servers_response)
    @cluster_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Lister tous les serveurs du cluster avec leurs détails"""
        try:
            return ollama_service.list_servers()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des serveurs: {str(e)}")
            return {'error': f"Erreur de récupération des serveurs: {str(e)}"}, 500


@cluster_ns.route('/health')
class ClusterHealth(Resource):
    @cluster_ns.doc('get_health')
    @cluster_ns.response(200, 'Succès', health_report)
    @cluster_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Obtenir un rapport de santé détaillé du cluster"""
        try:
            health_monitor = get_health_monitor()
            return health_monitor.get_health_report()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du rapport de santé: {str(e)}")
            return {'error': f"Erreur de rapport de santé: {str(e)}"}, 500


@cluster_ns.route('/health/stats')
class ClusterHealthStats(Resource):
    @cluster_ns.doc('get_health_stats')
    @cluster_ns.response(200, 'Succès')
    @cluster_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Obtenir des statistiques agrégées sur la santé du cluster"""
        try:
            hours = request.args.get('hours', 24, type=int)
            health_monitor = get_health_monitor()
            return health_monitor.get_cluster_health_stats(hours=hours)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques de santé: {str(e)}")
            return {'error': f"Erreur de statistiques: {str(e)}"}, 500


@cluster_ns.route('/health/server/<string:server>')
class ServerHealth(Resource):
    @cluster_ns.doc('get_server_health')
    @cluster_ns.response(200, 'Succès')
    @cluster_ns.response(404, 'Serveur non trouvé', error_response)
    @cluster_ns.response(500, 'Erreur serveur', error_response)
    def get(self, server):
        """Obtenir l'historique de santé d'un serveur spécifique"""
        try:
            hours = request.args.get('hours', 24, type=int)
            health_monitor = get_health_monitor()
            history = health_monitor.get_health_history(server=server, hours=hours)
            if not history.get('health') and not history.get('load'):
                return {'error': 'Serveur non trouvé ou aucune donnée disponible'}, 404
            return history
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de santé du serveur: {str(e)}")
            return {'error': f"Erreur de récupération: {str(e)}"}, 500


@ollama_ns.route('/status')
class Status(Resource):
    @ollama_ns.doc('get_status')
    @ollama_ns.response(200, 'Succès', status_response)
    @ollama_ns.response(500, 'Erreur serveur', error_response)
    def get(self):
        """Obtenir le statut actuel du proxy"""
        try:
            return ollama_service.get_status()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut: {str(e)}")
            return {'error': f"Erreur de statut: {str(e)}"}, 500


def init_swagger():
    """
    Initialise l'interface Swagger.
    Cette fonction doit être appelée après l'initialisation de l'application Flask.
    """
    logger.info("Initialisation de l'interface Swagger")
    return swagger_bp