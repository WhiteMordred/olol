"""
Module API pour le proxy Ollama.

Ce module contient les définitions d'API pour interagir avec les serveurs Ollama
à travers le proxy. Il fournit des endpoints stables et bien documentés.
"""

from flask import Blueprint

# Création du blueprint pour les routes API
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Importation des routes pour les enregistrer avec le blueprint
from .routes import register_api_routes

# Fonction pour initialiser l'API avec l'application Flask
def init_app(app, cluster_manager):
    """
    Initialise l'API avec l'application Flask et le gestionnaire de cluster.
    
    Args:
        app: L'application Flask
        cluster_manager: Le gestionnaire de cluster Ollama
    """
    # Enregistrement des routes avec le blueprint
    register_api_routes(api_bp, cluster_manager)
    
    # Enregistrement du blueprint avec l'application Flask
    app.register_blueprint(api_bp)
    
    return api_bp