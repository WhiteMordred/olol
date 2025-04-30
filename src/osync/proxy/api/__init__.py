"""
Module API pour le proxy Ollama.

Ce module contient les définitions d'API pour interagir avec les serveurs Ollama
à travers le proxy. Il fournit des endpoints stables et bien documentés.
"""

from flask import Blueprint

# Création du blueprint pour les routes API
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Ne pas importer routes ici pour éviter l'importation circulaire
# from .routes import register_api_routes

# Fonction pour initialiser l'API avec l'application Flask
def init_app(app, cluster_manager):
    """
    Initialise l'API avec l'application Flask et le gestionnaire de cluster.
    
    Args:
        app: L'application Flask
        cluster_manager: Le gestionnaire de cluster Ollama
    """
    # Import local pour éviter l'importation circulaire
    from .routes import register_api_routes
    
    # Enregistrement des routes avec le blueprint
    register_api_routes(app, cluster_manager)
    
    return api_bp