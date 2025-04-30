/**
 * Ollama Sync - Traductions françaises
 */

// French translations
window.I18n = window.I18n || {};
window.I18n.translations = window.I18n.translations || {};

window.I18n.translations['fr'] = {
    // Application general
    "app": {
        "name": "Ollama Sync",
        "copyright": "Ollama Sync © 2025",
        "version": "Version 1.0.0"
    },
    
    // Navigation
    "nav": {
        "dashboard": "Tableau de bord",
        "models": "Modèles",
        "servers": "Serveurs",
        "health": "État du système",
        "queue": "File d'attente",
        "settings": "Paramètres",
        "playground": "Terrain de jeu",
        "terminal": "Terminal",
        "swagger": "Documentation API",
        "api_documentation": "Documentation API",
        "language": "Langue"
    },
    
    // Notifications
    "notifications": {
        "title": "Notifications",
        "new_server": "Nouveau serveur détecté",
        "server_offline": "Serveur hors ligne",
        "high_load": "Charge élevée",
        "model_downloaded": "Modèle téléchargé",
        "just_now": "À l'instant",
        "see_all": "Voir toutes les notifications"
    },
    
    // Dashboard
    "dashboard": {
        "title": "Tableau de bord système",
        "system_stats": "Statistiques système",
        "active_servers": "Serveurs actifs",
        "total_models": "Modèles total",
        "active_requests": "Requêtes actives",
        "request_rate": "Taux de requêtes",
        "cpu_usage": "Utilisation CPU",
        "memory_usage": "Utilisation mémoire",
        "disk_usage": "Utilisation disque",
        "network_usage": "Utilisation réseau",
        "recent_activity": "Activité récente",
        "quick_actions": "Actions rapides"
    },
    
    // Models
    "models": {
        "title": "Gestion des modèles",
        "available_models": "Modèles disponibles",
        "model_name": "Nom du modèle",
        "size": "Taille",
        "modified": "Modifié",
        "quantization": "Quantification",
        "download": "Télécharger",
        "delete": "Supprimer",
        "pull": "Récupérer modèle",
        "model_details": "Détails du modèle",
        "parameters": "Paramètres",
        "server_location": "Localisation serveur",
        "loading": "Chargement des modèles..."
    },
    
    // Servers
    "servers": {
        "title": "Gestion des serveurs",
        "available_servers": "Serveurs disponibles",
        "hostname": "Nom d'hôte",
        "address": "Adresse",
        "status": "Statut",
        "load": "Charge",
        "online": "En ligne",
        "offline": "Hors ligne",
        "add_server": "Ajouter serveur",
        "remove": "Supprimer",
        "server_details": "Détails du serveur",
        "uptime": "Temps de fonctionnement",
        "cpu": "CPU",
        "memory": "Mémoire",
        "models_hosted": "Modèles hébergés"
    },
    
    // Health
    "health": {
        "title": "État du système",
        "system_status": "Statut du système",
        "server_status": "Statut du serveur",
        "response_time": "Temps de réponse",
        "uptime": "Temps de fonctionnement",
        "last_checked": "Dernière vérification",
        "healthy": "En bonne santé",
        "degraded": "Dégradé",
        "unhealthy": "Défaillant",
        "check_now": "Vérifier maintenant"
    },
    
    // Settings
    "settings": {
        "title": "Paramètres système",
        "general": "Général",
        "appearance": "Apparence",
        "network": "Réseau",
        "language": "Langue",
        "language_selector": "Sélection de langue",
        "current_language": "Langue actuelle",
        "theme": "Thème",
        "dark_mode": "Mode sombre",
        "light_mode": "Mode clair",
        "auto_discovery": "Découverte automatique",
        "save": "Sauvegarder",
        "reset": "Réinitialiser",
        "enable": "Activer",
        "disable": "Désactiver"
    },
    
    // Queue
    "queue": {
        "title": "File d'attente des requêtes",
        "current_queue": "File d'attente actuelle",
        "request_id": "ID de requête",
        "model": "Modèle",
        "timestamp": "Horodatage",
        "status": "Statut",
        "action": "Action",
        "pending": "En attente",
        "processing": "En cours",
        "completed": "Terminé",
        "failed": "Échoué",
        "cancel": "Annuler",
        "retry": "Réessayer"
    },
    
    // Playground
    "playground": {
        "title": "Terrain de jeu",
        "select_model": "Sélectionner un modèle",
        "parameters": "Paramètres",
        "temperature": "Température",
        "max_tokens": "Tokens max",
        "top_p": "Top P",
        "prompt": "Entrez votre prompt ici...",
        "generate": "Générer",
        "stop": "Arrêter",
        "response": "Réponse",
        "copy": "Copier",
        "clear": "Effacer",
        "save": "Sauvegarder"
    },
    
    // Terminal
    "terminal": {
        "title": "Interface Terminal",
        "welcome": "Bienvenue sur le Terminal Ollama Sync",
        "help": "Tapez 'help' pour une liste des commandes",
        "command": "Commande",
        "output": "Sortie",
        "clear": "Effacer le terminal"
    },
    
    // Common actions/buttons
    "actions": {
        "add": "Ajouter",
        "edit": "Modifier",
        "delete": "Supprimer",
        "cancel": "Annuler",
        "save": "Sauvegarder",
        "refresh": "Rafraîchir",
        "close": "Fermer",
        "confirm": "Confirmer",
        "back": "Retour",
        "next": "Suivant"
    },
    
    // Errors and states
    "states": {
        "loading": "Chargement...",
        "error": "Erreur",
        "success": "Succès",
        "warning": "Avertissement",
        "info": "Information"
    },
    
    // API documentation
    "api": {
        "title": "Documentation API",
        "description": "Documentation API interactive pour Ollama Sync",
        "endpoints": "Points de terminaison",
        "models": "Modèles",
        "try_it": "Essayer",
        "response": "Réponse",
        "status": "Statut"
    }
};