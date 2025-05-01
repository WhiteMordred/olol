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
        "playground": "Playground",
        "terminal": "Terminal",
        "log": "Journaux",
        "swagger": "Documentation API",
        "api_documentation": "Documentation API",
        "language": "Langue"
    },
    
    // Sidebar
    "sidebar": {
        "collapse": "Replier",
        "expand": "Déplier"
    },
    
    // Notifications
    "notifications": {
        "title": "Notifications",
        "new_server": "Nouveau serveur détecté",
        "server_offline": "Serveur hors ligne",
        "high_load": "Charge élevée",
        "model_downloaded": "Modèle téléchargé",
        "just_now": "À l'instant",
        "see_all": "Voir toutes les notifications",
        "success": "Succès",
        "error": "Erreur",
        "warning": "Avertissement",
        "info": "Information"
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
        "title": "Gestion des Modèles",
        "available_models": "Modèles disponibles",
        "model_name": "Nom du modèle",
        "size": "Taille",
        "modified": "Modifié",
        "quantization": "Quantification",
        "download": "Télécharger",
        "delete": "Supprimer",
        "pull": {
            "title": "Télécharger un modèle",
            "submit": "Télécharger",
            "progress": "Téléchargement...",
            "progress_msg": "Téléchargement de {{model}} en cours...",
            "complete": "Téléchargement terminé",
            "success": "{{model}} a été téléchargé avec succès"
        },
        "details": "Informations du modèle",
        "info": {
            "general": "Informations générales",
            "title_for": "Informations: {{name}}"
        },
        "parameters": "paramètres",
        "server_location": "Localisation serveur",
        "loading": "Chargement des modèles...",
        "status": {
            "available": "Disponible",
            "unavailable": "Non disponible"
        },
        "add": {
            "name": "Nom du modèle",
            "server": "Serveur cible"
        },
        "name": {
            "placeholder": "ex: llama2, mistral, etc.",
            "help": "Entrez le nom du modèle à télécharger depuis le catalogue Ollama."
        },
        "server": {
            "select": "Sélectionner un serveur",
            "all": "Tous les serveurs",
            "help": "Sélectionnez le serveur sur lequel vous souhaitez télécharger ce modèle."
        },
        "insecure": "Autoriser les sources non sécurisées",
        "none_available": "Aucun modèle disponible. Utilisez le bouton \"Télécharger un modèle\" pour en ajouter.",
        "filter": {
            "by": "Filtrer par",
            "all": "Tous les modèles",
            "local": "Modèles locaux",
            "downloaded": "Modèles téléchargés",
            "placeholder": "Rechercher un modèle..."
        },
        "table": {
            "name": "Nom",
            "size": "Taille:",
            "servers": "Serveurs disponibles:",
            "parameters": "Paramètres:",
            "modified": "Dernière mise à jour:",
            "family": "Famille",
            "quantization": "Quantization"
        },
        "test": "Tester",
        "copy_prompt": "Copier le prompt",
        "distribute": {
            "title": "Distribution en cours",
            "progress": "Distribution de {{model}} sur tous les serveurs...",
            "complete": "Distribution terminée",
            "success": "{{model}} a été distribué avec succès"
        },
        "delete": {
            "title": "Suppression en cours",
            "confirm": "Confirmer la suppression",
            "confirm_text": "Êtes-vous sûr de vouloir supprimer le modèle",
            "warning": "Cette action supprimera le modèle de tous les serveurs où il est installé.",
            "progress": "Suppression...",
            "progress_msg": "Suppression de {{model}}...",
            "complete": "Suppression terminée",
            "success": "{{model}} a été supprimé avec succès"
        },
        "availability": "Disponibilité",
        "prompt_template": "Template de prompt",
        "default_params": "Paramètres par défaut",
        "prompt_copied": "Template de prompt copié dans le presse-papier",
        "prompt_copy_error": "Impossible de copier le template"
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
        "models_hosted": "Modèles hébergés",
        "cluster_status": "État du cluster",
        "active": "Serveurs actifs",
        "manage": "Gérer les serveurs",
        "none_available": "Aucun serveur",
        "status": {
            "proxy_active": "Proxy actif"
        },
        "table": {
            "name": "Serveur",
            "status": "État"
        }
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
        "check_now": "Vérifier maintenant",
        "refresh": {
            "auto": "Actualisation auto"
        }
    },
    
    // Settings
    "settings": {
        "title": "Paramètres système",
        "general": "Général",
        "appearance": "Apparence",
        "network": "Réseau",
        "language": {
            "title": "Langue",
            "selector": "Sélection de langue", 
            "current": "Langue actuelle",
            "changed": "La langue a été changée avec succès"
        },
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
    "action": {
        "add": "Ajouter",
        "edit": "Modifier",
        "delete": "Supprimer",
        "cancel": "Annuler",
        "save": "Sauvegarder",
        "refresh": "Rafraîchir",
        "close": "Fermer",
        "confirm": "Confirmer",
        "back": "Retour",
        "next": "Suivant",
        "active": "Actualisation active"
    },
    
    // Success messages
    "success": {
        "copied": "Succès",
        "saved": "Sauvegardé avec succès",
        "updated": "Mise à jour réussie",
        "deleted": "Supprimé avec succès"
    },
    
    // Error messages
    "error": {
        "generic": "Erreur",
        "required_field": "Veuillez remplir tous les champs requis",
        "connection": "Erreur de connexion",
        "not_found": "Non trouvé",
        "server_error": "Erreur serveur",
        "permission": "Erreur de permission"
    },
    
    // Pagination
    "pagination": {
        "previous": "Précédent",
        "next": "Suivant",
        "showing": "Affichage de {{start}} à {{end}} sur {{total}} entrées"
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