/**
 * Traductions françaises pour Ollama Sync
 */
(function() {
    // Dictionnaire des traductions françaises
    const translations = {
        // Navigation et interface générale
        "dashboard": "Tableau de bord",
        "servers": "Serveurs",
        "models": "Modèles",
        "queue": "File d'attente",
        "playground": "Playground",
        "health": "Santé",
        "logs": "Journaux",
        "terminal": "Terminal",
        "settings": "Paramètres",
        "api_docs": "Documentation API",
        "collapse": "Replier",
        "expand": "Déplier",
        "language": "Langue",
        "copyright": "Ollama Sync © 2025",
        
        // Notifications
        "notifications": "Notifications",
        "see_all_notifications": "Voir toutes les notifications",
        "new_server_detected": "Nouveau serveur détecté",
        "high_load": "Charge élevée",
        "server_offline": "Serveur hors ligne",
        
        // État des serveurs
        "cluster_status": "État du cluster",
        "active_servers": "Serveurs actifs",
        "manage_servers": "Gérer les serveurs",
        "proxy_active": "Proxy actif",
        
        // Page serveurs
        "add_server": "Ajouter un serveur",
        "remove_server": "Supprimer un serveur",
        "server_address": "Adresse du serveur",
        "server_port": "Port",
        "server_status": "Statut",
        "server_load": "Charge",
        "server_models": "Modèles",
        "server_actions": "Actions",
        
        // Page modèles
        "model_name": "Nom du modèle",
        "model_size": "Taille",
        "model_version": "Version",
        "model_family": "Famille",
        "model_servers": "Serveurs",
        "pull_model": "Télécharger le modèle",
        "delete_model": "Supprimer le modèle",
        
        // File d'attente
        "request_id": "ID de la requête",
        "request_type": "Type",
        "request_status": "Statut",
        "request_server": "Serveur",
        "request_model": "Modèle",
        "request_priority": "Priorité",
        "request_time": "Heure",
        "clear_queue": "Vider la file",
        
        // Page santé
        "system_health": "Santé du système",
        "resource_usage": "Utilisation des ressources",
        "memory_usage": "Mémoire",
        "cpu_usage": "CPU",
        "disk_usage": "Disque",
        "network_traffic": "Trafic réseau",
        "last_updated": "Dernière mise à jour",
        
        // Paramètres
        "general_settings": "Paramètres généraux",
        "network_settings": "Paramètres réseau",
        "security_settings": "Paramètres de sécurité",
        "advanced_settings": "Paramètres avancés",
        "save_settings": "Enregistrer",
        "reset_settings": "Réinitialiser",
        
        // Messages
        "confirm_delete": "Êtes-vous sûr de vouloir supprimer cet élément ?",
        "operation_success": "Opération réussie",
        "operation_failed": "Échec de l'opération",
        "loading": "Chargement...",
        "no_data": "Aucune donnée disponible",
        
        // Boutons et actions
        "refresh": "Actualiser",
        "cancel": "Annuler",
        "confirm": "Confirmer",
        "edit": "Modifier",
        "delete": "Supprimer",
        "view": "Voir",
        "search": "Rechercher",
        "apply": "Appliquer",
        "reset": "Réinitialiser"
    };
    
    // Enregistrer les traductions
    if (window.OllamaI18n) {
        window.OllamaI18n.registerTranslations('fr', translations);
    } else {
        // Si le système i18n n'est pas encore chargé, attendre un peu
        window.addEventListener('DOMContentLoaded', function() {
            if (window.OllamaI18n) {
                window.OllamaI18n.registerTranslations('fr', translations);
            }
        });
    }
})();