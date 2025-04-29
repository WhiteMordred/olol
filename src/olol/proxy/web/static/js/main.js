/**
 * OLOL Proxy - Main JavaScript
 * Fonctions principales pour l'interface web du proxy OLOL
 */

// Fonctions d'utilitaires
function showToast(title, message, type = 'info') {
    const toast = document.getElementById('notificationToast');
    const toastTitle = document.getElementById('toastTitle');
    const toastMessage = document.getElementById('toastMessage');
    const toastTime = document.getElementById('toastTime');
    
    // Définir le contenu
    toastTitle.textContent = title;
    toastMessage.textContent = message;
    toastTime.textContent = 'À l\'instant';
    
    // Définir le type (couleur)
    toast.classList.remove('bg-success', 'bg-danger', 'bg-warning', 'bg-info');
    switch(type) {
        case 'success':
            toast.classList.add('bg-success', 'text-white');
            break;
        case 'error':
            toast.classList.add('bg-danger', 'text-white');
            break;
        case 'warning':
            toast.classList.add('bg-warning');
            break;
        default:
            toast.classList.add('bg-info', 'text-white');
    }
    
    // Initialiser et afficher le toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}

// Format d'affichage pour les dates
function formatDate(date) {
    const now = new Date();
    const diff = Math.floor((now - date) / 1000); // différence en secondes
    
    if (diff < 60) {
        return 'À l\'instant';
    } else if (diff < 3600) {
        const minutes = Math.floor(diff / 60);
        return `Il y a ${minutes} minute${minutes > 1 ? 's' : ''}`;
    } else if (diff < 86400) {
        const hours = Math.floor(diff / 3600);
        return `Il y a ${hours} heure${hours > 1 ? 's' : ''}`;
    } else {
        const days = Math.floor(diff / 86400);
        return `Il y a ${days} jour${days > 1 ? 's' : ''}`;
    }
}

// Fonction pour formater les nombres
function formatNumber(num) {
    return new Intl.NumberFormat('fr-FR').format(num);
}

// Fonction pour formater la taille des fichiers
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Classe pour gérer les actualisations automatiques
class AutoRefresh {
    constructor(elementId, refreshFunction, interval = 30000) {
        this.elementId = elementId;
        this.refreshFunction = refreshFunction;
        this.interval = interval;
        this.timerId = null;
        this.isActive = false;
    }
    
    start() {
        if (!this.isActive) {
            this.refreshFunction();
            this.timerId = setInterval(() => {
                this.refreshFunction();
            }, this.interval);
            this.isActive = true;
            
            // Mettre à jour l'UI
            const element = document.getElementById(this.elementId);
            if (element) {
                element.classList.add('active');
                const statusText = element.querySelector('.status-text');
                if (statusText) {
                    statusText.textContent = 'Actif';
                }
            }
        }
    }
    
    stop() {
        if (this.isActive) {
            clearInterval(this.timerId);
            this.isActive = false;
            
            // Mettre à jour l'UI
            const element = document.getElementById(this.elementId);
            if (element) {
                element.classList.remove('active');
                const statusText = element.querySelector('.status-text');
                if (statusText) {
                    statusText.textContent = 'Actualisation auto';
                }
            }
        }
    }
    
    toggle() {
        if (this.isActive) {
            this.stop();
        } else {
            this.start();
        }
    }
}

// Classe pour gérer les appels API
class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }
    
    async get(endpoint) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`);
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Erreur API GET ${endpoint}:`, error);
            throw error;
        }
    }
    
    async post(endpoint, data) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Erreur API POST ${endpoint}:`, error);
            throw error;
        }
    }
}

// Classe pour le Playground
class Playground {
    constructor(containerId, messageContainerId, inputId, sendButtonId, modelSelectId) {
        this.container = document.getElementById(containerId);
        this.messageContainer = document.getElementById(messageContainerId);
        this.input = document.getElementById(inputId);
        this.sendButton = document.getElementById(sendButtonId);
        this.modelSelect = document.getElementById(modelSelectId);
        this.apiClient = new ApiClient();
        this.conversationId = null;
        
        this.init();
    }
    
    init() {
        // Générer un ID de conversation unique
        this.conversationId = Date.now().toString(36) + Math.random().toString(36).substring(2);
        
        // Configurer les écouteurs d'événements
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Ajuster la hauteur du textarea automatiquement
        this.input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Afficher le message de bienvenue
        this.addMessage({
            role: 'ai',
            content: `Bienvenue dans le Playground OLOL ! Je suis prêt à vous aider. Que voulez-vous savoir ?`,
            timestamp: new Date()
        });
    }
    
    async sendMessage() {
        const message = this.input.value.trim();
        if (!message) return;
        
        // Obtenir le modèle sélectionné
        const selectedModel = this.modelSelect.value;
        
        // Ajouter le message de l'utilisateur à l'interface
        this.addMessage({
            role: 'user',
            content: message,
            timestamp: new Date()
        });
        
        // Réinitialiser l'input
        this.input.value = '';
        this.input.style.height = 'auto';
        this.input.focus();
        
        // Désactiver le bouton d'envoi pendant le traitement
        this.sendButton.disabled = true;
        this.sendButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Envoi...';
        
        try {
            // Appeler l'API de complétion
            const response = await this.apiClient.post('/api/v1/chat/completions', {
                model: selectedModel,
                messages: [{ role: 'user', content: message }],
                conversation_id: this.conversationId
            });
            
            // Ajouter la réponse à l'interface
            this.addMessage({
                role: 'ai',
                content: response.choices[0].message.content,
                timestamp: new Date()
            });
        } catch (error) {
            console.error('Erreur lors de l\'envoi du message:', error);
            
            // Afficher une erreur à l'utilisateur
            this.addMessage({
                role: 'ai',
                content: `Désolé, une erreur s'est produite : ${error.message}`,
                timestamp: new Date(),
                isError: true
            });
            
            // Afficher un toast d'erreur
            showToast('Erreur', 'Impossible de communiquer avec le modèle', 'error');
        } finally {
            // Réactiver le bouton d'envoi
            this.sendButton.disabled = false;
            this.sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }
    
    addMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role} ${message.isError ? 'error' : ''}`;
        
        // Formatter le contenu avec Markdown si nécessaire
        let formattedContent = message.content;
        
        // Ajouter le contenu et le timestamp
        messageDiv.innerHTML = `
            <div class="message-content">${formattedContent}</div>
            <span class="timestamp">${formatDate(message.timestamp)}</span>
        `;
        
        // Ajouter au conteneur et faire défiler vers le bas
        this.messageContainer.appendChild(messageDiv);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
}

// Initialiser les tooltips et popovers de Bootstrap
document.addEventListener('DOMContentLoaded', function() {
    // Initialiser les tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialiser les popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Initialiser le Playground si on est sur la page correspondante
    if (document.getElementById('playgroundContainer')) {
        window.playground = new Playground(
            'playgroundContainer',
            'messageContainer',
            'userInput',
            'sendButton',
            'modelSelect'
        );
    }
});