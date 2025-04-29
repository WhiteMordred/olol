/**
 * OLOL Proxy - Script JavaScript principal
 * Ce fichier contient les fonctions communes utilisées dans l'interface web
 */

// Exécuter le code quand le DOM est chargé
document.addEventListener('DOMContentLoaded', function() {
    // Initialiser les tooltips Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialiser les popovers Bootstrap
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Ajouter une classe 'active' aux liens de navigation en fonction de l'URL actuelle
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentLocation || (href !== '/' && currentLocation.startsWith(href))) {
            link.classList.add('active');
        }
    });
    
    // Gestionnaire pour les boutons de rafraîchissement automatique des données
    const autoRefreshToggles = document.querySelectorAll('.auto-refresh-toggle');
    autoRefreshToggles.forEach(toggle => {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            const target = this.getAttribute('data-target');
            const seconds = parseInt(this.getAttribute('data-seconds'), 10);
            toggleAutoRefresh(target, seconds);
        });
    });
    
    // Gestionnaire pour les notifications de type toast
    const toastElList = [].slice.call(document.querySelectorAll('.toast'));
    const toastList = toastElList.map(function (toastEl) {
        return new bootstrap.Toast(toastEl);
    });
    
    // Montrer les notifications au chargement si elles existent
    toastList.forEach(toast => toast.show());
    
    // Initialiser la recherche dans les tableaux
    const tableSearchInputs = document.querySelectorAll('.table-search');
    tableSearchInputs.forEach(input => {
        input.addEventListener('keyup', function() {
            const tableId = this.getAttribute('data-table');
            const table = document.getElementById(tableId);
            if (table) {
                filterTable(table, this.value.toLowerCase());
            }
        });
    });
    
    // Gestionnaire pour les boutons de chargement (spinner)
    const loadingButtons = document.querySelectorAll('.btn-loading');
    loadingButtons.forEach(button => {
        button.addEventListener('click', function() {
            const originalHtml = this.innerHTML;
            this.setAttribute('data-original-html', originalHtml);
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Chargement...';
            this.disabled = true;
            
            // Restaurer après le timeout (si l'action prend trop de temps)
            setTimeout(() => {
                if (this.innerHTML.includes('Chargement')) {
                    this.innerHTML = originalHtml;
                    this.disabled = false;
                }
            }, 10000);
        });
    });
});

/**
 * Système de notifications
 * Permet d'afficher des messages temporaires à l'utilisateur
 */
const NotificationSystem = {
    show: function(message, type = 'info', duration = 5000) {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            // Créer le conteneur s'il n'existe pas
            const container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(container);
        }
        
        // Créer l'élément toast
        const toastId = 'toast-' + Date.now();
        const toastHtml = `
            <div class="toast" id="${toastId}" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header">
                    <div class="rounded me-2 bg-${type}" style="width: 20px; height: 20px;"></div>
                    <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                    <small>${new Date().toLocaleTimeString()}</small>
                    <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        document.getElementById('toast-container').insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: duration });
        toast.show();
        
        // Supprimer l'élément après disparition
        toastElement.addEventListener('hidden.bs.toast', function() {
            this.remove();
        });
    },
    success: function(message, duration = 5000) {
        this.show(message, 'success', duration);
    },
    error: function(message, duration = 5000) {
        this.show(message, 'danger', duration);
    },
    warning: function(message, duration = 5000) {
        this.show(message, 'warning', duration);
    },
    info: function(message, duration = 5000) {
        this.show(message, 'info', duration);
    }
};

/**
 * Système d'actualisation automatique
 */
const AutoRefresh = {
    timers: {},
    start: function(targetId, seconds, callback) {
        // Arrêter le timer existant s'il y en a un
        if (this.timers[targetId]) {
            clearInterval(this.timers[targetId]);
        }
        
        // Démarrer un nouveau timer
        this.timers[targetId] = setInterval(callback, seconds * 1000);
        
        // Mettre à jour l'UI
        const toggle = document.querySelector(`.auto-refresh-toggle[data-target="${targetId}"]`);
        if (toggle) {
            toggle.classList.add('active');
            toggle.querySelector('.status-text').textContent = 'Activé';
        }
    },
    stop: function(targetId) {
        // Arrêter le timer
        if (this.timers[targetId]) {
            clearInterval(this.timers[targetId]);
            delete this.timers[targetId];
        }
        
        // Mettre à jour l'UI
        const toggle = document.querySelector(`.auto-refresh-toggle[data-target="${targetId}"]`);
        if (toggle) {
            toggle.classList.remove('active');
            toggle.querySelector('.status-text').textContent = 'Désactivé';
        }
    },
    isRunning: function(targetId) {
        return !!this.timers[targetId];
    }
};

function toggleAutoRefresh(targetId, seconds) {
    const isRunning = AutoRefresh.isRunning(targetId);
    
    if (isRunning) {
        AutoRefresh.stop(targetId);
    } else {
        // Récupérer le callback personnalisé pour chaque targetId
        let callback;
        switch (targetId) {
            case 'health-data':
                callback = refreshHealthData;
                break;
            case 'servers-list':
                callback = refreshServersList;
                break;
            case 'models-list':
                callback = refreshModelsList;
                break;
            default:
                callback = function() {
                    console.log('Actualisation pour ' + targetId);
                    location.reload();
                };
        }
        
        AutoRefresh.start(targetId, seconds, callback);
        // Exécuter immédiatement pour la première fois
        callback();
    }
}

/**
 * Fonctions de filtrage de tableau
 */
function filterTable(table, query) {
    const rows = table.querySelectorAll('tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    });
}

/**
 * Fonctions utilitaires
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    if (!seconds) return '0s';
    
    const days = Math.floor(seconds / (3600 * 24));
    seconds %= (3600 * 24);
    const hours = Math.floor(seconds / 3600);
    seconds %= 3600;
    const minutes = Math.floor(seconds / 60);
    seconds %= 60;
    seconds = Math.floor(seconds);
    
    let result = '';
    if (days > 0) result += `${days}j `;
    if (hours > 0) result += `${hours}h `;
    if (minutes > 0) result += `${minutes}m `;
    if (seconds > 0 || result === '') result += `${seconds}s`;
    
    return result.trim();
}

/**
 * Fonctions spécifiques pour actualiser les données
 */
function refreshHealthData() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            // Mise à jour des indicateurs de santé du cluster
            document.getElementById('cluster-total-servers').textContent = data.cluster_health.total_servers;
            document.getElementById('cluster-healthy-servers').textContent = data.cluster_health.healthy_servers;
            document.getElementById('cluster-unhealthy-servers').textContent = data.cluster_health.unhealthy_servers;
            document.getElementById('cluster-avg-load').textContent = (data.cluster_health.average_load * 100).toFixed(1) + '%';
            document.getElementById('cluster-avg-latency').textContent = data.cluster_health.average_latency.toFixed(1) + ' ms';
            
            // Mise à jour des indicateurs de santé par serveur
            const serversContainer = document.getElementById('servers-health-container');
            if (serversContainer) {
                serversContainer.innerHTML = '';
                
                for (const [server, details] of Object.entries(data.servers)) {
                    const serverCard = document.createElement('div');
                    serverCard.className = 'col-md-6 col-xl-4 mb-4';
                    
                    const healthClass = details.healthy ? 'success' : 'danger';
                    const healthStatus = details.healthy ? 'En ligne' : 'Hors ligne';
                    
                    serverCard.innerHTML = `
                        <div class="card border-${healthClass}">
                            <div class="card-header bg-${healthClass} bg-opacity-10 d-flex justify-content-between">
                                <h5 class="card-title mb-0">${server}</h5>
                                <span class="badge bg-${healthClass}">${healthStatus}</span>
                            </div>
                            <div class="card-body">
                                <div class="d-flex justify-content-between mb-2">
                                    <span>Charge:</span>
                                    <span>${(details.load * 100).toFixed(1)}%</span>
                                </div>
                                <div class="progress mb-3">
                                    <div class="progress-bar 
                                    ${details.load < 0.5 ? 'bg-success' : details.load < 0.8 ? 'bg-warning' : 'bg-danger'}" 
                                    role="progressbar" style="width: ${details.load * 100}%" 
                                    aria-valuenow="${details.load * 100}" aria-valuemin="0" aria-valuemax="100">
                                    </div>
                                </div>
                                <div class="d-flex justify-content-between mb-2">
                                    <span>Latence:</span>
                                    <span>${details.latency_ms.toFixed(1)} ms</span>
                                </div>
                                <div class="d-flex justify-content-between">
                                    <span>Modèles:</span>
                                    <span>${details.models.length}</span>
                                </div>
                            </div>
                            <div class="card-footer">
                                <a href="/servers/${encodeURIComponent(server)}" class="btn btn-sm btn-primary">Détails</a>
                            </div>
                        </div>
                    `;
                    
                    serversContainer.appendChild(serverCard);
                }
            }
            
            // Mettre à jour le timestamp
            document.getElementById('last-refresh').textContent = new Date().toLocaleTimeString();
        })
        .catch(error => {
            console.error('Erreur lors de l\'actualisation des données de santé:', error);
            NotificationSystem.error('Erreur lors de l\'actualisation des données de santé');
        });
}

function refreshServersList() {
    fetch('/api/servers')
        .then(response => response.json())
        .then(data => {
            const serversTable = document.getElementById('serversTable');
            if (!serversTable) return;
            
            const tbody = serversTable.querySelector('tbody');
            tbody.innerHTML = '';
            
            if (data.servers && data.servers.length > 0) {
                data.servers.forEach(server => {
                    const tr = document.createElement('tr');
                    
                    // Mise à jour des lignes du tableau
                    tr.innerHTML = `
                        <td>${server.address}</td>
                        <td>
                            <span class="badge bg-${server.healthy ? 'success' : 'danger'}">
                                ${server.healthy ? 'En ligne' : 'Hors ligne'}
                            </span>
                        </td>
                        <td>
                            <div class="progress">
                                <div class="progress-bar 
                                ${server.load < 0.5 ? 'bg-success' : server.load < 0.8 ? 'bg-warning' : 'bg-danger'}" 
                                role="progressbar" style="width: ${server.load * 100}%" 
                                aria-valuenow="${server.load * 100}" aria-valuemin="0" aria-valuemax="100">
                                    ${(server.load * 100).toFixed(0)}%
                                </div>
                            </div>
                        </td>
                        <td>${server.latency_ms.toFixed(1)} ms</td>
                        <td>
                            ${server.models.slice(0, 3).map(model => 
                                `<span class="badge bg-primary">${model}</span>`
                            ).join(' ')}
                            ${server.models.length > 3 ? 
                                `<span class="badge bg-secondary">+${server.models.length - 3}</span>` : ''}
                        </td>
                        <td>
                            <div class="btn-group btn-group-sm">
                                <button type="button" class="btn btn-primary view-server-btn" data-server="${server.address}">
                                    <i class="fas fa-eye"></i>
                                </button>
                                <button type="button" class="btn btn-warning edit-server-btn" data-server="${server.address}">
                                    <i class="fas fa-edit"></i>
                                </button>
                                <button type="button" class="btn btn-danger remove-server-btn" data-server="${server.address}">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </td>
                    `;
                    
                    tbody.appendChild(tr);
                });
                
                // Réattacher les gestionnaires d'événements
                attachServerButtonHandlers();
            } else {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center">Aucun serveur disponible</td></tr>';
            }
            
            // Mise à jour des statistiques
            if (data.stats) {
                document.getElementById('total-servers').textContent = data.stats.total;
                document.getElementById('online-servers').textContent = data.stats.online;
                document.getElementById('offline-servers').textContent = data.stats.offline;
            }
            
            // Mettre à jour le timestamp
            document.getElementById('last-refresh').textContent = new Date().toLocaleTimeString();
        })
        .catch(error => {
            console.error('Erreur lors de l\'actualisation de la liste des serveurs:', error);
            NotificationSystem.error('Erreur lors de l\'actualisation de la liste des serveurs');
        });
}

function attachServerButtonHandlers() {
    // Gestionnaires pour les boutons d'action
    document.querySelectorAll('.view-server-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const server = this.getAttribute('data-server');
            showServerDetails(server);
        });
    });
    
    document.querySelectorAll('.edit-server-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const server = this.getAttribute('data-server');
            // Logique pour éditer un serveur
        });
    });
    
    document.querySelectorAll('.remove-server-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const server = this.getAttribute('data-server');
            document.getElementById('deleteServerAddress').textContent = server;
            const modal = new bootstrap.Modal(document.getElementById('deleteServerModal'));
            modal.show();
        });
    });
}

function refreshModelsList() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const modelsContainer = document.getElementById('models-container');
            if (!modelsContainer) return;
            
            modelsContainer.innerHTML = '';
            
            if (data.models && data.models.length > 0) {
                data.models.forEach(model => {
                    const modelCard = document.createElement('div');
                    modelCard.className = 'col-md-6 col-xl-4 mb-4';
                    
                    modelCard.innerHTML = `
                        <div class="card model-card h-100">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h5 class="card-title mb-0">${model.name}</h5>
                                <span class="badge bg-${model.available ? 'success' : 'warning'}">
                                    ${model.available ? 'Disponible' : 'Non disponible'}
                                </span>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <small class="text-muted">Taille:</small>
                                    <div>${formatBytes(model.size)}</div>
                                </div>
                                <div class="mb-3">
                                    <small class="text-muted">Version:</small>
                                    <div>${model.version || 'N/A'}</div>
                                </div>
                                <div>
                                    <small class="text-muted">Serveurs:</small>
                                    <div>
                                        ${model.servers.map(server => 
                                            `<span class="badge bg-primary">${server}</span>`
                                        ).join(' ')}
                                    </div>
                                </div>
                            </div>
                            <div class="card-footer">
                                <a href="/models/${encodeURIComponent(model.name)}" class="btn btn-sm btn-primary">
                                    <i class="fas fa-info-circle"></i> Détails
                                </a>
                                <button class="btn btn-sm btn-success ms-1" onclick="testModel('${model.name}')">
                                    <i class="fas fa-play"></i> Tester
                                </button>
                            </div>
                        </div>
                    `;
                    
                    modelsContainer.appendChild(modelCard);
                });
            } else {
                modelsContainer.innerHTML = '<div class="col-12"><div class="alert alert-info">Aucun modèle disponible</div></div>';
            }
            
            // Mettre à jour le timestamp
            document.getElementById('last-refresh').textContent = new Date().toLocaleTimeString();
        })
        .catch(error => {
            console.error('Erreur lors de l\'actualisation de la liste des modèles:', error);
            NotificationSystem.error('Erreur lors de l\'actualisation de la liste des modèles');
        });
}

function testModel(modelName) {
    window.location.href = `/playground?model=${encodeURIComponent(modelName)}`;
}