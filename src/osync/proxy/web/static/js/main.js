/**
 * Ollama Sync - Main JavaScript
 * Dark Theme Version
 */

// Fonctions utilitaires
const OLOL = {
    // Couleurs du thème sombre pour les graphiques
    chartColors: {
        primary: '#0d6efd',
        secondary: '#6c757d',
        success: '#198754',
        danger: '#dc3545',
        warning: '#ffc107',
        info: '#0dcaf0',
        background: '#121212',
        surface: '#1e1e1e',
        border: '#333333',
        text: '#f1f1f1',
        textSecondary: '#a0a0a0'
    },
    
    // Configuration commune pour les graphiques en thème sombre
    commonChartOptions: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#f1f1f1'
                }
            },
            tooltip: {
                backgroundColor: 'rgba(30, 30, 30, 0.9)',
                titleColor: '#f1f1f1',
                bodyColor: '#f1f1f1',
                borderColor: '#333333',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)',
                    borderColor: 'rgba(255, 255, 255, 0.2)'
                },
                ticks: {
                    color: '#a0a0a0'
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)',
                    borderColor: 'rgba(255, 255, 255, 0.2)'
                },
                ticks: {
                    color: '#a0a0a0'
                }
            }
        }
    },
    
    // Gestionnaire de notifications Toast
    showToast: function(title, message, type = 'info') {
        const toast = document.getElementById('notificationToast');
        const toastTitle = document.getElementById('toastTitle');
        const toastMessage = document.getElementById('toastMessage');
        const toastTime = document.getElementById('toastTime');
        
        if (toast && toastTitle && toastMessage && toastTime) {
            // Définir l'icône en fonction du type
            let icon = 'info-circle';
            let colorClass = 'text-info';
            
            switch (type) {
                case 'success':
                    icon = 'check-circle';
                    colorClass = 'text-success';
                    break;
                case 'warning':
                    icon = 'exclamation-circle';
                    colorClass = 'text-warning';
                    break;
                case 'error':
                    icon = 'times-circle';
                    colorClass = 'text-danger';
                    break;
            }
            
            // Utiliser le système de traduction pour les titres standards
            let translatedTitle = title;
            if (window.I18n && ['success', 'error', 'warning', 'info'].includes(title.toLowerCase())) {
                translatedTitle = window.I18n.t(title.toLowerCase());
            }
            
            toastTitle.innerHTML = `<i class="fas fa-${icon} me-1 ${colorClass}"></i> ${translatedTitle}`;
            toastMessage.textContent = message;
            toastTime.textContent = window.I18n ? window.I18n.t('just_now') : 'À l\'instant';
            
            // Initialiser et afficher le toast
            const toastInstance = new bootstrap.Toast(toast);
            toastInstance.show();
        }
    },
    
    // Formatage des nombres
    formatNumber: function(num) {
        return new Intl.NumberFormat().format(num);
    },
    
    // Formatage de la mémoire
    formatMemory: function(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // Formatage du temps
    formatTime: function(seconds) {
        if (seconds < 60) {
            return `${seconds.toFixed(2)} s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            return `${minutes} min ${remainingSeconds.toFixed(0)} s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const remainingMinutes = Math.floor((seconds % 3600) / 60);
            return `${hours} h ${remainingMinutes} min`;
        }
    },
    
    // Actualiser automatiquement les données
    setupAutoRefresh: function(callback, interval = 30000) {
        const autoRefreshBtn = document.getElementById('autoRefreshToggle');
        if (!autoRefreshBtn) return;
        
        let refreshInterval;
        
        autoRefreshBtn.addEventListener('click', function() {
            this.classList.toggle('active');
            
            if (this.classList.contains('active')) {
                // Activer l'actualisation automatique
                const autoOnText = window.I18n ? window.I18n.t('auto_refresh_on') : 'Auto (ON)';
                this.innerHTML = '<i class="fas fa-sync-alt fa-spin me-2"></i>' + autoOnText;
                callback(); // Actualiser immédiatement
                refreshInterval = setInterval(callback, interval);
                
                const autoRefreshTitle = window.I18n ? window.I18n.t('auto_refresh') : 'Actualisation automatique';
                const autoRefreshMsg = window.I18n ? window.I18n.t('auto_refresh_enabled') : 'L\'actualisation automatique est activée';
                OLOL.showToast(autoRefreshTitle, autoRefreshMsg, 'info');
            } else {
                // Désactiver l'actualisation automatique
                const autoOffText = window.I18n ? window.I18n.t('auto_refresh_off') : 'Auto (OFF)';
                this.innerHTML = '<i class="fas fa-sync-alt me-2"></i>' + autoOffText;
                clearInterval(refreshInterval);
            }
        });
    },
    
    // Animer les compteurs
    animateCounter: function(element, targetValue, duration = 1000) {
        if (!element) return;
        
        const startValue = parseInt(element.textContent) || 0;
        const increment = (targetValue - startValue) / (duration / 16);
        let currentValue = startValue;
        
        const animate = () => {
            currentValue += increment;
            
            if ((increment >= 0 && currentValue >= targetValue) || 
                (increment < 0 && currentValue <= targetValue)) {
                element.textContent = OLOL.formatNumber(targetValue);
            } else {
                element.textContent = OLOL.formatNumber(Math.floor(currentValue));
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    },
    
    // Créer un graphique en ligne avec thème sombre
    createLineChart: function(ctx, labels, datasets, options = {}) {
        if (!ctx) return null;
        
        // Fusionner les options par défaut et personnalisées
        const chartOptions = {...OLOL.commonChartOptions, ...options};
        
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets.map(dataset => ({
                    ...dataset,
                    borderColor: dataset.borderColor || OLOL.chartColors.primary,
                    backgroundColor: dataset.backgroundColor || 'rgba(13, 110, 253, 0.1)',
                    borderWidth: dataset.borderWidth || 2,
                    pointRadius: dataset.pointRadius || 3,
                    pointHoverRadius: dataset.pointHoverRadius || 5,
                    tension: dataset.tension || 0.3
                }))
            },
            options: chartOptions
        });
    },
    
    // Créer un graphique en barres avec thème sombre
    createBarChart: function(ctx, labels, datasets, options = {}) {
        if (!ctx) return null;
        
        // Fusionner les options par défaut et personnalisées
        const chartOptions = {...OLOL.commonChartOptions, ...options};
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: datasets.map(dataset => ({
                    ...dataset,
                    borderColor: dataset.borderColor || OLOL.chartColors.primary,
                    backgroundColor: dataset.backgroundColor || 'rgba(13, 110, 253, 0.7)',
                    borderWidth: dataset.borderWidth || 1,
                    borderRadius: dataset.borderRadius || 4
                }))
            },
            options: chartOptions
        });
    },
    
    // Créer un graphique en donut avec thème sombre
    createDoughnutChart: function(ctx, labels, data, colors = [], options = {}) {
        if (!ctx) return null;
        
        // Couleurs par défaut du thème sombre si non spécifiées
        const defaultColors = [
            OLOL.chartColors.primary,
            OLOL.chartColors.success,
            OLOL.chartColors.warning,
            OLOL.chartColors.danger,
            OLOL.chartColors.info,
            OLOL.chartColors.secondary
        ];
        
        // Utiliser les couleurs spécifiées ou les couleurs par défaut
        const chartColors = colors.length > 0 ? colors : defaultColors;
        
        // Options de base pour les graphiques en donut
        const baseOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: OLOL.chartColors.text,
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(30, 30, 30, 0.9)',
                    titleColor: OLOL.chartColors.text,
                    bodyColor: OLOL.chartColors.text,
                    borderColor: OLOL.chartColors.border,
                    borderWidth: 1
                }
            },
            cutout: '70%'
        };
        
        // Fusionner les options par défaut et personnalisées
        const chartOptions = {...baseOptions, ...options};
        
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: chartColors,
                    borderColor: OLOL.chartColors.background,
                    borderWidth: 2,
                    hoverOffset: 10
                }]
            },
            options: chartOptions
        });
    },
    
    // Initialiser les datatables avec style sombre
    initDatatable: function(tableId, options = {}) {
        const tableElement = document.getElementById(tableId);
        if (!tableElement) return null;
        
        // Options par défaut pour le thème sombre
        const defaultOptions = {
            language: {
                // Utiliser la langue courante du système de traduction
                url: window.I18n?.currentLanguage === 'fr' 
                    ? '//cdn.datatables.net/plug-ins/1.11.5/i18n/fr-FR.json'
                    : '//cdn.datatables.net/plug-ins/1.11.5/i18n/en-GB.json'
            },
            pageLength: 10,
            lengthMenu: [5, 10, 25, 50],
            responsive: true,
            initComplete: function() {
                // Appliquer des styles personnalisés après l'initialisation
                document.querySelectorAll(`#${tableId}_wrapper .dataTables_paginate .page-link`).forEach(el => {
                    el.classList.add('bg-dark', 'text-light', 'border-dark');
                });
                
                document.querySelectorAll(`#${tableId}_wrapper select, #${tableId}_wrapper input`).forEach(el => {
                    el.classList.add('bg-dark', 'text-light', 'border-secondary');
                });
                
                document.querySelectorAll(`#${tableId}_wrapper label`).forEach(el => {
                    el.classList.add('text-light');
                });
            }
        };
        
        // Fusionner les options par défaut avec les options personnalisées
        const finalOptions = {...defaultOptions, ...options};
        
        // Initialiser DataTable avec les options fusionnées
        return new DataTable(`#${tableId}`, finalOptions);
    }
};

// Exposer la fonction showToast globalement pour le système de traduction
window.showToast = OLOL.showToast;

// Initialisation au chargement du document
document.addEventListener('DOMContentLoaded', function() {
    // Appliquer des styles spécifiques au thème sombre aux éléments interactifs
    document.querySelectorAll('input, select, textarea').forEach(element => {
        element.classList.add('bg-dark', 'text-light', 'border-secondary');
    });
    
    // Initialiser les tooltips Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
    
    // Gestionnaire de changement de langue
    function setupLanguageSwitcher() {
        const langSwitcher = document.getElementById('languageSwitcher');
        
        if (langSwitcher && window.I18n) {
            // Mise à jour de l'affichage du sélecteur de langue
            const currentLangText = langSwitcher.querySelector('.current-language');
            if (currentLangText) {
                currentLangText.textContent = window.I18n.availableLanguages[window.I18n.currentLanguage];
            }
            
            // Ajouter des écouteurs d'événements aux options de langue
            const langOptions = langSwitcher.querySelectorAll('.dropdown-item[data-lang]');
            langOptions.forEach(option => {
                const lang = option.getAttribute('data-lang');
                option.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    // Changer la langue
                    window.I18n.changeLanguage(lang).then(() => {
                        // Notifier l'utilisateur
                        const langChangedTitle = window.I18n.t('language_changed');
                        const langChangedMsg = window.I18n.t('language_changed_message');
                        OLOL.showToast(langChangedTitle, langChangedMsg, 'success');
                        
                        // Mise à jour de l'affichage du sélecteur
                        if (currentLangText) {
                            currentLangText.textContent = window.I18n.availableLanguages[lang];
                        }
                        
                        // Déclencher un événement pour que d'autres composants puissent réagir
                        window.dispatchEvent(new CustomEvent('languageChanged', {
                            detail: { language: lang }
                        }));
                    });
                });
            });
        }
    }
    
    // Initialiser le système de traduction et le sélecteur de langue
    if (window.I18n) {
        window.I18n.init().then(() => {
            setupLanguageSwitcher();
            
            // Ajouter just_now à tous les dictionnaires si nécessaire
            if (!window.I18n.translations.fr.just_now) {
                window.I18n.translations.fr.just_now = "À l'instant";
            }
            if (!window.I18n.translations.en.just_now) {
                window.I18n.translations.en.just_now = "Just now";
            }
            
            // Ajouter les traductions pour l'auto-refresh si elles n'existent pas
            if (!window.I18n.translations.fr.auto_refresh) {
                window.I18n.translations.fr.auto_refresh = "Actualisation automatique";
                window.I18n.translations.fr.auto_refresh_enabled = "L'actualisation automatique est activée";
                window.I18n.translations.fr.auto_refresh_on = "Auto (ON)";
                window.I18n.translations.fr.auto_refresh_off = "Auto (OFF)";
            }
            if (!window.I18n.translations.en.auto_refresh) {
                window.I18n.translations.en.auto_refresh = "Auto Refresh";
                window.I18n.translations.en.auto_refresh_enabled = "Auto refresh is enabled";
                window.I18n.translations.en.auto_refresh_on = "Auto (ON)";
                window.I18n.translations.en.auto_refresh_off = "Auto (OFF)";
            }
            
            // Ajouter les traductions pour le changement de langue si elles n'existent pas
            if (!window.I18n.translations.fr.language_changed_message) {
                window.I18n.translations.fr.language_changed_message = "La langue a été changée avec succès";
            }
            if (!window.I18n.translations.en.language_changed_message) {
                window.I18n.translations.en.language_changed_message = "Language has been changed successfully";
            }
        });
    }
    
    // Écouter les changements de langue pour mettre à jour les DataTables
    window.addEventListener('i18n:languageChanged', function(e) {
        // Recharger la page après un court délai pour que l'utilisateur voie le toast
        setTimeout(() => {
            window.location.reload();
        }, 1000);
    });
});