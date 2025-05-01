/**
 * Ollama Sync - Système de traduction (i18n)
 * Module de gestion des traductions pour l'interface web
 * Version 2.0 - Architecture modulaire par page
 */

const I18n = {
    // État du module
    initialized: false,
    
    // Langue actuelle
    currentLanguage: 'en', // Langue par défaut
    
    // Langues disponibles
    availableLanguages: {
        'en': 'English',
        'fr': 'Français'
        // Ajouter d'autres langues ici au besoin
    },
    
    // Dictionnaires de traductions (remplis dynamiquement)
    translations: {
        'en': {},
        'fr': {}
    },
    
    /**
     * Détermine les sections/pages à charger en fonction de la page courante
     * @returns {Array} - Liste des sections à charger
     */
    getSectionsToLoad: function() {
        // Toujours charger les traductions communes
        const sections = ['common'];
        
        // Déterminer la page actuelle en fonction de l'URL
        const path = window.location.pathname.toLowerCase();
        
        if (path.includes('dashboard') || path === '/' || path === '') {
            sections.push('dashboard');
        }
        if (path.includes('models')) {
            sections.push('models');
        }
        if (path.includes('servers')) {
            sections.push('servers');
        }
        if (path.includes('health')) {
            sections.push('health');
        }
        if (path.includes('settings')) {
            sections.push('settings');
        }
        if (path.includes('playground')) {
            sections.push('playground');
        }
        if (path.includes('queue')) {
            sections.push('queue');
        }
        if (path.includes('terminal')) {
            sections.push('terminal');
        }
        if (path.includes('log')) {
            sections.push('log');
        }
        if (path.includes('swagger')) {
            sections.push('swagger');
        }
        
        // Si aucune section spécifique n'est détectée, charger toutes les sections
        if (sections.length === 1) {
            sections.push('dashboard', 'models', 'servers', 'health', 'settings', 
                         'playground', 'queue', 'terminal', 'log', 'swagger');
        }
        
        return sections;
    },
    
    /**
     * Initialise le système de traduction
     * @returns {Promise} - Promise résolue quand le chargement est terminé
     */
    init: async function() {
        // Tenter de récupérer la langue préférée du navigateur ou stockée
        const savedLang = localStorage.getItem('olol_language');
        
        if (savedLang && this.availableLanguages[savedLang]) {
            this.currentLanguage = savedLang;
        } else {
            // Détecter la langue du navigateur
            const browserLang = navigator.language.split('-')[0];
            if (this.availableLanguages[browserLang]) {
                this.currentLanguage = browserLang;
            }
        }
        
        // Charger les traductions pour la langue actuelle
        await this.loadTranslations();
        
        // Traduire la page au chargement
        this.translatePage();
        
        // Ajouter un écouteur d'événements pour les changements de langue
        window.addEventListener('i18n:languageChanged', () => {
            this.translatePage();
        });
        
        // Marquer comme initialisé
        this.initialized = true;
        
        // Ajouter un helper pour le débogage des traductions
        if (window.location.search.includes('debug_i18n=1')) {
            this.enableDebugMode();
        }
        
        return Promise.resolve();
    },
    
    /**
     * Active le mode de débogage pour les traductions
     */
    enableDebugMode: function() {
        document.body.classList.add('i18n-debug');
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            element.setAttribute('title', `Translation key: ${key}`);
            element.style.outline = '1px dashed orange';
        });
        console.log('Mode débogage i18n activé');
        console.log('Traductions disponibles:', this.translations);
    },
    
    /**
     * Charge les fichiers de traduction JSON pour toutes les sections nécessaires
     * @returns {Promise} - Promise résolue quand le chargement est terminé
     */
    loadTranslations: async function() {
        // Obtenir les sections à charger
        const sections = this.getSectionsToLoad();
        const languages = ['en', this.currentLanguage]; // Toujours charger l'anglais comme fallback
        
        // Créer un ensemble unique de langues à charger
        const uniqueLanguages = [...new Set(languages)];
        
        const loadPromises = [];
        
        // Pour chaque langue, charger toutes les sections nécessaires
        for (const lang of uniqueLanguages) {
            for (const section of sections) {
                const url = `/static/js/i18n/${section}/${lang}.json`;
                
                const promise = fetch(url)
                    .then(response => {
                        if (!response.ok) {
                            console.warn(`Fichier de traduction non trouvé: ${url}`);
                            return null; // Continuer même si le fichier n'existe pas
                        }
                        return response.json();
                    })
                    .then(data => {
                        if (!data) return; // Si pas de données, ne rien faire
                        
                        // Initialiser la section si nécessaire
                        if (!this.translations[lang]) {
                            this.translations[lang] = {};
                        }
                        
                        // Fusionner les traductions
                        this.translations[lang] = {
                            ...this.translations[lang],
                            ...data
                        };
                    })
                    .catch(err => {
                        console.error(`Erreur lors du chargement des traductions ${section}/${lang}:`, err);
                    });
                
                loadPromises.push(promise);
            }
        }
        
        // Attendre que toutes les traductions soient chargées
        await Promise.all(loadPromises);
        
        return Promise.resolve();
    },
    
    /**
     * Change la langue active et traduit la page
     * @param {string} lang - Code de la langue à activer
     * @returns {Promise} - Promise résolue quand le changement est terminé
     */
    changeLanguage: async function(lang) {
        // Vérifier que la langue est disponible
        if (!this.availableLanguages[lang]) {
            console.error(`Langue ${lang} non disponible`);
            return Promise.reject(`Langue ${lang} non disponible`);
        }
        
        // Mettre à jour la langue courante
        this.currentLanguage = lang;
        
        // Sauvegarder la préférence
        localStorage.setItem('olol_language', lang);
        
        // Recharger les traductions pour la nouvelle langue
        await this.loadTranslations();
        
        // Traduire la page
        this.translatePage();
        
        // Déclencher un événement pour informer l'application du changement
        window.dispatchEvent(new CustomEvent('i18n:languageChanged', {
            detail: { language: lang }
        }));
        
        return Promise.resolve();
    },
    
    /**
     * Traduit un texte dans la langue active avec support pour les clés hiérarchiques
     * @param {string} key - Clé de traduction à rechercher (peut être imbriquée avec des points)
     * @param {Object} params - Paramètres à substituer dans la traduction
     * @returns {string} - Texte traduit ou clé si non trouvée
     */
    t: function(key, params = {}) {
        // Support des clés imbriquées (ex: "app.title")
        let translation = null;
        
        // Pour la langue actuelle
        let obj = this.translations[this.currentLanguage];
        if (obj) {
            translation = this.getNestedTranslation(obj, key);
        }
        
        // Si non trouvée, utiliser l'anglais comme fallback
        if (translation === null && this.currentLanguage !== 'en') {
            obj = this.translations['en'];
            if (obj) {
                translation = this.getNestedTranslation(obj, key);
            }
        }
        
        // Si toujours non trouvée, retourner la clé
        if (translation === null) {
            console.warn(`Traduction manquante pour la clé "${key}" en ${this.currentLanguage}`);
            return key;
        }
        
        // Si la traduction est un objet (ce qui ne devrait pas arriver), retourner la clé
        if (typeof translation === 'object') {
            console.warn(`La clé "${key}" pointe vers un objet et non une chaîne de caractères`);
            return key;
        }
        
        // Remplacer les paramètres si présents
        if (params && Object.keys(params).length > 0) {
            Object.keys(params).forEach(param => {
                const regex = new RegExp(`{{${param}}}`, 'g');
                translation = translation.replace(regex, params[param]);
            });
        }
        
        return translation;
    },
    
    /**
     * Récupère une traduction depuis un objet imbriqué avec support des points dans les clés
     * @param {Object} obj - Objet de traduction
     * @param {string} key - Clé à rechercher (peut contenir des points)
     * @returns {string|null} - Traduction trouvée ou null si non trouvée
     */
    getNestedTranslation: function(obj, key) {
        // Clé simple (sans hiérarchie)
        if (!key.includes('.')) {
            return obj[key] || null;
        }
        
        // Clé imbriquée
        const parts = key.split('.');
        let current = obj;
        
        // Parcourir la hiérarchie
        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            
            // Si la partie n'existe pas ou n'est pas un objet
            if (!current[part]) {
                return null;
            }
            
            // Si c'est la dernière partie, retourner la valeur
            if (i === parts.length - 1) {
                return current[part];
            }
            
            // Sinon, continuer à descendre dans la hiérarchie
            current = current[part];
            
            // Si on arrive sur une feuille avant la fin du chemin
            if (typeof current !== 'object') {
                return null;
            }
        }
        
        return null;
    },
    
    /**
     * Traduit tous les éléments de la page avec l'attribut data-i18n
     */
    translatePage: function() {
        // Sélectionner tous les éléments avec l'attribut data-i18n
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            
            // Traduire le contenu
            if (key) {
                const translation = this.t(key);
                if (translation !== key) { // Ne mettre à jour que si la traduction est disponible
                    element.textContent = translation;
                }
            }
            
            // Traduire les attributs (data-i18n-attr="title:key")
            for (const attr of element.attributes) {
                if (attr.name.startsWith('data-i18n-attr-')) {
                    const attrName = attr.name.replace('data-i18n-attr-', '');
                    const attrKey = attr.value;
                    element.setAttribute(attrName, this.t(attrKey));
                }
            }
            
            // Traduire les placeholders
            if (element.hasAttribute('data-i18n-placeholder')) {
                const placeholderKey = element.getAttribute('data-i18n-placeholder');
                element.setAttribute('placeholder', this.t(placeholderKey));
            }
        });
        
        // Traduire les titres de page
        const pageTitle = document.querySelector('title');
        if (pageTitle && pageTitle.hasAttribute('data-i18n')) {
            const titleKey = pageTitle.getAttribute('data-i18n');
            document.title = this.t(titleKey) + ' - Ollama Sync';
        }
    }
};

// Exposer l'objet I18n globalement
window.I18n = I18n;

// Initialiser le système de traduction dès que le DOM est chargé
document.addEventListener('DOMContentLoaded', () => {
    window.I18n.init().catch(err => {
        console.error("Erreur lors de l'initialisation du système de traduction:", err);
    });
});