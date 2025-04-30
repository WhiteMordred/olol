/**
 * Ollama Sync - Système de traduction (i18n)
 * Module de gestion des traductions pour l'interface web
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
        
        // Marquer comme initialisé
        this.initialized = true;
        
        return Promise.resolve();
    },
    
    /**
     * Charge les fichiers de traduction
     * @returns {Promise} - Promise résolue quand le chargement est terminé
     */
    loadTranslations: async function() {
        // Définir les langues à charger
        const langsToLoad = ['en', 'fr']; // Toujours charger l'anglais comme fallback
        
        // Charger chaque fichier de traduction
        const promises = langsToLoad.map(lang => {
            return fetch(`/static/js/i18n/${lang}.js`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Erreur de chargement du fichier de traduction ${lang}`);
                    }
                    return response.text();
                })
                .then(jsContent => {
                    // Exécuter le script pour enregistrer les traductions
                    // Le fichier de traduction doit définir sa variable sur window.I18n.translations[lang]
                    try {
                        eval(jsContent);
                    } catch (e) {
                        console.error(`Erreur dans le fichier de traduction ${lang}:`, e);
                    }
                })
                .catch(err => {
                    console.error(`Impossible de charger le fichier de traduction ${lang}:`, err);
                });
        });
        
        // Attendre que tous les fichiers soient chargés
        await Promise.all(promises);
        
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
        
        // Charger les traductions si nécessaire
        if (Object.keys(this.translations[lang]).length === 0) {
            await this.loadTranslations();
        }
        
        // Traduire la page
        this.translatePage();
        
        // Déclencher un événement pour informer l'application du changement
        window.dispatchEvent(new CustomEvent('i18n:languageChanged', {
            detail: { language: lang }
        }));
        
        return Promise.resolve();
    },
    
    /**
     * Traduit un texte dans la langue active
     * @param {string} key - Clé de traduction à rechercher
     * @param {Object} params - Paramètres à substituer dans la traduction
     * @returns {string} - Texte traduit ou clé si non trouvée
     */
    t: function(key, params = {}) {
        // Vérifier si la clé existe dans la langue actuelle
        let text = this.translations[this.currentLanguage][key];
        
        // Si non trouvée, utiliser l'anglais comme fallback
        if (!text && this.currentLanguage !== 'en') {
            text = this.translations['en'][key];
        }
        
        // Si toujours non trouvée, retourner la clé
        if (!text) {
            console.warn(`Traduction manquante pour la clé "${key}" en ${this.currentLanguage}`);
            return key;
        }
        
        // Remplacer les paramètres si présents
        if (params && Object.keys(params).length > 0) {
            Object.keys(params).forEach(param => {
                const regex = new RegExp(`{{${param}}}`, 'g');
                text = text.replace(regex, params[param]);
            });
        }
        
        return text;
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
                element.textContent = this.t(key);
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