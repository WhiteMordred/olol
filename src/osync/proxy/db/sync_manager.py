"""
Module de synchronisation entre TinyDB et les structures en mémoire.

Ce module fournit un mécanisme permettant de synchroniser efficacement les données
entre la base de données persistante (TinyDB) et les structures en mémoire vive,
offrant ainsi à la fois persistance et performances.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Set, TypeVar, Generic
from datetime import datetime

from .database import get_db

# Configuration du logging
logger = logging.getLogger(__name__)

# Type générique pour les tables
T = TypeVar('T')


class SyncManager:
    """
    Gestionnaire de synchronisation entre la base de données TinyDB et les structures en mémoire.
    
    Cette classe assure que les données en RAM restent synchronisées avec TinyDB,
    tout en optimisant les performances en minimisant les accès disque.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Implémentation du pattern Singleton pour SyncManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SyncManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialise le gestionnaire de synchronisation."""
        # Éviter la réinitialisation du singleton
        if self._initialized:
            return
            
        self.db = get_db()
        self._ram_cache = {}  # Cache en mémoire par table
        self._locks = {}      # Verrous par table
        self._initialized = True
        
        logger.info("SyncManager initialisé")
    
    def _get_table_lock(self, table_name: str) -> threading.RLock:
        """
        Récupère ou crée un verrou pour une table spécifique.
        
        Args:
            table_name: Nom de la table
            
        Returns:
            Verrou pour la table spécifiée
        """
        if table_name not in self._locks:
            self._locks[table_name] = threading.RLock()
        return self._locks[table_name]
    
    def load_initial_state(self) -> None:
        """
        Charge l'état initial depuis TinyDB vers la RAM.
        Cette méthode est appelée au démarrage du système.
        """
        # Liste des tables connues à charger
        tables = ["servers", "models", "server_stats", "request_stats", "config", "inference_queue"]
        
        for table_name in tables:
            try:
                with self._get_table_lock(table_name):
                    # Initialiser le cache en mémoire pour cette table
                    if table_name not in self._ram_cache:
                        self._ram_cache[table_name] = {}
                    
                    # Récupérer tous les documents de la table
                    docs = self.db.get_all(table_name)
                    
                    # Stocker dans le cache en mémoire
                    for doc in docs:
                        doc_id = doc.doc_id if hasattr(doc, 'doc_id') else doc.get('_id')
                        if doc_id:
                            self._ram_cache[table_name][doc_id] = doc
            
            except Exception as e:
                logger.error(f"Erreur lors du chargement initial de la table {table_name}: {e}")
        
        logger.info(f"État initial chargé: {len(self._ram_cache)} tables en mémoire")
    
    def write_and_sync(self, table_name: str, data: Dict[str, Any], doc_id: Optional[int] = None) -> int:
        """
        Écrit dans TinyDB et synchronise avec la RAM.
        
        Args:
            table_name: Nom de la table
            data: Données à écrire
            doc_id: ID du document (si mise à jour)
            
        Returns:
            ID du document inséré ou mis à jour
        """
        with self._get_table_lock(table_name):
            # Assurer que le cache pour cette table existe
            if table_name not in self._ram_cache:
                self._ram_cache[table_name] = {}
            
            # Écriture dans TinyDB
            if doc_id is not None:
                # Mise à jour d'un document existant
                self.db.update(table_name, data, lambda q: q.doc_id == doc_id)
                result_id = doc_id
            else:
                # Insertion d'un nouveau document
                result_id = self.db.insert(table_name, data)
            
            # Synchronisation avec le cache RAM
            self._ram_cache[table_name][result_id] = {**data, "doc_id": result_id}
            
            return result_id
    
    def read_from_ram(self, table_name: str, query_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Lit les données depuis la RAM.
        
        Args:
            table_name: Nom de la table
            query_func: Fonction de filtrage facultative qui prend un document et renvoie un booléen
            
        Returns:
            Liste des documents correspondants
        """
        with self._get_table_lock(table_name):
            # Si table non en cache, la charger
            if table_name not in self._ram_cache:
                self._ram_cache[table_name] = {}
                docs = self.db.get_all(table_name)
                for doc in docs:
                    doc_id = doc.doc_id if hasattr(doc, 'doc_id') else doc.get('_id')
                    if doc_id:
                        self._ram_cache[table_name][doc_id] = doc
            
            # Récupérer tous les documents de la table en mémoire
            all_docs = list(self._ram_cache[table_name].values())
            
            # Appliquer le filtre si fourni
            if query_func:
                return [doc for doc in all_docs if query_func(doc)]
            
            return all_docs
    
    def force_sync(self, table_name: Optional[str] = None) -> None:
        """
        Force la synchronisation entre TinyDB et la RAM.
        
        Args:
            table_name: Nom de la table spécifique à synchroniser, ou None pour toutes les tables
        """
        if table_name:
            # Synchroniser une table spécifique
            with self._get_table_lock(table_name):
                self._ram_cache[table_name] = {}
                docs = self.db.get_all(table_name)
                for doc in docs:
                    doc_id = doc.doc_id if hasattr(doc, 'doc_id') else doc.get('_id')
                    if doc_id:
                        self._ram_cache[table_name][doc_id] = doc
                logger.debug(f"Table {table_name} resynchronisée avec {len(docs)} documents")
        else:
            # Synchroniser toutes les tables connues
            tables = self.db.db.tables()
            for table in tables:
                self.force_sync(table)
    
    def delete_and_sync(self, table_name: str, doc_id: Optional[int] = None, query_func: Optional[Callable] = None) -> List[int]:
        """
        Supprime des documents et synchronise la RAM.
        
        Args:
            table_name: Nom de la table
            doc_id: ID du document à supprimer (exclusif avec query_func)
            query_func: Fonction de requête pour supprimer plusieurs documents
            
        Returns:
            Liste des IDs des documents supprimés
        """
        with self._get_table_lock(table_name):
            # Assurer que le cache pour cette table existe
            if table_name not in self._ram_cache:
                self._ram_cache[table_name] = {}
                self.force_sync(table_name)
            
            # Suppression dans TinyDB
            removed_ids = []
            if doc_id is not None:
                # Supprimer un document spécifique
                success = self.db.delete_by_id(table_name, doc_id)
                if success:
                    removed_ids = [doc_id]
            elif query_func:
                # Supprimer par requête
                removed_ids = self.db.delete(table_name, query_func)
            
            # Synchroniser avec le cache RAM
            for r_id in removed_ids:
                if r_id in self._ram_cache[table_name]:
                    del self._ram_cache[table_name][r_id]
            
            return removed_ids
    
    def update_cache_only(self, table_name: str, doc_id: int, data: Dict[str, Any]) -> None:
        """
        Met à jour uniquement le cache en mémoire sans écrire dans TinyDB.
        Utile pour les mises à jour temporaires ou les données volatiles.
        
        Args:
            table_name: Nom de la table
            doc_id: ID du document
            data: Données à mettre à jour
        """
        with self._get_table_lock(table_name):
            if table_name not in self._ram_cache:
                self._ram_cache[table_name] = {}
                self.force_sync(table_name)
            
            if doc_id in self._ram_cache[table_name]:
                self._ram_cache[table_name][doc_id].update(data)
            else:
                self._ram_cache[table_name][doc_id] = {**data, "doc_id": doc_id}


# Singleton pour accéder facilement au gestionnaire de synchronisation
def get_sync_manager() -> SyncManager:
    """
    Récupère l'instance unique du gestionnaire de synchronisation.
    
    Returns:
        Instance du SyncManager
    """
    return SyncManager()