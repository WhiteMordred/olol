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
import time
import json
import os

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
        self._change_log = {} # Journal des modifications par table
        self._batch_mode = False
        self._batch_operations = []
        self._auto_sync_interval = 300  # 5 minutes par défaut
        self._last_backup_time = time.time()
        self._backup_dir = "backups"
        
        # Créer le répertoire de sauvegarde s'il n'existe pas
        if not os.path.exists(self._backup_dir):
            try:
                os.makedirs(self._backup_dir)
            except Exception as e:
                logger.warning(f"Impossible de créer le répertoire de sauvegarde: {e}")
        
        self._initialized = True
        
        # Démarrer le thread d'auto-synchronisation
        self._start_auto_sync_thread()
        
        logger.info("SyncManager initialisé")
    
    def _start_auto_sync_thread(self):
        """
        Démarre un thread pour synchroniser périodiquement les données
        et effectuer des sauvegardes.
        """
        def auto_sync_task():
            while True:
                try:
                    time.sleep(self._auto_sync_interval)
                    self.force_sync()
                    
                    # Vérifier si une sauvegarde est nécessaire (toutes les 24h)
                    current_time = time.time()
                    if current_time - self._last_backup_time > 86400:  # 24h en secondes
                        self.create_backup()
                        self._last_backup_time = current_time
                        
                except Exception as e:
                    logger.error(f"Erreur dans le thread d'auto-synchronisation: {e}")
        
        sync_thread = threading.Thread(target=auto_sync_task, daemon=True)
        sync_thread.start()
        logger.debug("Thread d'auto-synchronisation démarré")
    
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
                            
                    # Initialiser le journal des modifications
                    if table_name not in self._change_log:
                        self._change_log[table_name] = []
            
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
            
            # Si en mode batch, simplement enregistrer l'opération pour exécution ultérieure
            if self._batch_mode:
                operation = {
                    'type': 'write',
                    'table': table_name,
                    'data': data,
                    'doc_id': doc_id
                }
                self._batch_operations.append(operation)
                
                # Attribuer un ID temporaire si c'est une nouvelle insertion
                if doc_id is None:
                    temp_id = -len(self._batch_operations)  # ID négatif temporaire
                    return temp_id
                return doc_id
            
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
            
            # Enregistrer l'opération dans le journal des modifications
            self._log_change(table_name, 'write', result_id, data)
            
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
            
            # Si en mode batch, simplement enregistrer l'opération
            if self._batch_mode:
                operation = {
                    'type': 'delete',
                    'table': table_name,
                    'doc_id': doc_id,
                    'query_func': query_func
                }
                self._batch_operations.append(operation)
                
                # Simuler la suppression dans le cache en mémoire
                if doc_id is not None and doc_id in self._ram_cache[table_name]:
                    del self._ram_cache[table_name][doc_id]
                    return [doc_id]
                return []
            
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
                    
                # Enregistrer l'opération dans le journal des modifications
                self._log_change(table_name, 'delete', r_id, None)
            
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
                
    def start_batch(self) -> None:
        """
        Démarre un mode de traitement par lots pour regrouper plusieurs opérations.
        Les opérations ne sont pas exécutées immédiatement mais stockées pour une
        exécution ultérieure via commit_batch().
        """
        self._batch_mode = True
        self._batch_operations = []
        logger.debug("Mode de traitement par lots démarré")
        
    def commit_batch(self) -> Dict[str, Any]:
        """
        Exécute toutes les opérations mises en file d'attente pendant le mode par lots.
        
        Returns:
            Dictionnaire contenant les résultats des opérations par lots
        """
        if not self._batch_mode:
            return {"status": "error", "message": "Pas en mode traitement par lots"}
        
        results = {
            "writes": {},
            "deletes": {},
            "errors": []
        }
        
        # Désactiver le mode batch pour éviter la récursion
        self._batch_mode = False
        batch_ops = self._batch_operations.copy()
        self._batch_operations = []
        
        # Exécuter chaque opération
        for op in batch_ops:
            try:
                if op['type'] == 'write':
                    result_id = self.write_and_sync(
                        op['table'], 
                        op['data'], 
                        op['doc_id']
                    )
                    
                    if op['table'] not in results["writes"]:
                        results["writes"][op['table']] = []
                    
                    results["writes"][op['table']].append({
                        "doc_id": result_id,
                        "status": "success"
                    })
                    
                elif op['type'] == 'delete':
                    removed_ids = self.delete_and_sync(
                        op['table'], 
                        op['doc_id'], 
                        op['query_func']
                    )
                    
                    if op['table'] not in results["deletes"]:
                        results["deletes"][op['table']] = []
                    
                    results["deletes"][op['table']].append({
                        "removed_ids": removed_ids,
                        "status": "success"
                    })
            
            except Exception as e:
                results["errors"].append({
                    "operation": op,
                    "error": str(e)
                })
                logger.error(f"Erreur lors de l'exécution d'une opération par lots: {e}")
        
        logger.info(f"Traitement par lots terminé: {len(batch_ops)} opérations exécutées")
        return results
    
    def cancel_batch(self) -> None:
        """
        Annule le mode de traitement par lots en cours et supprime toutes les opérations
        en attente.
        """
        self._batch_mode = False
        nb_ops = len(self._batch_operations)
        self._batch_operations = []
        logger.debug(f"Mode de traitement par lots annulé: {nb_ops} opérations supprimées")
        
    def _log_change(self, table_name: str, operation_type: str, doc_id: int, data: Optional[Dict[str, Any]]) -> None:
        """
        Enregistre une modification dans le journal des changements.
        
        Args:
            table_name: Nom de la table modifiée
            operation_type: Type d'opération ('write', 'delete', etc.)
            doc_id: ID du document concerné
            data: Données associées à l'opération (peut être None pour 'delete')
        """
        if table_name not in self._change_log:
            self._change_log[table_name] = []
            
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_type,
            "doc_id": doc_id
        }
        
        self._change_log[table_name].append(change_record)
        
        # Limiter la taille du journal à 1000 entrées par table
        if len(self._change_log[table_name]) > 1000:
            self._change_log[table_name] = self._change_log[table_name][-1000:]
    
    def get_changes(self, table_name: Optional[str] = None, since: Optional[datetime] = None) -> Dict[str, List[Dict]]:
        """
        Récupère le journal des modifications pour une ou toutes les tables.
        
        Args:
            table_name: Nom de la table spécifique (ou None pour toutes)
            since: Date à partir de laquelle filtrer les modifications
            
        Returns:
            Journal des modifications filtré
        """
        result = {}
        
        # Déterminer les tables à inclure
        tables_to_include = [table_name] if table_name else list(self._change_log.keys())
        
        for table in tables_to_include:
            if table in self._change_log:
                if since:
                    # Convertir la date en texte pour la comparaison
                    since_iso = since.isoformat()
                    result[table] = [
                        change for change in self._change_log[table]
                        if change["timestamp"] >= since_iso
                    ]
                else:
                    result[table] = self._change_log[table].copy()
        
        return result
        
    def create_backup(self) -> str:
        """
        Crée une sauvegarde de la base de données actuelle.
        
        Returns:
            Chemin du fichier de sauvegarde créé
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{self._backup_dir}/db_backup_{timestamp}.json"
        
        try:
            # Créer un dictionnaire avec toutes les données
            backup_data = {}
            
            for table_name in self._ram_cache.keys():
                backup_data[table_name] = list(self._ram_cache[table_name].values())
            
            # Écrire le fichier de sauvegarde
            with open(backup_filename, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Sauvegarde créée avec succès: {backup_filename}")
            return backup_filename
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la sauvegarde: {e}")
            return ""
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restaure la base de données à partir d'un fichier de sauvegarde.
        
        Args:
            backup_path: Chemin du fichier de sauvegarde
            
        Returns:
            True si la restauration a réussi, False sinon
        """
        if not os.path.exists(backup_path):
            logger.error(f"Fichier de sauvegarde introuvable: {backup_path}")
            return False
        
        try:
            # Lire le fichier de sauvegarde
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Restaurer chaque table
            for table_name, docs in backup_data.items():
                # Effacer la table existante
                self.db.truncate(table_name)
                
                # Insérer les documents restaurés
                for doc in docs:
                    # Supprimer l'ID de document pour éviter les conflits
                    doc_copy = doc.copy()
                    doc_id = None
                    if 'doc_id' in doc_copy:
                        doc_id = doc_copy.pop('doc_id')
                    
                    # Réinsérer le document
                    self.db.insert(table_name, doc_copy, doc_id)
            
            # Resynchroniser le cache RAM avec la base de données restaurée
            self.force_sync()
            
            logger.info(f"Base de données restaurée avec succès depuis: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la restauration de la sauvegarde: {e}")
            return False
    
    def set_auto_sync_interval(self, seconds: int) -> None:
        """
        Définit l'intervalle d'auto-synchronisation en secondes.
        
        Args:
            seconds: Intervalle en secondes entre deux synchronisations automatiques
        """
        self._auto_sync_interval = max(10, seconds)  # Minimum 10 secondes
        logger.debug(f"Intervalle d'auto-synchronisation défini à {self._auto_sync_interval} secondes")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère des statistiques sur le gestionnaire de synchronisation.
        
        Returns:
            Dictionnaire contenant diverses métriques
        """
        stats = {
            "tables": {},
            "total_documents": 0,
            "batch_mode": self._batch_mode,
            "pending_batch_operations": len(self._batch_operations) if self._batch_mode else 0,
            "auto_sync_interval": self._auto_sync_interval,
            "last_backup_time": datetime.fromtimestamp(self._last_backup_time).isoformat() if self._last_backup_time else None,
        }
        
        # Statistiques par table
        for table_name, docs in self._ram_cache.items():
            table_stats = {
                "document_count": len(docs),
                "change_log_entries": len(self._change_log.get(table_name, [])),
            }
            stats["tables"][table_name] = table_stats
            stats["total_documents"] += len(docs)
        
        return stats


# Singleton pour accéder facilement au gestionnaire de synchronisation
def get_sync_manager() -> SyncManager:
    """
    Récupère l'instance unique du gestionnaire de synchronisation.
    
    Returns:
        Instance du SyncManager
    """
    return SyncManager()