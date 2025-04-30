"""
Module de persistance des données utilisant TinyDB comme moteur de stockage.

Ce module fournit une couche d'abstraction pour la persistance des données,
permettant potentiellement de remplacer TinyDB par une autre solution de stockage
à l'avenir sans impacter le reste du code.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Callable
from datetime import datetime
from tinydb import TinyDB, Query, where
from tinydb.table import Document, Table
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

# Configuration du logging
logger = logging.getLogger(__name__)

# Type générique pour les identifiants
T = TypeVar('T')


class DatabaseManager:
    """
    Gestionnaire de base de données qui abstrait l'utilisation de TinyDB.
    
    Cette classe fournit une interface CRUD simple pour interagir avec une base de données,
    tout en cachant les détails d'implémentation spécifiques à TinyDB.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialise le gestionnaire de base de données.
        
        Args:
            db_path: Chemin vers le fichier de base de données. Si None, utilise un chemin par défaut.
        """
        if db_path is None:
            # Utiliser un chemin par défaut dans le répertoire utilisateur
            home_dir = os.path.expanduser("~")
            db_dir = os.path.join(home_dir, ".osync")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "osync.json")
        
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Initialiser la base de données avec mise en cache pour de meilleures performances
        self.db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
        logger.info(f"Base de données initialisée: {db_path}")
    
    def get_table(self, table_name: str) -> Table:
        """
        Récupère une table spécifique de la base de données.
        
        Args:
            table_name: Nom de la table à récupérer
            
        Returns:
            La table demandée
        """
        return self.db.table(table_name)
    
    def insert(self, table_name: str, data: Dict[str, Any]) -> int:
        """
        Insère des données dans une table spécifique.
        
        Args:
            table_name: Nom de la table cible
            data: Dictionnaire des données à insérer
            
        Returns:
            ID du document inséré
        """
        # Ajouter un timestamp de création si non présent
        if 'created_at' not in data:
            data['created_at'] = datetime.now().isoformat()
        
        table = self.get_table(table_name)
        doc_id = table.insert(data)
        logger.debug(f"Document inséré dans {table_name} avec ID {doc_id}")
        return doc_id
    
    def insert_multiple(self, table_name: str, data_list: List[Dict[str, Any]]) -> List[int]:
        """
        Insère plusieurs documents dans une table spécifique.
        
        Args:
            table_name: Nom de la table cible
            data_list: Liste de dictionnaires à insérer
            
        Returns:
            Liste des IDs des documents insérés
        """
        # Ajouter des timestamps de création si non présents
        now = datetime.now().isoformat()
        for data in data_list:
            if 'created_at' not in data:
                data['created_at'] = now
        
        table = self.get_table(table_name)
        doc_ids = table.insert_multiple(data_list)
        logger.debug(f"Insertion multiple dans {table_name}: {len(doc_ids)} documents")
        return doc_ids
    
    def get_all(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Récupère tous les documents d'une table.
        
        Args:
            table_name: Nom de la table
            
        Returns:
            Liste de tous les documents
        """
        table = self.get_table(table_name)
        return table.all()
    
    def get_by_id(self, table_name: str, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupère un document par son ID.
        
        Args:
            table_name: Nom de la table
            doc_id: ID du document
            
        Returns:
            Document trouvé ou None si non trouvé
        """
        table = self.get_table(table_name)
        doc = table.get(doc_id=doc_id)
        return doc
    
    def search(self, table_name: str, query_func: Callable) -> List[Dict[str, Any]]:
        """
        Recherche des documents avec une fonction de requête personnalisée.
        
        Args:
            table_name: Nom de la table
            query_func: Fonction qui prend un objet Query et renvoie une requête
            
        Returns:
            Liste des documents correspondant à la requête
            
        Example:
            db.search('servers', lambda q: q.address == 'localhost:50051')
        """
        table = self.get_table(table_name)
        q = Query()
        return table.search(query_func(q))
    
    def search_one(self, table_name: str, query_func: Callable) -> Optional[Dict[str, Any]]:
        """
        Recherche un seul document correspondant à la requête.
        
        Args:
            table_name: Nom de la table
            query_func: Fonction qui prend un objet Query et renvoie une requête
            
        Returns:
            Premier document correspondant ou None
        """
        results = self.search(table_name, query_func)
        return results[0] if results else None
    
    def update(self, table_name: str, data: Dict[str, Any], query_func: Callable) -> List[int]:
        """
        Met à jour les documents correspondant à la requête.
        
        Args:
            table_name: Nom de la table
            data: Données à mettre à jour
            query_func: Fonction qui prend un objet Query et renvoie une requête
            
        Returns:
            Liste des IDs des documents mis à jour
        """
        # Ajouter un timestamp de mise à jour
        data['updated_at'] = datetime.now().isoformat()
        
        table = self.get_table(table_name)
        q = Query()
        result = table.update(data, query_func(q))
        logger.debug(f"Mis à jour dans {table_name}: {len(result)} documents")
        return result
    
    def upsert(self, table_name: str, data: Dict[str, Any], query_func: Callable) -> List[int]:
        """
        Met à jour un document s'il existe, sinon l'insère.
        
        Args:
            table_name: Nom de la table
            data: Données à insérer ou mettre à jour
            query_func: Fonction qui prend un objet Query et renvoie une requête
            
        Returns:
            Liste des IDs des documents modifiés
        """
        # Ajouter des timestamps appropriés
        now = datetime.now().isoformat()
        if 'updated_at' not in data:
            data['updated_at'] = now
        
        table = self.get_table(table_name)
        q = Query()
        result = table.upsert(data, query_func(q))
        
        action = "mis à jour ou inséré"
        logger.debug(f"Document {action} dans {table_name}")
        return result
    
    def delete(self, table_name: str, query_func: Callable) -> List[int]:
        """
        Supprime les documents correspondant à la requête.
        
        Args:
            table_name: Nom de la table
            query_func: Fonction qui prend un objet Query et renvoie une requête
            
        Returns:
            Liste des IDs des documents supprimés
        """
        table = self.get_table(table_name)
        q = Query()
        result = table.remove(query_func(q))
        logger.debug(f"Suppression dans {table_name}: {len(result)} documents")
        return result
    
    def delete_by_id(self, table_name: str, doc_id: int) -> bool:
        """
        Supprime un document par son ID.
        
        Args:
            table_name: Nom de la table
            doc_id: ID du document à supprimer
            
        Returns:
            True si supprimé, False sinon
        """
        table = self.get_table(table_name)
        result = table.remove(doc_ids=[doc_id])
        success = len(result) > 0
        if success:
            logger.debug(f"Document avec ID {doc_id} supprimé de {table_name}")
        return success
    
    def count(self, table_name: str, query_func: Optional[Callable] = None) -> int:
        """
        Compte le nombre de documents correspondant à une requête.
        
        Args:
            table_name: Nom de la table
            query_func: Fonction de requête facultative
            
        Returns:
            Nombre de documents
        """
        table = self.get_table(table_name)
        if query_func:
            q = Query()
            return len(table.search(query_func(q)))
        return len(table)
    
    def clear_table(self, table_name: str) -> None:
        """
        Vide une table spécifique.
        
        Args:
            table_name: Nom de la table à vider
        """
        table = self.get_table(table_name)
        table.truncate()
        logger.debug(f"Table {table_name} vidée")
    
    def close(self) -> None:
        """Ferme la connexion à la base de données."""
        self.db.close()
        logger.info("Connexion à la base de données fermée")


# Singleton pour accéder facilement à la base de données depuis n'importe où dans l'application
_db_instance = None

def get_db(db_path: str = None) -> DatabaseManager:
    """
    Récupère l'instance unique du gestionnaire de base de données.
    
    Args:
        db_path: Chemin facultatif vers le fichier de base de données
        
    Returns:
        Instance du DatabaseManager
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance