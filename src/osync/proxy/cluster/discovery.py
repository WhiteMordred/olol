#!/usr/bin/env python3

import asyncio
import logging
import socket
from typing import Dict, List, Optional

class OllamaServerDiscovery:
    """
    Classe responsable de la découverte des serveurs Ollama dans le réseau
    """
    
    def __init__(self, network: str = "192.168.1.0/24", port: int = 11434):
        self.network = network
        self.port = port
        self.servers = {}
        self.logger = logging.getLogger(__name__)
    
    async def discover_servers(self) -> Dict[str, dict]:
        """
        Découvre les serveurs Ollama disponibles sur le réseau
        
        Returns:
            Dict[str, dict]: Dictionnaire des serveurs avec leurs informations
        """
        self.logger.info(f"Découverte des serveurs Ollama sur {self.network}:{self.port}")
        
        # Implémentation simplifiée - dans une vraie application, 
        # il faudrait scanner le réseau et vérifier les services
        # Pour le moment, nous simulons la découverte
        
        discovered = {
            "server1": {
                "address": "192.168.1.100",
                "port": self.port,
                "status": "active"
            },
            "server2": {
                "address": "192.168.1.101",
                "port": self.port,
                "status": "active"
            }
        }
        
        self.servers = discovered
        return discovered
    
    async def check_server_health(self, server_address: str, server_port: int) -> bool:
        """
        Vérifie si un serveur Ollama est disponible et en bonne santé
        
        Args:
            server_address: L'adresse IP du serveur
            server_port: Le port du serveur
            
        Returns:
            bool: True si le serveur est disponible et en bonne santé, False sinon
        """
        try:
            # Créer une connexion socket pour vérifier si le serveur répond
            with socket.create_connection((server_address, server_port), timeout=2.0):
                self.logger.info(f"Serveur {server_address}:{server_port} est disponible")
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            self.logger.warning(f"Serveur {server_address}:{server_port} n'est pas disponible")
            return False
    
    async def monitor_servers(self, interval: int = 60):
        """
        Surveille périodiquement les serveurs pour vérifier leur santé
        
        Args:
            interval: Intervalle en secondes entre les vérifications
        """
        while True:
            self.logger.info(f"Vérification de la santé des serveurs toutes les {interval} secondes")
            for server_id, info in self.servers.items():
                is_healthy = await self.check_server_health(info['address'], info['port'])
                self.servers[server_id]['status'] = "active" if is_healthy else "inactive"
            
            await asyncio.sleep(interval)