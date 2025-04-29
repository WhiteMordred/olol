"""
Modèles de données pour l'API du proxy Ollama.

Ce module définit les structures de données utilisées pour les requêtes
et les réponses de l'API du proxy.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GenerateRequest:
    """Requête pour l'endpoint /api/generate."""
    model: str
    prompt: str
    stream: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResponse:
    """Réponse de l'endpoint /api/generate."""
    model: str
    response: str
    done: bool = True
    context: List[int] = field(default_factory=list)
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@dataclass
class ChatMessage:
    """Message pour une conversation chat."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    images: List[str] = field(default_factory=list)


@dataclass
class ChatRequest:
    """Requête pour l'endpoint /api/chat."""
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Réponse de l'endpoint /api/chat."""
    model: str
    message: ChatMessage
    done: bool = True
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@dataclass
class EmbeddingsRequest:
    """Requête pour l'endpoint /api/embeddings."""
    model: str
    prompt: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingsResponse:
    """Réponse de l'endpoint /api/embeddings."""
    model: str
    embedding: List[float]
    

@dataclass
class ModelContextInfo:
    """Informations sur le contexte d'un modèle."""
    current: int = 4096
    max: int = 8192
    

@dataclass
class ModelInfo:
    """Informations sur un modèle."""
    name: str
    size: Optional[int] = None
    modified_at: Optional[datetime] = None
    version: Optional[str] = None
    available: bool = True
    servers: List[str] = field(default_factory=list)
    context: ModelContextInfo = field(default_factory=ModelContextInfo)
    

@dataclass
class ServerInfo:
    """Informations sur un serveur."""
    address: str
    healthy: bool = False
    load: float = 0.0
    models: List[str] = field(default_factory=list)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class StatusResponse:
    """Réponse de l'endpoint /api/status."""
    timestamp: float
    server_time: str
    proxy_uptime: int
    active_requests: int
    total_requests: int
    distributed_available: bool = False
    distributed_enabled: bool = False
    server_count: int = 0
    server_addresses: List[str] = field(default_factory=list)


@dataclass
class ModelsResponse:
    """Réponse de l'endpoint /api/models."""
    timestamp: float
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    model_count: int = 0
    

@dataclass
class ServersResponse:
    """Réponse de l'endpoint /api/servers."""
    servers: Dict[str, ServerInfo] = field(default_factory=dict)
    count: int = 0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class TransferRequest:
    """Requête pour l'endpoint /api/transfer."""
    model: str
    source: str
    target: str


@dataclass
class TransferResponse:
    """Réponse de l'endpoint /api/transfer."""
    success: bool = True
    message: Optional[str] = None
    error: Optional[str] = None
    
    
# Fonctions utilitaires pour convertir entre les formats de requêtes/réponses
def dict_to_generate_request(data: Dict) -> GenerateRequest:
    """Convertit un dictionnaire en GenerateRequest."""
    return GenerateRequest(
        model=data.get('model', ''),
        prompt=data.get('prompt', ''),
        stream=data.get('stream', False),
        options=data.get('options', {})
    )


def dict_to_chat_request(data: Dict) -> ChatRequest:
    """Convertit un dictionnaire en ChatRequest."""
    messages = []
    for msg_dict in data.get('messages', []):
        messages.append(ChatMessage(
            role=msg_dict.get('role', 'user'),
            content=msg_dict.get('content', ''),
            images=msg_dict.get('images', [])
        ))
    
    return ChatRequest(
        model=data.get('model', ''),
        messages=messages,
        stream=data.get('stream', False),
        options=data.get('options', {})
    )


def dict_to_embeddings_request(data: Dict) -> EmbeddingsRequest:
    """Convertit un dictionnaire en EmbeddingsRequest."""
    return EmbeddingsRequest(
        model=data.get('model', ''),
        prompt=data.get('prompt', ''),
        options=data.get('options', {})
    )