"""Statistics tracking for Ollama proxy server."""

import threading
import time
from typing import Dict, Any

# Statistics state management
stats_lock = threading.Lock()
request_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "generate_requests": 0,
    "chat_requests": 0,
    "embedding_requests": 0,
    "server_stats": {},
    "start_time": time.time()
}


def update_request_stats(request_type: str, increment: bool = True) -> None:
    """Update request statistics.
    
    Args:
        request_type: Type of request ('chat', 'generate', 'embedding')
        increment: True to increment, False to decrement (for active requests)
    """
    with stats_lock:
        # Always increment total
        if increment:
            request_stats["total_requests"] += 1
            request_stats[f"{request_type}_requests"] += 1
            
        # Update active count
        delta = 1 if increment else -1
        request_stats["active_requests"] += delta
        
        # Ensure we don't go negative
        if request_stats["active_requests"] < 0:
            request_stats["active_requests"] = 0


def update_server_stats(server_address: str, stats_data: Dict[str, Any]) -> None:
    """Update statistics for a specific server.
    
    Args:
        server_address: Address of the server
        stats_data: Dictionary of statistics to update
    """
    with stats_lock:
        if server_address not in request_stats["server_stats"]:
            request_stats["server_stats"][server_address] = {}
            
        # Update with new data
        request_stats["server_stats"][server_address].update(stats_data)
        # Add timestamp of last update
        request_stats["server_stats"][server_address]["last_update"] = time.time()


def get_stats_snapshot() -> Dict[str, Any]:
    """Get a snapshot of the current statistics.
    
    Returns:
        Dictionary containing statistics
    """
    with stats_lock:
        # Return a copy to avoid modification while being used
        return request_stats.copy()


def reset_stats() -> None:
    """Reset all statistics except for server_stats and start_time."""
    with stats_lock:
        server_stats = request_stats["server_stats"].copy()
        start_time = request_stats["start_time"]
        
        # Reset counters
        request_stats.update({
            "total_requests": 0,
            "active_requests": 0,
            "generate_requests": 0,
            "chat_requests": 0,
            "embedding_requests": 0,
            "server_stats": server_stats,
            "start_time": start_time
        })