"""Protocol buffer definitions for OLOL."""

# Import the pb2 modules for easy access
from .. import ollama_pb2, ollama_pb2_grpc

# Make them available at package level
__all__ = ["ollama_pb2", "ollama_pb2_grpc"]