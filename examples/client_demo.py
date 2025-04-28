#!/usr/bin/env python
"""Simple example demonstrating how to use the OLOL library to talk to Ollama."""

import argparse
import asyncio
import sys

from olol import AsyncOllamaClient
from olol.sync import OllamaClient


def sync_example(host: str, port: int, model: str, prompt: str) -> None:
    """Run a synchronous client example.
    
    Args:
        host: Server hostname
        port: Server port
        model: Model name to use
        prompt: Text prompt to send
    """
    print(f"Running sync client with {model} on {host}:{port}")
    client = OllamaClient(host=host, port=port)
    
    try:
        print("Prompt:", prompt)
        print("Response:", end=" ", flush=True)
        
        for response in client.generate(model, prompt):
            if not response.done:
                print(response.response, end="", flush=True)
            else:
                print(f"\nCompletion took {response.total_duration}ms")
    finally:
        client.close()


async def async_example(host: str, port: int, model: str, prompt: str) -> None:
    """Run an asynchronous client example.
    
    Args:
        host: Server hostname
        port: Server port
        model: Model name to use
        prompt: Text prompt to send
    """
    print(f"Running async client with {model} on {host}:{port}")
    client = AsyncOllamaClient(host=host, port=port)
    
    try:
        print("Prompt:", prompt)
        print("Response:", end=" ", flush=True)
        
        async for response in client.generate(model, prompt):
            if not response.done:
                print(response.response, end="", flush=True)
            else:
                print(f"\nCompletion took {response.total_duration}ms")
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="OLOL Client Demo")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--model", default="llama2", help="Model name")
    parser.add_argument("--prompt", default="Explain what a distributed LLM system is", 
                      help="Text prompt")
    parser.add_argument("--async", dest="async_mode", action="store_true",
                      help="Use async client")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code
    """
    args = parse_args()
    
    try:
        if args.async_mode:
            asyncio.run(async_example(args.host, args.port, args.model, args.prompt))
        else:
            sync_example(args.host, args.port, args.model, args.prompt)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())