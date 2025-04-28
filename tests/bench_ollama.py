import asyncio
import statistics
import time

from src.ollama_async_client import AsyncOllamaClient


async def run_benchmark(num_requests=100):
    client = AsyncOllamaClient()
    times = []
    
    try:
        for i in range(num_requests):
            start = time.time()
            await client.run_model("llama2", "Simple test prompt")
            end = time.time()
            times.append(end - start)
            
        print(f"Average response time: {statistics.mean(times):.3f}s")
        print(f"Median response time: {statistics.median(times):.3f}s")
        print(f"95th percentile: {statistics.quantiles(times, n=20)[-1]:.3f}s")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(run_benchmark())