#!/usr/bin/env python3
"""
Ollama Cluster Benchmark Tool

This script benchmarks the performance of an Ollama cluster by running
multiple types of inference workloads and measuring response times,
throughput, and reliability.
"""

import json
import logging
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ollama_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OllamaBenchmark:
    def __init__(self, api_endpoint, model, output_dir="benchmark_results"):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.model = model
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Test prompts with varying complexity
        self.prompts = {
            "short": "What is the capital of France?",
            "medium": "Explain how gRPC works and its benefits over traditional REST APIs.",
            "long": "Write a detailed analysis of distributed systems architectures, comparing microservices, monoliths, and serverless approaches. Include considerations for scalability, fault tolerance, and development complexity."
        }
    
    def run_single_benchmark(self, prompt_type="medium", num_requests=10):
        """Run a benchmark with a single prompt type"""
        prompt = self.prompts[prompt_type]
        
        logger.info(f"Running {prompt_type} prompt benchmark with {num_requests} requests")
        
        latencies = []
        errors = 0
        
        with tqdm(total=num_requests, desc=f"{prompt_type} prompts") as pbar:
            for i in range(num_requests):
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        f"{self.api_endpoint}/api/generate",
                        json={"model": self.model, "prompt": prompt},
                        timeout=120  # 2 minute timeout for long prompts
                    )
                    response.raise_for_status()
                    
                    end_time = time.time()
                    latency = end_time - start_time
                    latencies.append(latency)
                    
                    # Check if we got a valid response
                    data = response.json()
                    if "response" not in data or not data["response"]:
                        logger.warning(f"Empty response for {prompt_type} prompt")
                
                except Exception as e:
                    logger.error(f"Request failed: {str(e)}")
                    errors += 1
                
                pbar.update(1)
                
                # Small delay between requests to avoid overwhelming the server
                time.sleep(0.5)
        
        # Calculate statistics
        if latencies:
            stats = {
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "mean_latency": statistics.mean(latencies),
                "median_latency": statistics.median(latencies),
                "p90_latency": np.percentile(latencies, 90) if len(latencies) >= 10 else None,
                "p95_latency": np.percentile(latencies, 95) if len(latencies) >= 20 else None,
                "p99_latency": np.percentile(latencies, 99) if len(latencies) >= 100 else None,
                "std_dev": statistics.stdev(latencies) if len(latencies) >= 2 else 0,
                "throughput": len(latencies) / sum(latencies) if sum(latencies) > 0 else 0,
                "success_rate": (num_requests - errors) / num_requests if num_requests > 0 else 0,
                "error_count": errors,
                "sample_size": len(latencies)
            }
        else:
            stats = {
                "error": "All requests failed",
                "error_count": errors,
                "success_rate": 0
            }
        
        # Save raw latencies for plotting
        stats["raw_latencies"] = latencies
        
        return stats
    
    def run_concurrent_benchmark(self, concurrency_levels=[1, 2, 5, 10], prompt_type="short", requests_per_level=10):
        """Test how the system handles different levels of concurrency"""
        logger.info(f"Running concurrency benchmark with levels: {concurrency_levels}")
        
        prompt = self.prompts[prompt_type]
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            latencies = []
            errors = 0
            start_time = time.time()
            
            def make_request():
                try:
                    req_start = time.time()
                    response = requests.post(
                        f"{self.api_endpoint}/api/generate",
                        json={"model": self.model, "prompt": prompt},
                        timeout=120
                    )
                    response.raise_for_status()
                    req_end = time.time()
                    return {"success": True, "latency": req_end - req_start}
                except Exception as e:
                    logger.error(f"Request failed: {str(e)}")
                    return {"success": False, "error": str(e)}
            
            # Use ThreadPoolExecutor to run concurrent requests
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                
                # Submit all requests
                for _ in range(requests_per_level):
                    futures.append(executor.submit(make_request))
                
                # Process results as they complete
                for future in tqdm(futures, desc=f"Concurrency {concurrency}"):
                    result = future.result()
                    if result["success"]:
                        latencies.append(result["latency"])
                    else:
                        errors += 1
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            if latencies:
                results[concurrency] = {
                    "min_latency": min(latencies),
                    "max_latency": max(latencies),
                    "mean_latency": statistics.mean(latencies),
                    "median_latency": statistics.median(latencies),
                    "p90_latency": np.percentile(latencies, 90) if len(latencies) >= 10 else None,
                    "throughput": len(latencies) / total_time if total_time > 0 else 0,
                    "requests_per_second": concurrency / statistics.mean(latencies) if latencies else 0,
                    "success_rate": (requests_per_level - errors) / requests_per_level if requests_per_level > 0 else 0,
                    "error_count": errors,
                    "total_time": total_time,
                    "raw_latencies": latencies
                }
            else:
                results[concurrency] = {
                    "error": "All requests failed",
                    "error_count": errors,
                    "success_rate": 0,
                    "total_time": total_time
                }
        
        return results
    
    def run_session_consistency_test(self, num_sessions=5, messages_per_session=3):
        """Test chat session consistency across multiple exchanges"""
        logger.info(f"Running session consistency test with {num_sessions} sessions, {messages_per_session} messages each")
        
        session_results = {}
        
        for session_idx in range(num_sessions):
            session_id = str(uuid.uuid4())
            session_results[session_id] = {"messages": [], "latencies": [], "errors": 0}
            
            # Initial message
            messages = [{"role": "user", "content": "Hello, can you remember a random 5-digit number for me?"}]
            
            for msg_idx in range(messages_per_session):
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        f"{self.api_endpoint}/api/chat",
                        json={"model": self.model, "messages": messages, "session_id": session_id},
                        timeout=60
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    end_time = time.time()
                    
                    # Store response
                    assistant_msg = data.get("message", {}).get("content", "")
                    session_results[session_id]["messages"].append({
                        "role": "assistant",
                        "content": assistant_msg[:100] + "..." if len(assistant_msg) > 100 else assistant_msg
                    })
                    
                    # Store latency
                    latency = end_time - start_time
                    session_results[session_id]["latencies"].append(latency)
                    
                    # Add to messages for next round
                    messages.append({"role": "assistant", "content": assistant_msg})
                    
                    if msg_idx < messages_per_session - 1:
                        # Add next user message based on previous exchange
                        if msg_idx == 0:
                            next_msg = "Thank you. Can you repeat that number back to me?"
                        else:
                            next_msg = "Is that the same number you told me initially?"
                        
                        messages.append({"role": "user", "content": next_msg})
                        session_results[session_id]["messages"].append({
                            "role": "user",
                            "content": next_msg
                        })
                
                except Exception as e:
                    logger.error(f"Session {session_id}, message {msg_idx} failed: {str(e)}")
                    session_results[session_id]["errors"] += 1
                    # Try to continue with the session
                    messages.append({"role": "user", "content": "Let's continue. What were we talking about?"})
            
            # Clean up session
            try:
                requests.delete(f"{self.api_endpoint}/api/sessions/{session_id}", timeout=10)
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {str(e)}")
        
        # Analyze for consistency
        consistent_sessions = 0
        for session_id, data in session_results.items():
            if data["errors"] == 0 and len(data["messages"]) >= 2*messages_per_session - 1:
                # Simple check - if we completed all messages without errors
                consistent_sessions += 1
        
        consistency_rate = consistent_sessions / num_sessions if num_sessions > 0 else 0
        
        return {
            "session_count": num_sessions,
            "messages_per_session": messages_per_session,
            "consistent_sessions": consistent_sessions,
            "consistency_rate": consistency_rate,
            "session_details": session_results
        }
    
    def run_all_benchmarks(self):
        """Run all benchmark tests and collect results"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results = {
            "timestamp": timestamp,
            "api_endpoint": self.api_endpoint,
            "model": self.model
        }
        
        # 1. Single request performance for different prompt types
        logger.info("Running single request benchmarks for different prompt types")
        single_results = {}
        for prompt_type in self.prompts.keys():
            single_results[prompt_type] = self.run_single_benchmark(prompt_type, num_requests=10)
        self.results["single_request"] = single_results
        
        # 2. Concurrency test
        logger.info("Running concurrency benchmark")
        self.results["concurrency"] = self.run_concurrent_benchmark(
            concurrency_levels=[1, 3, 5, 10],
            prompt_type="short",
            requests_per_level=10
        )
        
        # 3. Session consistency test
        logger.info("Running session consistency test")
        self.results["session_consistency"] = self.run_session_consistency_test(
            num_sessions=5,
            messages_per_session=3
        )
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        self.generate_visualizations()
        
        return self.results
    
    def save_results(self):
        """Save benchmark results to JSON file"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create a copy of results without raw latencies for JSON file
        results_for_json = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_for_json[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        results_for_json[key][subkey] = {k: v for k, v in subvalue.items() 
                                                        if k != "raw_latencies"}
                    else:
                        results_for_json[key][subkey] = subvalue
            else:
                results_for_json[key] = value
        
        # Save to file
        filename = os.path.join(self.output_dir, f"benchmark_{self.model}_{timestamp}.json")
        with open(filename, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        return filename
    
    def generate_visualizations(self):
        """Generate visualizations of benchmark results"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 1. Latency distribution for different prompt types
        plt.figure(figsize=(12, 6))
        for prompt_type, data in self.results.get("single_request", {}).items():
            if "raw_latencies" in data and data["raw_latencies"]:
                plt.hist(data["raw_latencies"], alpha=0.5, label=prompt_type)