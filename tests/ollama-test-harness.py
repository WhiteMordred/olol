import argparse
import json
import logging
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ollama_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OllamaClusterTest:
    def __init__(self, api_endpoint, verbose=False):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.verbose = verbose
        self.test_results = {}
        self.sessions = {}
        
    def log(self, message):
        if self.verbose:
            logger.info(message)
        
    def run_all_tests(self):
        """Run all test cases and report results"""
        self.test_results = {}
        
        # Basic connectivity test
        self.test_results["connectivity"] = self.test_connectivity()
        
        # Model listing test
        models_result = self.test_list_models()
        self.test_results["list_models"] = models_result
        
        if models_result["success"]:
            available_models = models_result["data"]["models"]
            if available_models:
                # Get the first model for testing
                test_model = available_models[0]["name"]
                self.log(f"Using model {test_model} for tests")
                
                # Run single inference test
                self.test_results["single_inference"] = self.test_single_inference(test_model)
                
                # Run chat session test
                self.test_results["chat_session"] = self.test_chat_session(test_model)
                
                # Run concurrent requests test
                self.test_results["concurrent_requests"] = self.test_concurrent_requests(test_model)
                
                # Run server distribution test
                self.test_results["server_distribution"] = self.test_server_distribution(test_model)
            else:
                logger.warning("No models available for testing")
        
        # Print summary
        self.print_summary()
        
        return self.test_results
        
    def test_connectivity(self):
        """Test basic connectivity to the API proxy"""
        self.log("Testing connectivity to API proxy...")
        
        try:
            # Try to get server status
            response = requests.get(f"{self.api_endpoint}/api/status", timeout=5)
            response.raise_for_status()
            
            return {
                "success": True,
                "message": "Successfully connected to API proxy",
                "data": response.json()
            }
        except Exception as e:
            logger.error(f"Connectivity test failed: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to connect to API proxy: {str(e)}",
                "data": None
            }
    
    def test_list_models(self):
        """Test model listing API"""
        self.log("Testing model listing...")
        
        try:
            response = requests.get(f"{self.api_endpoint}/api/models", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            model_count = len(data.get("models", []))
            
            return {
                "success": True,
                "message": f"Successfully listed {model_count} models",
                "data": data
            }
        except Exception as e:
            logger.error(f"List models test failed: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to list models: {str(e)}",
                "data": None
            }
    
    def test_single_inference(self, model_name):
        """Test a single inference request"""
        self.log(f"Testing single inference with model {model_name}...")
        
        prompt = "Explain what a gRPC proxy is in one sentence."
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_endpoint}/api/generate",
                json={"model": model_name, "prompt": prompt},
                timeout=30
            )
            response.raise_for_status()
            
            elapsed_time = time.time() - start_time
            data = response.json()
            
            return {
                "success": True,
                "message": f"Inference completed in {elapsed_time:.2f} seconds",
                "data": {
                    "response": data.get("response", "")[:100] + "..." if data.get("response") else "",
                    "elapsed_time": elapsed_time
                }
            }
        except Exception as e:
            logger.error(f"Single inference test failed: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to run inference: {str(e)}",
                "data": None
            }
    
    def test_chat_session(self, model_name):
        """Test chat session creation and message exchange"""
        self.log(f"Testing chat session with model {model_name}...")
        
        session_id = str(uuid.uuid4())
        messages = [
            {"role": "user", "content": "Hello, who are you?"}
        ]
        
        try:
            # Send first message
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_endpoint}/api/chat",
                json={"model": model_name, "messages": messages, "session_id": session_id},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            elapsed_time_1 = time.time() - start_time
            
            # Store for later use
            self.sessions[session_id] = data
            
            # Send second message referencing first
            messages.append({"role": "assistant", "content": data.get("message", {}).get("content", "")})
            messages.append({"role": "user", "content": "Can you tell me more about yourself?"})
            
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_endpoint}/api/chat",
                json={"model": model_name, "messages": messages, "session_id": session_id},
                timeout=30
            )
            response.raise_for_status()
            
            elapsed_time_2 = time.time() - start_time
            data2 = response.json()
            
            # Clean up session
            delete_response = requests.delete(
                f"{self.api_endpoint}/api/sessions/{session_id}",
                timeout=10
            )
            
            return {
                "success": True,
                "message": "Chat session test completed successfully",
                "data": {
                    "session_id": session_id,
                    "first_response": data.get("message", {}).get("content", "")[:100] + "...",
                    "second_response": data2.get("message", {}).get("content", "")[:100] + "...",
                    "first_time": elapsed_time_1,
                    "second_time": elapsed_time_2
                }
            }
        except Exception as e:
            logger.error(f"Chat session test failed: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to test chat session: {str(e)}",
                "data": None
            }
    
    def test_concurrent_requests(self, model_name, num_requests=5):
        """Test multiple concurrent requests"""
        self.log(f"Testing {num_requests} concurrent requests with model {model_name}...")
        
        prompt = "What is the capital city of France? Answer in one word."
        results = []
        errors = []
        
        def make_request():
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_endpoint}/api/generate",
                    json={"model": model_name, "prompt": prompt},
                    timeout=60
                )
                response.raise_for_status()
                
                elapsed_time = time.time() - start_time
                return {"success": True, "elapsed_time": elapsed_time}
            except Exception as e:
                logger.error(f"Concurrent request failed: {str(e)}")
                return {"success": False, "error": str(e)}
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            
            for future in tqdm(futures, desc="Running concurrent requests"):
                result = future.result()
                if result["success"]:
                    results.append(result["elapsed_time"])
                else:
                    errors.append(result["error"])
        
        if not results:
            return {
                "success": False,
                "message": f"All {num_requests} concurrent requests failed",
                "data": {"errors": errors}
            }
        
        avg_time = statistics.mean(results) if results else 0
        return {
            "success": True,
            "message": f"Completed {len(results)} of {num_requests} concurrent requests",
            "data": {
                "successful_requests": len(results),
                "failed_requests": len(errors),
                "average_time": avg_time,
                "min_time": min(results) if results else 0,
                "max_time": max(results) if results else 0,
                "errors": errors[:3]  # Show only first few errors
            }
        }
    
    def test_server_distribution(self, model_name, num_requests=10):
        """Test if requests are distributed across servers"""
        self.log(f"Testing server distribution with {num_requests} requests...")
        
        # First get cluster status to see available servers
        try:
            status_response = requests.get(f"{self.api_endpoint}/api/status", timeout=5)
            status_response.raise_for_status()
            status_data = status_response.json()
            
            servers = status_data.get("servers", {})
            if not servers:
                return {
                    "success": False,
                    "message": "No servers found in status response",
                    "data": None
                }
            
            online_servers = [server for server, info in servers.items() 
                             if info.get("status") == "online"]
            
            if len(online_servers) <= 1:
                return {
                    "success": True,
                    "message": f"Only {len(online_servers)} server(s) available, distribution test skipped",
                    "data": {"servers": online_servers}
                }
            
            # Now make multiple requests and track which server handled them
            prompt = "Generate a random number between 1 and 100."
            server_loads = {}
            
            for i in range(num_requests):
                try:
                    # Add a unique identifier to prompt to prevent caching
                    unique_prompt = f"{prompt} Request ID: {uuid.uuid4()}"
                    
                    response = requests.post(
                        f"{self.api_endpoint}/api/generate",
                        json={"model": model_name, "prompt": unique_prompt},
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    # Check response headers or content for server information
                    # This assumes your proxy adds some server identification
                    # If not available directly, make a status call after each request
                    
                    status_after = requests.get(f"{self.api_endpoint}/api/status", timeout=5)
                    status_after.raise_for_status()
                    after_data = status_after.json()
                    
                    # Track server loads to see if they change
                    for server, info in after_data.get("servers", {}).items():
                        current_load = info.get("current_load", 0)
                        if server not in server_loads:
                            server_loads[server] = []
                        server_loads[server].append(current_load)
                    
                    time.sleep(1)  # Small delay between requests
                    
                except Exception as e:
                    logger.error(f"Request {i} failed: {str(e)}")
            
            # Analyze distribution
            load_changes = {}
            for server, loads in server_loads.items():
                if len(loads) > 1:
                    # Calculate if load changed during test
                    load_changes[server] = max(loads) - min(loads)
            
            # Check if multiple servers showed activity
            active_servers = [server for server, change in load_changes.items() if change > 0]
            
            if len(active_servers) > 1:
                return {
                    "success": True,
                    "message": f"Requests distributed across {len(active_servers)} servers",
                    "data": {
                        "active_servers": active_servers,
                        "load_changes": load_changes
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "Requests not distributed across multiple servers",
                    "data": {
                        "active_servers": active_servers,
                        "load_changes": load_changes
                    }
                }
            
        except Exception as e:
            logger.error(f"Server distribution test failed: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to test server distribution: {str(e)}",
                "data": None
            }
    
    def print_summary(self):
        """Print a summary of test results"""
        print("\n" + "="*80)
        print("OLLAMA CLUSTER TEST RESULTS SUMMARY")
        print("="*80)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{status} - {test_name}: {result['message']}")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            print(f"\n{test_name.upper()}:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result['message']}")
            
            if result["data"]:
                if isinstance(result["data"], dict):
                    for key, value in result["data"].items():
                        if isinstance(value, (list, dict)):
                            print(f"  {key}: {json.dumps(value, indent=2)[:200]}...")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  Data: {result['data']}")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Ollama Cluster Test Harness")
    parser.add_argument("--api-endpoint", default="http://localhost:11434", 
                        help="Ollama API endpoint URL")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--save-report", action="store_true",
                        help="Save test results to JSON file")
    parser.add_argument("--concurrent-requests", type=int, default=5,
                        help="Number of concurrent requests for load testing")
                        
    args = parser.parse_args()
    
    print(f"Starting Ollama Cluster Test against {args.api_endpoint}")
    test_runner = OllamaClusterTest(args.api_endpoint, args.verbose)
    
    start_time = time.time()
    results = test_runner.run_all_tests()
    total_time = time.time() - start_time
    
    print(f"\nAll tests completed in {total_time:.2f} seconds")
    
    if args.save_report:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"ollama_test_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "api_endpoint": args.api_endpoint,
                "duration": total_time,
                "results": results
            }, f, indent=2)
        print(f"Test report saved to {filename}")

if __name__ == "__main__":
    main()
