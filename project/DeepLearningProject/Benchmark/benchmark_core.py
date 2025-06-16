"""
benchmark_core.py - Core functions for language model benchmarking

This module contains the main functions for conducting benchmark tests with Ollama.
"""

import time
import random
import requests
from ollama_server import check_ollama_server, start_ollama_server
from model_manager import check_model_exists, load_model

# Global variables for benchmark status
_benchmark_running = False
# Cache for already checked models, prevents duplicate checks
_checked_models = set()
# Unique ID for the current benchmark execution
_execution_id = None

def benchmark_model(api_url, model_name, prompt, max_tokens=100, temperature=0.7, request_timeout=120, retry_timeout=300):
    """Performs a benchmark for a single model with a prompt."""
    request_data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    start_time = time.time()
    try:
        # Timeout for the request
        response = requests.post(f"{api_url}/generate", json=request_data, timeout=request_timeout)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            generation_time = end_time - start_time
            tokens_per_second = result.get('eval_count', 0) / generation_time if generation_time > 0 else 0
            
            return {
                "success": True,
                "response": result.get('response', ''),
                "total_duration": result.get('total_duration', 0) / 1_000_000_000,  # ns to s
                "load_duration": result.get('load_duration', 0) / 1_000_000_000,    # ns to s
                "eval_count": result.get('eval_count', 0),
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second
            }
        else:
            return {
                "success": False,
                "error": f"Error: {response.status_code} - {response.text}"
            }
    except requests.exceptions.Timeout:
        print(f"    ⚠️ Timeout for request to {model_name}. Trying with increased timeout...")
        try:
            # Second attempt with even higher timeout
            response = requests.post(f"{api_url}/generate", json=request_data, timeout=retry_timeout)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generation_time = end_time - start_time
                tokens_per_second = result.get('eval_count', 0) / generation_time if generation_time > 0 else 0
                
                return {
                    "success": True,
                    "response": result.get('response', ''),
                    "total_duration": result.get('total_duration', 0) / 1_000_000_000,
                    "load_duration": result.get('load_duration', 0) / 1_000_000_000,
                    "eval_count": result.get('eval_count', 0),
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second
                }
            else:
                return {
                    "success": False,
                    "error": f"Error: {response.status_code} - {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error on second attempt: {str(e)}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def run_benchmark(api_url, models, tasks, temperature=0.7, request_timeout=120, retry_timeout=300):
    """Performs the complete benchmark for all models and tasks."""
    global _benchmark_running, _checked_models, _execution_id
    
    # Generate a unique execution ID
    current_execution_id = f"{time.time()}-{random.randint(1000, 9999)}"
    
    # Check if benchmark is already running
    if _benchmark_running:
        print("⚠️ Benchmark is already running.")
        return None
    
    # Check if this is a duplicate call of the same execution
    if _execution_id == current_execution_id:
        print("⚠️ Duplicate call of the same benchmark execution detected and skipped.")
        return None
    
    # Lock for this execution
    _benchmark_running = True
    _execution_id = current_execution_id
    
    # Clear model cache
    _checked_models = set()
    
    # Check if input parameters are valid
    if not models or len(models) == 0:
        print("❌ No models specified for benchmarking.")
        _benchmark_running = False
        return None
    
    if not tasks or len(tasks) == 0:
        print("❌ No tasks specified for benchmarking.")
        _benchmark_running = False
        return None
    
    try:
        # Check/start server if needed
        if not check_ollama_server(api_url):
            if not start_ollama_server(api_url):
                print("❌ Ollama server could not be started.")
                return None
        
        # Check/load models (with cache checking)
        for model in models:
            # Check if we've already checked this model
            if model in _checked_models:
                continue
                
            _checked_models.add(model)
            
            if not check_model_exists(api_url, model):
                print(f"⚠️ Model {model} not found. Loading...")
                if not load_model(api_url, model):
                    print(f"❌ Model {model} could not be loaded.")
                    return None
            else:
                print(f"✅ Model {model} is ready.")
        
        # Process tasks in advance to avoid duplicate tasks
        unique_tasks = {}
        for task in tasks:
            unique_tasks[task['name']] = task
            
        # Run benchmark
        results = []
        for task_name, task in unique_tasks.items():
            print(f"\n🧪 Task: {task_name}")
            
            # Each model only once per task
            processed_models = set()
            for model in models:
                if model in processed_models:
                    continue
                    
                processed_models.add(model)
                print(f"  🤖 {model}...")
                res = benchmark_model(
                    api_url,
                    model,
                    task['prompt'],
                    max_tokens=task.get('max_tokens', 100),
                    temperature=temperature,
                    request_timeout=request_timeout,
                    retry_timeout=retry_timeout                )
                
                if res['success']:
                    results.append({
                        "Model": model,
                        "Task": task_name,
                        "Prompt": task['prompt'],
                        "Response": res.get('response', ''),
                        "Generation Time (s)": res.get('generation_time', 0),
                        "Load Time (s)": res.get('load_duration', 0),
                        "Tokens Generated": res.get('eval_count', 0),
                        "Tokens per Second": res.get('tokens_per_second', 0)
                    })
                    print(f"    ✓ {res.get('eval_count', 0)} tokens in {res.get('generation_time', 0):.2f}s")
                else:
                    print(f"    ❌ Error: {res.get('error','Unknown error')}")
        
        return results
    finally:
        # Release lock
        _benchmark_running = False