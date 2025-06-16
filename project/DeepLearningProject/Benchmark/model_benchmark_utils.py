"""
model_benchmark_utils.py - Utility functions for LLM benchmarking with Ollama

This module contains all functions for benchmark testing of language models
via the Ollama API, including server management, model checking and loading,
benchmark execution and result visualization.
"""

import requests
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import subprocess
import platform
import os
import sys
from benchmark_core import run_benchmark, benchmark_model

# Global variables for benchmark status
_benchmark_running = False
# Cache for already checked models, prevents duplicate checks
_checked_models = set()
# Unique ID for the current benchmark execution
_execution_id = None

def check_ollama_server(api_url):
    """Checks if the Ollama server is running and reachable."""
    try:
        response = requests.get(f"{api_url}/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_ollama_server(api_url):
    """Starts the Ollama server if it's not already running."""
    print("🚀 Trying to start the Ollama server...")
    
    # Check if we're on Windows or Linux/Mac
    if platform.system() == "Windows":
        try:
            # The Ollama server is started in the background
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE  # Open new console window
            )
            
            # Wait until the server is started
            print("⏳ Waiting for the Ollama server to start...")
            time.sleep(5)  # Wait 5 seconds
            
            # Check if the server is running
            if check_ollama_server(api_url):
                print("✅ Ollama server was started successfully.")
                return True
            else:
                print("⚠️ Ollama server could not be started. Try manually.")
                return False
                
        except Exception as e:
            print(f"❌ Error starting the Ollama server: {str(e)}")
            print("ℹ️ Please start the Ollama server manually with the command 'ollama serve'")
            return False
    else:  # Mac or Linux
        try:
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Equivalent to nohup under Unix
            )
            
            # Wait until the server is started
            print("⏳ Waiting for the Ollama server to start...")
            time.sleep(5)  # Wait 5 seconds
            
            # Check if the server is running
            if check_ollama_server(api_url):
                print("✅ Ollama server was started successfully.")
                return True
            else:
                print("⚠️ Ollama server could not be started. Try manually.")
                return False
                
        except Exception as e:
            print(f"❌ Error starting the Ollama server: {str(e)}")
            print("ℹ️ Please start the Ollama server manually with the command 'ollama serve'")
            return False

def check_model_exists(api_url, model_name):
    """Checks if a specific model exists in Ollama."""
    try:
        response = requests.get(f"{api_url}/tags")
        models = response.json().get('models', [])
        exists = any(model['name'] == model_name for model in models)
        return exists
    except Exception as e:
        print(f"⚠️ Error checking model {model_name}: {str(e)}")
        return False

def load_model(api_url, model_name):
    """Loads a model from Ollama if it's not already present."""
    print(f"Loading model {model_name}...")
    try:
        response = requests.post(
            f"{api_url}/pull",
            json={"name": model_name},
            stream=True
        )
        
        last_status = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                status = data.get('status', '')
                
                # Only show new status updates
                if status != last_status and status:
                    # Show progress without newline
                    print(f"Status: {status}", end='\r')
                    last_status = status
                
                if 'completed' in status.lower():
                    print(f"\n✅ {model_name} was loaded successfully")
                    return True
                elif 'error' in data:
                    print(f"\n❌ Error loading {model_name}: {data['error']}")
                    return False
        
        return True
    except Exception as e:
        print(f"\n❌ Exception loading {model_name}: {str(e)}")
        return False

def run_benchmark_test(api_url, models, tasks, temperature=0.7):
    """
    Runs the complete benchmark and visualizes the results.
    
    Args:
        api_url: The URL of the Ollama API
        models: List of models to test
        tasks: List of benchmark tasks
        temperature: Sampling temperature for the models
        
    Returns:
        DataFrame with benchmark results or None on errors    """
    global _run_once
    
    try:
        if _run_once:
            print("⚠️ This benchmark has already been executed. If you want to run it again, please restart the kernel (Kernel → Restart).")
            return None
    except NameError:
        # The variable doesn't exist, so this is the first execution
        _run_once = True
    
    # Format model names for output
    model_names = ", ".join(models)
    print(f"🚀 Starting benchmark test for {model_names}...\n")
    
    # Run benchmark
    benchmark_results = run_benchmark(
        api_url=api_url,
        models=models,
        tasks=tasks,
        temperature=temperature
    )
    
    if benchmark_results:
        # Visualize results
        visualize_results(benchmark_results, models, tasks)
        
        # Save results as CSV
        df = pd.DataFrame(benchmark_results)
        df.to_csv('model_benchmark_results.csv', index=False)
        print("\n💾 Results have been saved to 'model_benchmark_results.csv'.")
        
        return benchmark_results
    else:
        print("\n❌ No results available for visualization or saving.")
        return None

def visualize_results(results, models, tasks):
    """Visualizes the benchmark results with charts and tables."""
    if not results:
        print("No results available for visualization.")
        return
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Get unique task names
    task_names = list(set([task['name'] for task in tasks]))
    
    # 1. Compare generation times
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(task_names))
    width = 0.35
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        times = []
        for task_name in task_names:
            # Safe access to values with error handling
            task_data = model_data[model_data['Task'] == task_name]['Generation Time (s)'].values
            if len(task_data) > 0:
                times.append(task_data[0])
            else:
                # If no data available for this task/model combination
                times.append(0)  # Or another default value                print(f"⚠️ No data for model {model} and task '{task_name}'")
        
        plt.bar(x + i*width, times, width, label=model)
    
    plt.xlabel('Task')
    plt.ylabel('Generation Time (s)')
    plt.title('Comparison of Generation Times')
    plt.xticks(x + width/2, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2. Compare tokens per second
    plt.figure(figsize=(12, 6))
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        tokens_per_sec = []
        for task_name in task_names:
            # Safe access to values with error handling
            task_data = model_data[model_data['Task'] == task_name]['Tokens per Second'].values
            if len(task_data) > 0:
                tokens_per_sec.append(task_data[0])
            else:
                # If no data available for this task/model combination
                tokens_per_sec.append(0)  # Or another default value
        
        plt.bar(x + i*width, tokens_per_sec, width, label=model)
    
    plt.xlabel('Task')
    plt.ylabel('Tokens per Second')
    plt.title('Comparison of Tokens per Second')
    plt.xticks(x + width/2, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 3. Summary table
    if len(df) > 0:  # Make sure data is available
        summary = df.groupby('Model').agg({
            'Generation Time (s)': 'mean',
            'Tokens Generated': 'mean',
            'Tokens per Second': 'mean'
        }).reset_index()
        
        print("\n📊 Summary of Performance Metrics:")
        display(summary)
    else:
        print("\n⚠️ No data available for summary.")
    
    # 4. Show qualitative evaluation
    print("\n📝 Qualitative Evaluation of Responses:")
    # Get unique tasks from results
    unique_task_names = df['Task'].unique()
    
    for task_name in unique_task_names:
        matching_tasks = [task for task in tasks if task['name'] == task_name]
        if not matching_tasks:
            continue
            
        task = matching_tasks[0]
        print(f"\n🔍 Task: {task_name}")
        print(f"Prompt: {task['prompt']}")
        
        for model in models:
            model_response = df[(df['Model'] == model) & (df['Task'] == task_name)]['Response'].values
            if len(model_response) > 0:
                print(f"\n🤖 {model} Response:")
                print(model_response[0])
                print("-" * 80)
            else:
                print(f"\n⚠️ No response from {model} available for this task.")
                print("-" * 80)

# Global variable to track if the benchmark has already been executed
_run_once = False