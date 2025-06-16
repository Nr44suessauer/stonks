"""
model_manager.py - Model management for Ollama

This module contains functions for checking and loading models via the Ollama API.
"""

import requests
import json
import time

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