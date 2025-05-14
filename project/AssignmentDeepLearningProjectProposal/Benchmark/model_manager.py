"""
model_manager.py - Modellverwaltung für Ollama

Dieses Modul enthält Funktionen zum Prüfen und Laden von Modellen über die Ollama-API.
"""

import requests
import json
import time

def check_model_exists(api_url, model_name):
    """Überprüft, ob ein bestimmtes Modell in Ollama vorhanden ist."""
    try:
        response = requests.get(f"{api_url}/tags")
        models = response.json().get('models', [])
        exists = any(model['name'] == model_name for model in models)
        return exists
    except Exception as e:
        print(f"⚠️ Fehler beim Prüfen des Modells {model_name}: {str(e)}")
        return False

def load_model(api_url, model_name):
    """Lädt ein Modell von Ollama, falls es noch nicht vorhanden ist."""
    print(f"Lade Modell {model_name}...")
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
                
                # Nur neue Status-Updates anzeigen
                if status != last_status and status:
                    # Fortschritt anzeigen ohne Zeilenumbruch
                    print(f"Status: {status}", end='\r')
                    last_status = status
                
                if 'completed' in status.lower():
                    print(f"\n✅ {model_name} wurde erfolgreich geladen")
                    return True
                elif 'error' in data:
                    print(f"\n❌ Fehler beim Laden von {model_name}: {data['error']}")
                    return False
        
        return True
    except Exception as e:
        print(f"\n❌ Exception beim Laden von {model_name}: {str(e)}")
        return False