"""
benchmark_core.py - Kern-Funktionen für das Benchmarking von Sprachmodellen

Dieses Modul enthält die Hauptfunktionen zur Durchführung von Benchmark-Tests mit Ollama.
"""

import time
import random
import requests
from ollama_server import check_ollama_server, start_ollama_server
from model_manager import check_model_exists, load_model

# Globale Variablen für den Benchmark-Status
_benchmark_running = False
# Cache für bereits geprüfte Modelle, verhindert doppelte Prüfungen
_checked_models = set()
# Eindeutige ID für die aktuelle Benchmark-Ausführung
_execution_id = None

def benchmark_model(api_url, model_name, prompt, max_tokens=100, temperature=0.7, request_timeout=120, retry_timeout=300):
    """Führt einen Benchmark für ein einzelnes Modell mit einem Prompt durch."""
    request_data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    start_time = time.time()
    try:
        # Timeout für die Anfrage
        response = requests.post(f"{api_url}/generate", json=request_data, timeout=request_timeout)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            generation_time = end_time - start_time
            tokens_per_second = result.get('eval_count', 0) / generation_time if generation_time > 0 else 0
            
            return {
                "success": True,
                "response": result.get('response', ''),
                "total_duration": result.get('total_duration', 0) / 1_000_000_000,  # ns zu s
                "load_duration": result.get('load_duration', 0) / 1_000_000_000,    # ns zu s
                "eval_count": result.get('eval_count', 0),
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second
            }
        else:
            return {
                "success": False,
                "error": f"Fehler: {response.status_code} - {response.text}"
            }
    except requests.exceptions.Timeout:
        print(f"    ⚠️ Timeout bei der Anfrage an {model_name}. Versuche es mit erhöhtem Timeout...")
        try:
            # Zweiter Versuch mit noch höherem Timeout
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
                    "error": f"Fehler: {response.status_code} - {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fehler beim zweiten Versuch: {str(e)}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def run_benchmark(api_url, models, tasks, temperature=0.7, request_timeout=120, retry_timeout=300):
    """Führt den vollständigen Benchmark für alle Modelle und Aufgaben durch."""
    global _benchmark_running, _checked_models, _execution_id
    
    # Erzeuge eine einzigartige Ausführungs-ID
    current_execution_id = f"{time.time()}-{random.randint(1000, 9999)}"
    
    # Prüfen, ob der Benchmark bereits läuft
    if _benchmark_running:
        print("⚠️ Benchmark läuft bereits.")
        return None
    
    # Prüfen, ob dies ein Doppelaufruf derselben Ausführung ist
    if _execution_id == current_execution_id:
        print("⚠️ Doppelter Aufruf derselben Benchmark-Ausführung erkannt und übersprungen.")
        return None
    
    # Sperren für diese Ausführung
    _benchmark_running = True
    _execution_id = current_execution_id
    
    # Modell-Cache leeren
    _checked_models = set()
    
    # Überprüfen, ob die Eingabeparameter gültig sind
    if not models or len(models) == 0:
        print("❌ Keine Modelle zum Benchmarking angegeben.")
        _benchmark_running = False
        return None
    
    if not tasks or len(tasks) == 0:
        print("❌ Keine Aufgaben zum Benchmarking angegeben.")
        _benchmark_running = False
        return None
    
    try:
        # Server prüfen/ggf. starten
        if not check_ollama_server(api_url):
            if not start_ollama_server(api_url):
                print("❌ Ollama-Server konnte nicht gestartet werden.")
                return None
        
        # Modelle prüfen/laden (mit Cache-Prüfung)
        for model in models:
            # Prüft, ob wir dieses Modell bereits überprüft haben
            if model in _checked_models:
                continue
                
            _checked_models.add(model)
            
            if not check_model_exists(api_url, model):
                print(f"⚠️ Modell {model} nicht gefunden. Lade...")
                if not load_model(api_url, model):
                    print(f"❌ Modell {model} konnte nicht geladen werden.")
                    return None
            else:
                print(f"✅ Modell {model} ist bereit.")
        
        # Tasks im Voraus verarbeiten, um doppelte Aufgaben zu vermeiden
        unique_tasks = {}
        for task in tasks:
            unique_tasks[task['name']] = task
            
        # Benchmark durchführen
        results = []
        for task_name, task in unique_tasks.items():
            print(f"\n🧪 Aufgabe: {task_name}")
            
            # Jedes Modell nur einmal pro Aufgabe
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
                    retry_timeout=retry_timeout
                )
                
                if res['success']:
                    results.append({
                        "Modell": model,
                        "Aufgabe": task_name,
                        "Prompt": task['prompt'],
                        "Antwort": res['response'],
                        "Generierungszeit (s)": res['generation_time'],
                        "Ladezeit (s)": res['load_duration'],
                        "Tokens generiert": res['eval_count'],
                        "Tokens pro Sekunde": res['tokens_per_second']
                    })
                    print(f"    ✓ {res['eval_count']} Tokens in {res['generation_time']:.2f}s")
                else:
                    print(f"    ❌ Fehler: {res.get('error','Unbekannter Fehler')}")
        
        return results
    finally:
        # Sperren aufheben
        _benchmark_running = False