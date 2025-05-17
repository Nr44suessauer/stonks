"""
model_benchmark_utils.py - Hilfsfunktionen für LLM-Benchmarking mit Ollama

Dieses Modul enthält alle Funktionen für das Benchmark-Testen von Sprachmodellen
über die Ollama API, einschließlich Server-Management, Modellprüfung und -laden,
Benchmark-Durchführung und Ergebnisvisualisierung.
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

# Globale Variablen für den Benchmark-Status
_benchmark_running = False
# Cache für bereits geprüfte Modelle, verhindert doppelte Prüfungen
_checked_models = set()
# Eindeutige ID für die aktuelle Benchmark-Ausführung
_execution_id = None

def check_ollama_server(api_url):
    """Überprüft, ob der Ollama-Server läuft und erreichbar ist."""
    try:
        response = requests.get(f"{api_url}/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_ollama_server(api_url):
    """Startet den Ollama-Server, falls er nicht bereits läuft."""
    print("🚀 Versuche den Ollama-Server zu starten...")
    
    # Prüfen, ob wir auf Windows oder Linux/Mac sind
    if platform.system() == "Windows":
        try:
            # Der Ollama-Server wird im Hintergrund gestartet
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE  # Neues Konsolenfenster öffnen
            )
            
            # Warten, bis der Server gestartet ist
            print("⏳ Warte, bis der Ollama-Server gestartet ist...")
            time.sleep(5)  # 5 Sekunden warten
            
            # Überprüfen, ob der Server läuft
            if check_ollama_server(api_url):
                print("✅ Ollama-Server wurde erfolgreich gestartet.")
                return True
            else:
                print("⚠️ Ollama-Server konnte nicht gestartet werden. Versuche es manuell.")
                return False
                
        except Exception as e:
            print(f"❌ Fehler beim Starten des Ollama-Servers: {str(e)}")
            print("ℹ️ Bitte starten Sie den Ollama-Server manuell mit dem Befehl 'ollama serve'")
            return False
    else:  # Mac oder Linux
        try:
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Äquivalent zu nohup unter Unix
            )
            
            # Warten, bis der Server gestartet ist
            print("⏳ Warte, bis der Ollama-Server gestartet ist...")
            time.sleep(5)  # 5 Sekunden warten
            
            # Überprüfen, ob der Server läuft
            if check_ollama_server(api_url):
                print("✅ Ollama-Server wurde erfolgreich gestartet.")
                return True
            else:
                print("⚠️ Ollama-Server konnte nicht gestartet werden. Versuche es manuell.")
                return False
                
        except Exception as e:
            print(f"❌ Fehler beim Starten des Ollama-Servers: {str(e)}")
            print("ℹ️ Bitte starten Sie den Ollama-Server manuell mit dem Befehl 'ollama serve'")
            return False

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

def run_benchmark_test(api_url, models, tasks, temperature=0.7):
    """
    Führt den vollständigen Benchmark aus und visualisiert die Ergebnisse.
    
    Args:
        api_url: Die URL der Ollama API
        models: Liste der zu testenden Modelle
        tasks: Liste der Benchmark-Aufgaben
        temperature: Sampling-Temperatur für die Modelle
        
    Returns:
        DataFrame mit den Benchmark-Ergebnissen oder None bei Fehlern
    """
    global _run_once
    
    try:
        if _run_once:
            print("⚠️ Dieser Benchmark wurde bereits ausgeführt. Wenn Sie ihn erneut ausführen möchten, starten Sie bitte den Kernel neu (Kernel → Restart).")
            return None
    except NameError:
        # Die Variable existiert nicht, also ist dies die erste Ausführung
        _run_once = True
    
    # Modellnamen für die Ausgabe formatieren
    model_names = ", ".join(models)
    print(f"🚀 Starte Benchmark-Test für {model_names}...\n")
    
    # Benchmark durchführen
    benchmark_results = run_benchmark(
        api_url=api_url,
        models=models,
        tasks=tasks,
        temperature=temperature
    )
    
    if benchmark_results:
        # Ergebnisse visualisieren
        visualize_results(benchmark_results, models, tasks)
        
        # Speichere die Ergebnisse als CSV
        df = pd.DataFrame(benchmark_results)
        df.to_csv('model_benchmark_results.csv', index=False)
        print("\n💾 Ergebnisse wurden in 'model_benchmark_results.csv' gespeichert.")
        
        return benchmark_results
    else:
        print("\n❌ Keine Ergebnisse zum Visualisieren oder Speichern vorhanden.")
        return None

def visualize_results(results, models, tasks):
    """Visualisiert die Benchmark-Ergebnisse mit Grafiken und Tabellen."""
    if not results:
        print("Keine Ergebnisse zum Visualisieren vorhanden.")
        return
    
    # Umwandeln der Ergebnisse in einen DataFrame
    df = pd.DataFrame(results)
    
    # Eindeutige Aufgabennamen erfassen
    task_names = list(set([task['name'] for task in tasks]))
    
    # 1. Generierungszeiten vergleichen
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(task_names))
    width = 0.35
    
    for i, model in enumerate(models):
        model_data = df[df['Modell'] == model]
        times = []
        for task_name in task_names:
            # Sicherer Zugriff auf die Werte mit Fehlerbehandlung
            task_data = model_data[model_data['Aufgabe'] == task_name]['Generierungszeit (s)'].values
            if len(task_data) > 0:
                times.append(task_data[0])
            else:
                # Falls keine Daten für diese Aufgabe/Modell-Kombination vorhanden sind
                times.append(0)  # Oder einen anderen Standardwert
                print(f"⚠️ Keine Daten für Modell {model} und Aufgabe '{task_name}'")
        
        plt.bar(x + i*width, times, width, label=model)
    
    plt.xlabel('Aufgabe')
    plt.ylabel('Generierungszeit (s)')
    plt.title('Vergleich der Generierungszeiten')
    plt.xticks(x + width/2, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2. Tokens pro Sekunde vergleichen
    plt.figure(figsize=(12, 6))
    
    for i, model in enumerate(models):
        model_data = df[df['Modell'] == model]
        tokens_per_sec = []
        for task_name in task_names:
            # Sicherer Zugriff auf die Werte mit Fehlerbehandlung
            task_data = model_data[model_data['Aufgabe'] == task_name]['Tokens pro Sekunde'].values
            if len(task_data) > 0:
                tokens_per_sec.append(task_data[0])
            else:
                # Falls keine Daten für diese Aufgabe/Modell-Kombination vorhanden sind
                tokens_per_sec.append(0)  # Oder einen anderen Standardwert
        
        plt.bar(x + i*width, tokens_per_sec, width, label=model)
    
    plt.xlabel('Aufgabe')
    plt.ylabel('Tokens pro Sekunde')
    plt.title('Vergleich der Tokens pro Sekunde')
    plt.xticks(x + width/2, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 3. Zusammenfassende Tabelle
    if len(df) > 0:  # Stellen Sie sicher, dass Daten vorhanden sind
        summary = df.groupby('Modell').agg({
            'Generierungszeit (s)': 'mean',
            'Tokens generiert': 'mean',
            'Tokens pro Sekunde': 'mean'
        }).reset_index()
        
        print("\n📊 Zusammenfassung der Performance-Metriken:")
        display(summary)
    else:
        print("\n⚠️ Keine Daten für die Zusammenfassung verfügbar.")
    
    # 4. Qualitative Bewertung anzeigen
    print("\n📝 Qualitative Bewertung der Antworten:")
    # Eindeutige Tasks aus Ergebnissen erfassen
    unique_task_names = df['Aufgabe'].unique()
    
    for task_name in unique_task_names:
        matching_tasks = [task for task in tasks if task['name'] == task_name]
        if not matching_tasks:
            continue
            
        task = matching_tasks[0]
        print(f"\n🔍 Aufgabe: {task_name}")
        print(f"Prompt: {task['prompt']}")
        
        for model in models:
            model_response = df[(df['Modell'] == model) & (df['Aufgabe'] == task_name)]['Antwort'].values
            if len(model_response) > 0:
                print(f"\n🤖 {model} Antwort:")
                print(model_response[0])
                print("-" * 80)
            else:
                print(f"\n⚠️ Keine Antwort von {model} für diese Aufgabe verfügbar.")
                print("-" * 80)

# Globale Variable, um zu verfolgen ob der Benchmark bereits ausgeführt wurde
_run_once = False