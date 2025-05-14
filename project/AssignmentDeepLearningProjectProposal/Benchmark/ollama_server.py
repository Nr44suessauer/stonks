"""
ollama_server.py - Server-Management für Ollama

Dieses Modul enthält Funktionen zum Prüfen und Starten des Ollama-Servers.
"""

import requests
import subprocess
import platform
import time

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