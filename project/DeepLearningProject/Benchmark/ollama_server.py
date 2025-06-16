"""
ollama_server.py - Server management for Ollama

This module contains functions for checking and starting the Ollama server.
"""

import requests
import subprocess
import platform
import time

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