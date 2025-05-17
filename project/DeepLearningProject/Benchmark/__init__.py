"""
__init__.py - Hauptmodul für LLM-Benchmarking

Dieses Paket enthält alle Funktionen für das Benchmark-Testen von Sprachmodellen
über die Ollama API, organisiert in verschiedenen Modulen.
"""

# Importiere die wichtigsten Funktionen aus den Modulen
from ollama_server import check_ollama_server, start_ollama_server
from model_manager import check_model_exists, load_model
from benchmark_core import benchmark_model, run_benchmark
from visualization import visualize_results
from model_benchmark_utils import run_benchmark_test

# Exportiere diese Funktionen direkt aus dem Hauptpaket
__all__ = [
    'check_ollama_server',
    'start_ollama_server',
    'check_model_exists',
    'load_model',
    'benchmark_model',
    'run_benchmark',
    'visualize_results',
    'run_benchmark_test'
]