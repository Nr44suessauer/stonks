# LLM Benchmark-Framework

## Projektübersicht

Dieses Projekt bietet ein modulares Framework zum Benchmarking von Large Language Models (LLMs), die über die Ollama-API verfügbar sind. Es ermöglicht den Vergleich verschiedener Modelle hinsichtlich Geschwindigkeit, Performance und Antwortqualität für beliebige Aufgaben. Die Auswertung und Visualisierung erfolgt direkt im Jupyter Notebook.

---

## Verzeichnisstruktur

```
DeepLearning/
├── model_benchmark.ipynb      # Hauptnotebook für Konfiguration, Ausführung & Auswertung
├── model_benchmark_utils.py   # Hilfsfunktionen: orchestriert Benchmark, Auswertung, Visualisierung
├── benchmark_core.py          # Kernlogik: Durchführung einzelner Benchmarks, Timing, Metriken
├── model_manager.py           # Modellverwaltung: Verfügbarkeit prüfen, Modelle laden
├── ollama_server.py           # Ollama-Server-Management: Start, Statusprüfung, Fehlerbehandlung
├── visualization.py           # Ergebnisvisualisierung: Diagramme, Zusammenfassungen
├── model_benchmark_results.csv# Ergebnisse der Benchmarks (inkl. Prompts & Antworten)
├── __init__.py                # Paket-Initialisierung
└── ReadMe.md                  # Diese Dokumentation
```

---

## Architekturübersicht

```
+-------------------+
| model_benchmark   |
|    .ipynb         |
+-------------------+
          |
          v
+-------------------+
| model_benchmark_  |
|   utils.py        |
+-------------------+
          |
   +------+------+-------------------+
   |      |      |                   |
   v      v      v                   v
ollama_  model_  benchmark_     visualization.py
server.py manager.py core.py
```

- **model_benchmark.ipynb**: Zentrale Steuerung, Konfiguration und Auswertung. Enthält die Hauptlogik für den Ablauf, ruft alle Kernfunktionen auf.
- **model_benchmark_utils.py**: Orchestriert den Benchmark-Prozess, enthält Hilfsfunktionen wie `run_benchmark_test()`, Auswertung und Visualisierung.
- **ollama_server.py**: Startet und prüft den Ollama-Server, enthält z.B. `start_ollama_server()`, `check_ollama_status()`.
- **model_manager.py**: Prüft und lädt Modelle, z.B. mit `check_model_availability()`, `load_model()`.
- **benchmark_core.py**: Führt Benchmarks für Modelle & Aufgaben aus, misst Zeit, berechnet Metriken, z.B. `run_single_benchmark()`.
- **visualization.py**: Visualisiert und fasst Ergebnisse zusammen, z.B. Balkendiagramme, Zusammenfassungen, `plot_results()`.

---

## Ablaufdiagramm (detailliert)

```
+-----------------------------+
| Notebook-Start              |
+-----------------------------+
            |
            v
+-----------------------------+
| Import & Modulprüfung       |
| (z.B. import pandas,        |
|  check_ollama_status())     |
+-----------------------------+
            |
            v
+-----------------------------+
| Konfiguration:              |
| - MODELS                    |
| - BENCHMARK_TASKS           |
| - Parameter (TEMPERATURE...)|
+-----------------------------+
            |
            v
+-----------------------------+
| run_benchmark_test()        |
| (in model_benchmark_utils)  |
+-----------------------------+
            |
            v
+-----------------------------+
| Server- & Modell-Check      |
| (check_ollama_status,       |
|  check_model_availability)  |
+-----------------------------+
            |
            v
+-----------------------------+
| Benchmark-Durchführung      |
| (run_single_benchmark,      |
|  Timing, Metriken)          |
+-----------------------------+
            |
            v
+-----------------------------+
| Ergebnisse speichern        |
| (in model_benchmark_results.csv) |
+-----------------------------+
            |
            v
+-----------------------------+
| Visualisierung & Fazit      |
| (plot_results, summary)     |
+-----------------------------+
```

---

## Konfigurierbare Parameter

- **MODELS** *(List[str])*: Liste der zu vergleichenden Modelle, z.B. `["llama3.2", "deepseek-r1:1.5b"]`
- **BENCHMARK_TASKS** *(List[dict])*: Aufgaben mit Name, Prompt und max_tokens, z.B. `{ "name": "Textgen", "prompt": "Schreibe einen Absatz...", "max_tokens": 50 }`
- **TEMPERATURE** *(float)*: Sampling-Temperatur für die Modelle, z.B. `0.2`
- **REQUEST_TIMEOUT** *(int)*: Timeout für Modellantworten in Sekunden, z.B. `30`
- **RETRY_TIMEOUT** *(int)*: Timeout für einen zweiten Versuch in Sekunden, z.B. `10`

---

## Ergebnisdateien

- **model_benchmark_results.csv**: Enthält alle Benchmarkergebnisse einschließlich Metriken, Prompts und Modellantworten.

**Beispiel für einen Eintrag:**

| Modell         | Aufgabe           | Generierungszeit (s) | Tokens generiert | Tokens pro Sekunde | Prompt                        | Antwort                 |
|---------------|-------------------|----------------------|------------------|--------------------|-------------------------------|-------------------------|
| llama3.2      | Textgenerierung   | 1.23                 | 28               | 22.8               | Schreibe einen kurzen Absatz...| ...                     |
| deepseek-r1   | Fakten-Wissen     | 1.01                 | 30               | 29.7               | Was ist der Unterschied ...   | ...                     |

---

## Erweiterungsmöglichkeiten

- Hinzufügen weiterer Modelle und Aufgaben durch Anpassung der Parameter `MODELS` und `BENCHMARK_TASKS`.
- Anpassung der Visualisierung, beispielsweise durch weitere Diagrammtypen oder Exportfunktionen.
- Erweiterung um zusätzliche Metriken, wie etwa Kosten oder Energieverbrauch.

---

## Voraussetzungen

- Python 3.x
- Ollama installiert und lauffähig (https://ollama.com/)
- Benötigte Python-Pakete: pandas, matplotlib, numpy, requests, IPython

**Installation der Pakete:**

```powershell
pip install pandas matplotlib numpy requests ipython
```

---

## Kontakt

Für Rückfragen oder Erweiterungsvorschläge: marcn@[bitte ergänzen]
