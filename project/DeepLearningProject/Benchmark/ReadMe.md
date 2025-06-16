# LLM Benchmark Framework

## Project Overview

This project provides a modular framework for benchmarking Large Language Models (LLMs) available through the Ollama API. It enables comparison of different models regarding speed, performance, and response quality for arbitrary tasks. Evaluation and visualization are performed directly in the Jupyter Notebook.

---

## Directory Structure

```
DeepLearning/
├── model_benchmark.ipynb      # Main notebook for configuration, execution & evaluation
├── model_benchmark_utils.py   # Utility functions: orchestrates benchmark, evaluation, visualization
├── benchmark_core.py          # Core logic: execution of individual benchmarks, timing, metrics
├── model_manager.py           # Model management: check availability, load models
├── ollama_server.py           # Ollama server management: start, status check, error handling
├── visualization.py           # Results visualization: charts, summaries
├── model_benchmark_results.csv# Benchmark results (including prompts & responses)
├── __init__.py                # Package initialization
└── ReadMe.md                  # This documentation
```

---

## Architecture Overview

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

- **model_benchmark.ipynb**: Central control, configuration and evaluation. Contains the main logic for the workflow, calls all core functions.
- **model_benchmark_utils.py**: Orchestrates the benchmark process, contains utility functions like `run_benchmark_test()`, evaluation and visualization.
- **ollama_server.py**: Starts and checks the Ollama server, contains e.g. `start_ollama_server()`, `check_ollama_status()`.
- **model_manager.py**: Checks and loads models, e.g. with `check_model_availability()`, `load_model()`.
- **benchmark_core.py**: Executes benchmarks for models & tasks, measures time, calculates metrics, e.g. `run_single_benchmark()`.
- **visualization.py**: Visualizes and summarizes results, e.g. bar charts, summaries, `plot_results()`.

---

## Detailed Workflow Diagram

```
+-----------------------------+
| Notebook Start              |
+-----------------------------+
            |
            v
+-----------------------------+
| Import & Module Check       |
| (e.g. import pandas,        |
|  check_ollama_status())     |
+-----------------------------+
            |
            v
+-----------------------------+
| Configuration:              |
| - MODELS                    |
| - BENCHMARK_TASKS           |
| - Parameters (TEMPERATURE..)|
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
| Server & Model Check        |
| (check_ollama_status,       |
|  check_model_availability)  |
+-----------------------------+
            |
            v
+-----------------------------+
| Benchmark Execution         |
| (run_single_benchmark,      |
|  Timing, Metrics)           |
+-----------------------------+
            |
            v
+-----------------------------+
| Save Results                |
| (in model_benchmark_results.csv) |
+-----------------------------+
            |
            v
+-----------------------------+
| Visualization & Summary     |
| (plot_results, summary)     |
+-----------------------------+
```

---

## Configurable Parameters

- **MODELS** *(List[str])*: List of models to compare, e.g. `["llama3.2", "deepseek-r1:1.5b"]`
- **BENCHMARK_TASKS** *(List[dict])*: Tasks with name, prompt and max_tokens, e.g. `{ "name": "Text Generation", "prompt": "Write a paragraph...", "max_tokens": 50 }`
- **TEMPERATURE** *(float)*: Sampling temperature for the models, e.g. `0.2`
- **REQUEST_TIMEOUT** *(int)*: Timeout for model responses in seconds, e.g. `30`
- **RETRY_TIMEOUT** *(int)*: Timeout for retry attempts in seconds, e.g. `10`

---

## Result Files

- **model_benchmark_results.csv**: Contains all benchmark results including metrics, prompts and model responses.

**Example entry:**

| Model         | Task              | Generation Time (s) | Tokens Generated | Tokens per Second | Prompt                        | Response                |
|---------------|-------------------|---------------------|------------------|-------------------|-------------------------------|-------------------------|
| llama3.2      | Text Generation   | 1.23                | 28               | 22.8              | Write a short paragraph...    | ...                     |
| deepseek-r1   | Factual Knowledge | 1.01                | 30               | 29.7              | What is the difference ...    | ...                     |

---

## Extension Possibilities

- Adding more models and tasks by adjusting the `MODELS` and `BENCHMARK_TASKS` parameters.
- Customizing visualization, for example through additional chart types or export functions.
- Extension with additional metrics, such as costs or energy consumption.

---

## Requirements

- Python 3.x
- Ollama installed and running (https://ollama.com/)
- Required Python packages: pandas, matplotlib, numpy, requests, IPython

**Package Installation:**

```powershell
pip install pandas matplotlib numpy requests ipython
```

---

## Contact

For questions or extension suggestions: marcn@[please complete]
