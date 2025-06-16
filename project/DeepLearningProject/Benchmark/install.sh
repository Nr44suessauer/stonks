#!/bin/bash

echo "============================================"
echo " LLM Benchmark Framework - Installation"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python is not installed!"
        echo "Please install Python from https://www.python.org/downloads/"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python found:"
$PYTHON_CMD --version
echo

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    if ! command -v pip &> /dev/null; then
        echo "❌ pip is not available!"
        echo "Please install pip"
        exit 1
    else
        PIP_CMD="pip"
    fi
else
    PIP_CMD="pip3"
fi

echo "✅ pip found:"
$PIP_CMD --version
echo

# Upgrade pip to the latest version
echo "🔄 Updating pip..."
$PYTHON_CMD -m pip install --upgrade pip
echo

# Install all Python dependencies
echo "📦 Installing Python dependencies..."
$PIP_CMD install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Error installing Python dependencies!"
    exit 1
fi
echo

# Check if Ollama is installed
echo "🔍 Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed!"
    echo
    echo "Ollama is required for this project."
    echo "Please visit: https://ollama.ai/"
    echo
    echo "For macOS: brew install ollama"
    echo "For Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo
    echo "After installing Ollama, run this script again."
    echo
    exit 1
else
    echo "✅ Ollama found:"
    ollama --version
    echo
fi

# Install Jupyter Notebook extensions for better support
echo "📓 Installing Jupyter Notebook extensions..."
$PIP_CMD install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
echo

# Check if all important modules can be imported
echo "🧪 Testing Python modules..."
$PYTHON_CMD -c "import requests, pandas, numpy, matplotlib, jupyter, IPython; print('✅ All modules imported successfully')"
if [ $? -ne 0 ]; then
    echo "❌ Some modules could not be imported!"
    exit 1
fi
echo

# Create virtual environment (optional but recommended)
echo "🏗️  Creating virtual environment (optional)..."
$PYTHON_CMD -m venv venv
echo "Virtual environment created in 'venv' folder"
echo "To activate it, run: source venv/bin/activate"
echo

echo "============================================"
echo " ✅ Installation completed successfully!"
echo "============================================"
echo
echo "Next steps:"
echo "1. Start Ollama: ollama serve"
echo "2. Load a model: ollama pull llama2"
echo "3. Open the notebook: jupyter notebook model_benchmark.ipynb"
echo
echo "For the virtual environment:"
echo "- Activate: source venv/bin/activate"
echo "- Deactivate: deactivate"
echo
