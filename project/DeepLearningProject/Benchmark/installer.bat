@echo off
echo ============================================
echo  LLM Benchmark Framework - Quick Installer
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found:
python --version
echo.

REM Use python -m pip to avoid version conflicts
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

echo 📦 Installing core packages (this may take a few minutes)...
echo.

REM Install packages one by one with minimal output
echo 🔹 Installing requests...
python -m pip install requests --quiet --no-warn-script-location
echo   ✅ requests installed

echo 🔹 Installing pandas...
python -m pip install pandas --quiet --no-warn-script-location
echo   ✅ pandas installed

echo 🔹 Installing numpy...
python -m pip install numpy --quiet --no-warn-script-location
echo   ✅ numpy installed

echo 🔹 Installing matplotlib...
python -m pip install matplotlib --quiet --no-warn-script-location
echo   ✅ matplotlib installed

echo 🔹 Installing IPython...
python -m pip install ipython --quiet --no-warn-script-location
echo   ✅ IPython installed

echo 🔹 Installing JupyterLab...
python -m pip install jupyterlab --quiet --no-warn-script-location
if %errorlevel% neq 0 (
    echo   ⚠️  JupyterLab failed, installing notebook...
    python -m pip install notebook --quiet --no-warn-script-location
    echo   ✅ notebook installed
) else (
    echo   ✅ JupyterLab installed
)

echo 🔹 Installing additional tools...
python -m pip install seaborn plotly tqdm colorama scipy --quiet --no-warn-script-location
echo   ✅ additional tools installed
echo.

REM Quick test of critical modules
echo 🧪 Testing modules...
python -c "import requests, pandas, numpy, matplotlib, IPython" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Some modules failed to import!
    echo Trying to fix...
    python -m pip install --force-reinstall requests pandas numpy matplotlib ipython --quiet
    python -c "import requests, pandas, numpy, matplotlib, IPython" >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Module import still failing!
        pause
        exit /b 1
    )
)
echo ✅ All core modules working!
echo.

REM Check Ollama
echo 🔍 Checking Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Ollama not found. Please install from: https://ollama.ai/
) else (
    echo ✅ Ollama found:
    ollama --version
)
echo.

echo ============================================
echo  ✅ Installation completed successfully!
echo ============================================
echo.
echo Next steps:
echo 1. Start Ollama: ollama serve
echo 2. Pull a model: ollama pull llama2
echo 3. Open JupyterLab: jupyter lab
echo.
pause
