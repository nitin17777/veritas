@echo off
setlocal
echo ===================================================
echo   Starting VERITAS Fact Verification Platform
echo ===================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    echo Please create the virtual environment and install requirements first.
    pause
    exit /b 1
)

echo [1/2] Verifying frontend assets...
if not exist "frontend\dist\index.html" (
    echo Frontend build not found, building frontend...
    cd frontend
    call npm run build
    cd ..
)

echo.
echo [2/2] Launching Starlette / Uvicorn server on http://localhost:8000 ...
echo The React application and REST API are both served at:
echo    http://localhost:8000
echo.
echo Press Ctrl+C to stop the server.
echo.

.\.venv\Scripts\python.exe -m uvicorn app.api_server:app --host 0.0.0.0 --port 8000
