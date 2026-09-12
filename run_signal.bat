@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    echo Python launcher "py" was not found.
    echo Install Python 3.12 or newer and try again.
    pause
    exit /b 1
)

echo Starting Project SIGNAL Phase 1...
py -3 -m signal_lab.cli.simulate %*

if errorlevel 1 (
    echo.
    echo Project SIGNAL exited with an error.
    pause
)

endlocal
