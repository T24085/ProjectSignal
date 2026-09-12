@echo off
setlocal

cd /d "%~dp0"

rem Edit these three values before starting a new search.
set RUNS=1000
set STEPS_PER_RUN=5000
set PARTICLES=1000
rem Exploratory searches can use a larger interval; keep 10 for final measurements.
set OBSERVATION_INTERVAL=10

echo Project SIGNAL Phase 2B structure search
echo Genomes: %RUNS%  Steps per genome: %STEPS_PER_RUN%  Particles: %PARTICLES%
echo.

py -3 -m signal_lab.cli.search --runs %RUNS% --steps %STEPS_PER_RUN% --particles %PARTICLES% --observation-interval %OBSERVATION_INTERVAL% %*

if errorlevel 1 (
    echo.
    echo Search exited with an error.
    pause
)

endlocal
