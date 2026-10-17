@echo off
REM ============================================================
REM  Worker: BACKEND.  Called by ...\启动项目.bat via `start`.
REM
REM  ASCII ONLY (cmd parses .bat with the OEM codepage).
REM
REM  WHY THIS FILE EXISTS (two real bugs, both verified)
REM  --------------------------------------------------
REM  1) The launcher used to inline this command:
REM         start "t" cmd /k "cd /d "%BACKEND%" && "%PYTHON%" main.py api"
REM     That line has an ODD number of quotes. `start` split on the
REM     first quote pair, so `&&` was consumed by the LAUNCHER and
REM     cmd /k received a truncated command. The window stayed open
REM     (/k) with a dead prompt -> looked like it was running while
REM     nothing listened on 8001. Verified by reading the spawned
REM     command line via Win32_Process.
REM     FIX: keep each command in its own file -> nothing nested.
REM
REM  2) `cd /d "%~dp0.."` went ONE level up (to C:\...\bidding-agent)
REM     while the project is at C:\...\bidding-agent\bidding-agent
REM     -> "can't open file 'C:\...\bidding-agent\main.py'".
REM     FIX: use the ABSOLUTE path below. Do not reintroduce a
REM     relative hop here; it was off by one and failed silently.
REM ============================================================
title BiddingAgent BACKEND :8001
set "BACKEND=C:\Users\10518\bidding-agent\bidding-agent"
set "PYTHON=%BACKEND%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [backend] ERROR: venv python not found:
    echo [backend]        %PYTHON%
    pause
    exit /b 1
)

pushd "%BACKEND%"
echo [backend] cwd = %CD%
echo [backend] python: %PYTHON%
"%PYTHON%" --version
echo [backend] starting: main.py api
echo.
"%PYTHON%" main.py api
set "RC=%ERRORLEVEL%"
echo.
echo [backend] EXITED with code %RC%
echo [backend] window kept open so the error above stays visible.
popd
pause
