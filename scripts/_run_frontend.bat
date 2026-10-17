@echo off
REM ============================================================
REM  Worker: FRONTEND (Vite dev server).  Called by ...\启动项目.bat.
REM
REM  ASCII ONLY (cmd parses .bat with the OEM codepage).
REM
REM  Same two reasons as _run_backend.bat:
REM    1) the launcher must NOT embed this in a quoted string
REM       (nested quotes made `&&` get eaten -> dead prompt window);
REM    2) use the ABSOLUTE path -- a `%~dp0..` hop was off by one
REM       and produced "can't open file ...\main.py".
REM ============================================================
title BiddingAgent FRONTEND :5173
set "FRONTEND=C:\Users\10518\bidding-agent\frontend"

if not exist "%FRONTEND%\package.json" (
    echo [frontend] ERROR: package.json not found:
    echo [frontend]        %FRONTEND%
    pause
    exit /b 1
)

pushd "%FRONTEND%"
echo [frontend] cwd = %CD%
echo [frontend] starting: npm run dev
echo.
call npm run dev
set "RC=%ERRORLEVEL%"
echo.
echo [frontend] EXITED with code %RC%
echo [frontend] window kept open so the error above stays visible.
popd
pause
