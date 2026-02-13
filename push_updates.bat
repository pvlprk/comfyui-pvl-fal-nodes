@echo off
setlocal enabledelayedexpansion

REM Usage:
REM   push_updates.bat "your commit message"
REM If no message is provided, a timestamped default is used.

pushd "%~dp0" >nul

where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Git is not installed or not in PATH.
  popd >nul
  pause
  exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
  echo [ERROR] This folder is not a git repository.
  popd >nul
  pause
  exit /b 1
)

for /f "delims=" %%b in ('git branch --show-current') do set "BRANCH=%%b"
if "%BRANCH%"=="" (
  echo [ERROR] Could not detect current branch.
  popd >nul
  pause
  exit /b 1
)

set "MSG=%~1"
if "%MSG%"=="" (
  REM Cleaner timestamp message (no weird locale/time formatting surprises)
  for /f "tokens=1-3 delims=." %%a in ("%time%") do set "T=%%a"
  set "T=%T: =0%"
  set "MSG=update %date% %T%"
)

echo.
echo [INFO] Current branch: %BRANCH%
echo [INFO] Commit message: %MSG%

REM Optional but recommended: avoid push rejection if remote advanced
git fetch --all --prune >nul 2>nul
git pull --ff-only
if errorlevel 1 (
  echo [ERROR] Pull failed (non fast-forward). Resolve manually, then rerun.
  popd >nul
  pause
  exit /b 1
)

git add -A
if errorlevel 1 (
  echo [ERROR] Failed to stage changes.
  popd >nul
  pause
  exit /b 1
)

REM If nothing staged, exit cleanly
git diff --cached --quiet
if errorlevel 1 (
  goto do_commit
)

echo [INFO] No changes to commit. Nothing to push.
popd >nul
pause
exit /b 0

:do_commit
git commit -m "%MSG%"
if errorlevel 1 (
  echo [ERROR] Commit failed.
  popd >nul
  pause
  exit /b 1
)

git ls-remote --exit-code --heads origin %BRANCH% >nul 2>nul
if errorlevel 1 (
  echo [INFO] Remote branch not found. Pushing with upstream set...
  git push -u origin %BRANCH%
) else (
  git push origin %BRANCH%
)

if errorlevel 1 (
  echo [ERROR] Push failed.
  popd >nul
  pause
  exit /b 1
)

echo [OK] Push completed successfully.
popd >nul
pause
exit /b 0
