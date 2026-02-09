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

set "MSG=%~1"
if "%MSG%"=="" set "MSG=update %date% %time%"

for /f "delims=" %%b in ('git branch --show-current') do set "BRANCH=%%b"
if "%BRANCH%"=="" (
  echo [ERROR] Could not detect current branch.
  popd >nul
  pause
  exit /b 1
)

echo.
echo [INFO] Current branch: %BRANCH%
echo [INFO] Commit message: %MSG%

git add -A -- .
if errorlevel 1 (
  echo [ERROR] Failed to stage changes.
  popd >nul
  pause
  exit /b 1
)

git diff --cached --quiet
if not errorlevel 1 (
  goto do_commit
)
echo [INFO] No staged changes to commit. Nothing to push.
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
