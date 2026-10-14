@echo off
cd /d "%~dp0"
py -3.14 -m fraudvision.desktop
if errorlevel 1 (
  echo.
  echo Install the required image library if needed:
  echo py -3.14 -m pip install -r requirements.txt
  pause
)
