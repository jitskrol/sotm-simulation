@echo off
rem Hebbian Simulation launcher for Windows. Double-click to start.
rem
rem The simulation needs Python with GTK4. On Windows these come from MSYS2
rem (https://www.msys2.org). On first run this launcher offers to install
rem MSYS2 (via winget) and the required packages (via pacman), then starts
rem the simulation. Later runs start it straight away.

setlocal
title Hebbian Simulation
cd /d "%~dp0"

if defined MSYS2_ROOT (set "MSYSROOT=%MSYS2_ROOT%") else (set "MSYSROOT=C:\msys64")
set "PY=%MSYSROOT%\ucrt64\bin\python.exe"
set "PYW=%MSYSROOT%\ucrt64\bin\pythonw.exe"
set "BASH=%MSYSROOT%\usr\bin\bash.exe"
set "PACKAGES=mingw-w64-ucrt-x86_64-gtk4 mingw-w64-ucrt-x86_64-python mingw-w64-ucrt-x86_64-python-gobject mingw-w64-ucrt-x86_64-python-cairo mingw-w64-ucrt-x86_64-python-numpy mingw-w64-ucrt-x86_64-adwaita-icon-theme"

if not exist "%BASH%" goto no_msys
:check
"%PY%" -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk; import numpy, cairo" >nul 2>&1
if errorlevel 1 goto install
goto run

:no_msys
echo.
echo  The simulation needs MSYS2, which provides Python and GTK4 for Windows.
echo  It was not found in %MSYSROOT%.
echo.
where winget >nul 2>&1
if errorlevel 1 goto manual_msys
choice /M " Install MSYS2 now with winget"
if errorlevel 2 goto manual_msys
winget install --id MSYS2.MSYS2 -e --accept-source-agreements --accept-package-agreements
if not exist "%BASH%" goto manual_msys
goto install

:manual_msys
echo.
echo  Please install MSYS2 from https://www.msys2.org into C:\msys64
echo  (the default), then double-click this launcher again.
echo.
pause
exit /b 1

:install
echo.
echo  Python, GTK4 and numpy still need to be installed inside MSYS2
echo  (a download of a few hundred MB, only needed once).
echo.
choice /M " Install them now"
if errorlevel 2 exit /b 1
"%BASH%" -lc "pacman -Syu --noconfirm"
"%BASH%" -lc "pacman -Syu --noconfirm"
"%BASH%" -lc "pacman -S --needed --noconfirm %PACKAGES%"
"%PY%" -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk; import numpy, cairo" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  The installation did not complete. Open the "MSYS2 UCRT64" terminal and run:
    echo    pacman -S --needed %PACKAGES%
    echo  then double-click this launcher again.
    echo.
    pause
    exit /b 1
)

:run
set "PATH=%MSYSROOT%\ucrt64\bin;%PATH%"
if exist "%PYW%" (
    start "" "%PYW%" "%~dp0hebbian_sim.py"
) else (
    "%PY%" "%~dp0hebbian_sim.py"
    if errorlevel 1 pause
)
exit /b 0
