@echo off
setlocal

:: Check if Python 3.12 is installed
echo Checking for Python 3.12 installation...

for /f "tokens=*" %%i in ('python --version 2^>nul') do set PYTHONVERSION=%%i

echo %PYTHONVERSION% | findstr /c:"3.12" >nul
if %ERRORLEVEL% equ 0 (
    echo Python 3.12 is installed.
) else (
    echo Python 3.12 is not installed.
    echo Installing Python 3.12 using winget...
    winget install --id Python.Python.3.12
)

:: Create Python virtual environment
echo Creating Python virtual environment...
python -m venv C:\Users\%USERNAME%\Autonomous-24-25

:: Activate the virtual environment
echo Activating Python virtual environment...
call C:\Users\%USERNAME%\Autonomous-24-25\Scripts\activate.bat

:: Install required Python packages from requirements.txt
echo Installing Python packages from requirements.txt...
pip install -r requirements.txt

:: Deactivate the virtual environment
echo Deactivating Python virtual environment...
deactivate

echo Setup completed! Your environment is ready.