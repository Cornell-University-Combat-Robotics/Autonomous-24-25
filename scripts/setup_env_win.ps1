# Check and install Chocolatey
# if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
#     Write-Output "Chocolatey not found. Installing Chocolatey..."
#     Set-ExecutionPolicy Bypass -Scope Process -Force
#     iex "& { $(irm https://chocolatey.org/install.ps1 -UseBasicP) }"
# } else {
#     Write-Output "Chocolatey is already installed."
# }

# Install Python 3.12 with winget
# Check if Python 3.12 is installed
$pythonPath = Get-Command python3.12 -ErrorAction SilentlyContinue

if ($pythonPath) {
    Write-Output "Python 3.12 is installed."
} else {
    Write-Output "Python 3.12 is not installed."
    Write-Output "Installing Python 3.12 using winget..."
    
    # Install Python 3.12 using winget
    winget install --id Python.Python.3.12
}

# Create a Python virtual environment
Write-Output "Creating Python virtual environment..."
python -m venv C:\Users\$env:USERNAME\Autonomous-24-25

# # Activate the virtual environment
Write-Output "Activating Python virtual environment..."
& C:\Users\$env:USERNAME\Autonomous-24-25\Scripts\Activate.ps1

# Install required Python packages from requirements.txt
Write-Output "Installing Python packages from requirements.txt..."
cd ..
pip install -r requirements.txt

# # Deactivate the virtual environment
Write-Output "Deactivating Python virtual environment..."
deactivate

Write-Output "Setup completed! Your environment is ready."
