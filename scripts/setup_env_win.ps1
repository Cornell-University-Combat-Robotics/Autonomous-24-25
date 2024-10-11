# Check and install Chocolatey
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Output "Chocolatey not found. Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    iex "& { $(irm https://chocolatey.org/install.ps1 -UseBasicP) }"
} else {
    Write-Output "Chocolatey is already installed."
}

# Install Python 3.12
Write-Output "Installing Python 3.12..."
choco install python --version=3.12.0 -y

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
