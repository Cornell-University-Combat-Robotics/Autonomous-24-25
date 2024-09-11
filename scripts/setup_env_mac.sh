#!/bin/bash

# Check and install Homebrew
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

# Update Homebrew and install necessary tools
echo "Updating Homebrew and installing tools..."
brew update
brew install python@3.12

# Create a Python virtual environment
echo "Creating Python virtual environment..."
python3.12 -m venv ~/Autonomous-24-25

# Activate the virtual environment
echo "Activating Python virtual environment..."
source ~/Autonomous-24-25/bin/activate

# Install required Python packages from requirements.txt
echo "Installing Python packages from requirements.txt..."
cd ..
pip install -r requirements.txt

# Deactivate the virtual environment
echo "Deactivating Python virtual environment..."
deactivate

echo "Setup completed! Your environment is ready."
