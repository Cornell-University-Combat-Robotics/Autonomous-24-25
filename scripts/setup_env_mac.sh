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
if command -v python3.12 &> /dev/null
then
    echo "Python 3.12 is installed."
else
    echo "Python 3.12 is not installed."
    brew install python@3.12
fi

# Create a Python virtual environment
VENV_PATH= "~/Autonomous-24-25"
# Check if the virtual environment directory exists
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists at $VENV_PATH."

    # Check if the virtual environment is active
    if [[ "$VIRTUAL_ENV" == "$VENV_PATH" ]]; then
        echo "Virtual environment is active."
    else
        echo "Virtual environment is not active."
    fi
else
    echo "Virtual environment does not exist at $VENV_PATH."
    echo "Creating Python virtual environment..."
    python3.12 -m venv ~/Autonomous-24-25
fi

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
