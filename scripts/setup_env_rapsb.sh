#!/bin/bash

# Update and upgrade the system
echo "Updating and upgrading the system..."
sudo apt-get update -y && sudo apt-get upgrade -y

# Install raspicam utilities for camera
echo "Installing camera utilities..."
sudo apt-get install -y raspicam-utils

# # Install sshpass for password-based SSH login
# echo "Installing sshpass for password-based SSH..."
# sudo apt-get install -y sshpass

# Install Python, pip, and virtualenv
echo "Installing Python, pip, and virtualenv..."
sudo apt-get install -y python3==3.12 python3-pip python3-venv

# Create a Python virtual environment if not exist
VENV_PATH= "~/Env/Autonomous-24-25"

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
    python3.12 -m venv ~/Env/Autonomous-24-25
fi

# Activate the virtual environment
echo "Activating Python virtual environment..."
source ~/Env/Autonomous-24-25/bin/activate

# Install required Python packages
echo "Installing Python packages..."
cd ..
pip install -r requirements.txt  # Make sure to use the updated the requirements.txt file

# Deactivate the virtual environment
echo "Deactivating Python virtual environment..."
deactivate

# Enable camera interface
echo "Enabling the Raspberry Pi camera interface..."
sudo raspi-config nonint do_camera 0

# Reboot prompt for camera changes to take effect
echo "Setup completed! You need to reboot for camera changes to take effect."
echo "Would you like to reboot now? (y/n)"
read REBOOT

if [ "$REBOOT" = "y" ]; then
    echo "Rebooting now..."
    sudo reboot
else
    echo "You can manually reboot later to apply changes."
fi
