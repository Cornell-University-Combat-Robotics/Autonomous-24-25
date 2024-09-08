#!/bin/bash

# Update and upgrade the system
echo "Updating and upgrading the system..."
sudo apt-get update -y && sudo apt-get upgrade -y

# Install raspicam utilities for camera
echo "Installing camera utilities..."
sudo apt-get install -y raspicam-utils

# Install sshpass for password-based SSH login
echo "Installing sshpass for password-based SSH..."
sudo apt-get install -y sshpass

# Install Python, pip, and virtualenv
echo "Installing Python, pip, and virtualenv..."
sudo apt-get install -y python3.12 python3.12-pip python3.12-venv

# Create a Python virtual environment
echo "Creating Python virtual environment..."
python3.12 -m venv ~/Autonomous-24-25

# Activate the virtual environment
# echo "Activating Python virtual environment..."
# source ~/Autonomous-24-25/bin/activate

# Install required Python packages
echo "Installing Python packages..."
cd ..
pip install -r requirements.txt  # Make sure to use the updated the requirements.txt file

# Deactivate the virtual environment
# echo "Deactivating Python virtual environment..."
# deactivate

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
