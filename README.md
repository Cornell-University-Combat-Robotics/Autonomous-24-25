# Autonomous-24-25

'main' Folder:

general code can be used in all platform 


'raspberry_pi' Folder:

specific code can be used in only raspberry_pi


'scripts' Folder:

# To-Do:
Test setup script for Raspberry Pi & Windows

bash script for automation
Note: Make sure to navigate to the 'Autonomous-24-25/scripts' directionary


    # Mac & Raspberry Pi

    #Raspberry Pi command:
    chmod +x [script].sh
    ./[script].sh

    #Mac:
    bash [script].sh


    #Activate the virtual environment
    source ~/Autonomous-24-25/bin/activate

    #Deactivate the virtual environment
    deactivate
    

    # Windows
    In order for the setup script to work, you need use run the script in PowerShell

    .\[script].ps1 

    #Activate the virtual environment
    Write-Output "Activating Python virtual environment..."
    & C:\Users\$env:USERNAME\Autonomous-24-25\Scripts\Activate.ps1

    #Deactivate the virtual environment
    Write-Output "Deactivating Python virtual environment..."
    deactivate




2. Set Up Your Mac
a. Enable SSH on Your Mac

    Open System Preferences: Click on the Apple logo in the top-left corner and select System Preferences.

    Go to Sharing: Open the Sharing pane.

    Enable Remote Login: Check the box next to Remote Login. This will enable SSH access to your Mac.

    Allow Access: Make sure your Mac is set to allow access for the appropriate users. You can choose "All users" or select specific users.

b. Find Your Mac's IP Address

    Open System Preferences.

    Go to Network: Open the Network pane.

    Select Your Network Connection: Click on the active network connection (Wi-Fi or Ethernet).

    Find the IP Address: Your IP address will be listed on the right side under "Status" (e.g., 192.168.1.100).