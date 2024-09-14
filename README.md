# Autonomous-24-25

'main' Folder:

general code can be used in all platform 


'raspberry_pi' Folder:

specific code can be used in only raspberry_pi


'scripts' Folder:

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