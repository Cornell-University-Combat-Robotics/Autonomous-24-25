# Autonomous-24-25 Environment Setup

This provide setup scripts to configure development environments for macOS, Raspberry Pi (Linux), and Windows platforms. The setup includes installing necessary software, creating a Python virtual environment, and installing dependencies from requirements.txt.

## Prerequisites
For macOS & RaspberryPi, the beginning of the scripts check and install the prerequisites, so you don't need to worry aobut it.
    macOS: Ensure you have internet access to install Homebrew and Python.
    Raspberry Pi (Linux): Ensure your system is up-to-date and the Raspberry Pi is properly connected.

For Window, you need to do this manually:
    Windows: Requires Run CommandPrompt as adminstrator for Winget installation. [Recommendated]
    (Or Requires PowerShell with administrator privileges for Winget installation.)

    Run this in PowerShell terminal:
    '''Set-ExecutionPolicy RemoteSigned'''

    Policy type in Powershell:
    Restricted: No scripts are allowed to run (default in Windows Powershell).
    AllSigned: Only scripts signed by a trusted publisher can run.
    RemoteSigned: Scripts created on your local computer run, but scripts downloaded from the internet need to be signed by a trusted publisher.
    Unrestricted: All scripts can run.





## Environment Activation/Deactivation
Note: Make sure to navigate to the 'Autonomous-24-25/scripts' directionary


### Mac & Raspberry Pi
    bash [script].sh


    #Activate the virtual environment
    source ~/Autonomous-24-25/bin/activate

    #Deactivate the virtual environment
    deactivate
    

### Windows
#### Command Prompt:
    .\[script].cmd

    #Activate the virtual environment
    echo Activating Python virtual environment...
    call C:\Users\%USERNAME%\Autonomous-24-25\Scripts\activate.bat

    #Deactivate the virtual environment
    echo Deactivating Python virtual environment...
    deactivate

#### Power Shell
    .\[script].ps1 

    #Activate the virtual environment
    Write-Output "Activating Python virtual environment..."
    & C:\Users\$env:USERNAME\Autonomous-24-25\Scripts\Activate.ps1

    #Deactivate the virtual environment
    Write-Output "Deactivating Python virtual environment..."
    deactivate


## Customization

    Modify the requirements.txt file as needed to include additional Python packages for your project.
    For custom environment paths, update the VENV_PATH in the script files.

## Troubleshooting

    Ensure you have adequate permissions to install software and create directories.
    If the virtual environment fails to activate, make sure the correct Python version is installed.

## Note

    macOS and Linux: Use chmod +x <script_name> to make the script executable.
    Windows: Run PowerShell scripts as Administrator for proper installation of software.