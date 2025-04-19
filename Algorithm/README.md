# Algorithm

## Description
This code tests the Ram Ram algorithm by controlling the movements of two robots through a Tkinter window. 

## Mode
We have 2 mode inside ram.py
1. `TEST_MODE`: When `True`, a csv file with inputs and outputs to the algorithm will be generated in `Algorithm/RamRamTest`

## Packages
In order to get the Algorithm code to work, you need to install the following libraries

- numpy: `pip install numpy`

In order to get the pygame_test code to work, you need to install the following libraries
- Pygame : `pip install Pygame`
- Tkinter : `pip install Tk`

In order to get the algo_motor_integration code to work, you need to install the following libraries
- Pyserial : `python -m pip install pyserial`

## Running the Algorithm

You should be able to run the code in either the Autonomous-24-25 folder or the Algorithm folder

IF Algorithm Folder:
Run `python3 pygame_test.py`

IF Autonomous-24-25:
Run `python3 Algorithm/pygame_test.py`

## Testing

A Tkinter window will pop up, showing a blue dot and a red dot connected by a line in the corners. 

The red dot with the arrow is our bot and the blue dot is the enemy. 
Our bot is controlled by the WAD keys (W: forward, A: turn left, D: turn right).

Enemy is controlled by the arrow keys.