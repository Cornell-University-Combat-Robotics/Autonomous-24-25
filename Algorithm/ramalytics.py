import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import csv

CSV = "RamRamTest/ram_ram_test95.csv"

def plotVelocity(df):
    column_list = df['huey_velocity'].tolist()
    df['row_number'] = np.arange(len(df))
    
    result = []
    for item in column_list:
        coords = item.strip('[]').split()
        coords_list = ((float(coords[0])**2+float(coords[1])**2)**0.5)/200
        result.append(coords_list)

    # Print the result
    df['huey_velocity'] = result
    

    df.plot('row_number',y=['huey_velocity','speed'])

def plotTurn(df):
    turn = df['turn'].tolist()
    facing = df['huey_facing'].tolist()
    df['row_number'] = np.arange(len(df))

    dfacing = []

    for i in range(len(turn)-1):
        diff = (facing[i+1]-facing[i])
        dfacing.append(abs(diff))

    dfacing.append(diff)

    df['turn'] = dfacing    

    df.plot('row_number',y=['huey_facing','turn'])


fields = ['time','huey_velocity','speed','huey_facing','turn']
df = pd.read_csv(CSV, skipinitialspace=True, usecols=fields)
plotTurn(df)
plt.show()
    