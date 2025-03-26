import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import csv

CSV = "RamRamTest/ram_ram_test101.csv"

def plotVelocity(df):
    # Create a copy of the dataframe for plotting
    plot_df = df.copy()
    column_list = plot_df['huey_velocity'].tolist()
    plot_df['row_number'] = np.arange(len(plot_df))
    
    result = []
    for item in column_list:
        coords = item.strip('[]').split()
        coords_list = ((float(coords[0])**2+float(coords[1])**2)**0.5)/200
        result.append(coords_list)

    plot_df['huey_velocity'] = result
    
    plot_df_filtered = plot_df.iloc[5:, :]
    
    # Plot with the filtered data
    plot_df_filtered.plot('row_number', y=['speed','huey_velocity'])

def plotTurn(df):
    turn = df['turn'].tolist()
    facing = df['huey_facing'].tolist()
    df['row_number'] = np.arange(len(df))



    dfacing = []

    for i in range(len(facing)-1):
        diff = (facing[i+1]-facing[i])
        diff = (diff + 180) % 360 - 180
        dfacing.append((diff))

    for i in range(len(turn)):
        turn[i] = turn[i]*180

    dfacing.append(diff)

    df['facing_diff'] = dfacing    
    df['turn'] = turn  

    df.plot('row_number',y=['turn','facing_diff'])


fields = ['time','huey_velocity','speed','huey_facing','turn']
df = pd.read_csv(CSV, skipinitialspace=True, usecols=fields)
plotTurn(df)
plotVelocity(df)
plt.xlabel('frame')
plt.show()
    