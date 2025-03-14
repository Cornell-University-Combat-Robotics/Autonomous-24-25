import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import csv

CSV = "RamRamTest/ram_ram_test95.csv"

def plotVelocity():
    fields = ['time','huey_velocity','speed']

    df = pd.read_csv(CSV, skipinitialspace=True, usecols=fields)
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

    plt.show()

plotVelocity()
    