from tkinter import *
import csv
import time
import os 


cursor_pos = None
fpath = None
fields = ['time', 'delta_time', 'bots', 'huey_pos', 'huey_facing', 'enemy_pos', 'huey_old_pos', 'huey_velocity', 'enemy_old_pos[-1]'
          , 'enemy_velocity', 'speed', 'turn', 'left_speed', 'right_speed', 'angle', 'direction']

class NoEnemyError(Exception):
    pass

def obj_detection_sim(width, height):
    #Create an instance of tkinter frame or window
    win= Tk()
    #Set the geometry of tkinter frame
    win.geometry(str(width) + "x" + str(height))
    def callback(e):
        global no_enemy
        x= e.x
        y= e.y
        # For simulation, default is width = 640, height is 360
        # print("Pointer is currently at %d, %d" %(x,y))
        global cursor_pos
        if (y > (height * 0.8)) or (y < (height * 0.2)):
            cursor_pos = None
        else:
            cursor_pos = ((x-(width/2))/(width/2), ((y-(height/2))/(-height/2)))
        # print(cursor_pos)
    win.bind('<Motion>',callback)
    win.mainloop()



''' Establish Test File for saving states of Ram RAM   

    NOTE: run this from Autonomous-23-24 or the things will not save in the right place
'''
def test_file_init():
    global fpath
    if 'Algorithm' in str(os.getcwd()):
        myDirectory = os.path.join(os.getcwd(), os.path.join('RamRamTest'))
    else:
        myDirectory = os.path.join(os.getcwd(), os.path.join('Algorithm', 'RamRamTest'))
    i = 0

    # remember to join the entire file path
    while os.path.exists(os.path.join(myDirectory, "ram_ram_test%s.csv" % i)):
        i += 1
    fpath = os.path.join(myDirectory, "ram_ram_test%s.csv" % i)
    # print(fpath)
    with open(fpath, 'w', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader() 

''' Update Test File for saving states of Ram Ram 

    to add field: add it to field array at the top, add it to arguement, and add it to dict at the end
'''
def test_file_update(delta_time = None, bots  = None, huey_pos = None, huey_facing  = None, enemy_pos  = None, 
                     huey_old_pos  = None, huey_velocity  = None, enemy_old_pos = None, enemy_velocity = None, 
                     speed  = None, turn  = None, left_speed = None, right_speed = None, angle = None, direction = None):
    # 'time', 'delta_time', 'bots', 'huey_pos', 'huey_facing', 'enemy_pos', 'huey_old_pos', 'huey_velocity', 'enemy_old_pos[-1]'
    #       , 'enemy_velocity', 'speed', 'turn', 'left_speed', 'right_speed'
    if (fpath is None):
        test_file_init()
    
    with open(fpath, 'a', newline='') as f: 
        writer = csv.DictWriter(f, fieldnames=fields)
        update_dict = {'time' : time.strftime("%H:%M:%S", time.localtime()), 'delta_time' : str(delta_time), 'bots' : str(bots), 'huey_pos' : str(huey_pos),
                       'huey_facing' : str(huey_facing), 
                       'enemy_pos' : str(enemy_pos), 'huey_old_pos': str(huey_old_pos), 'huey_velocity' : str(huey_velocity), 
                       'enemy_old_pos[-1]' : str(enemy_old_pos), 'enemy_velocity' : str(enemy_velocity), 'speed': str(speed),'turn': str(turn), 
                       'left_speed' : str(left_speed), 'right_speed' : str(right_speed), 'angle' : str(angle), 'direction' 
                       : str(direction)}
        writer.writerow(update_dict) 

