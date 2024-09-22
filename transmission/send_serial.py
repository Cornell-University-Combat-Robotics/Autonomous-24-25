import argparse
import time
import tkinter as tk
from tkinter import ttk

import serial.tools.list_ports


# Initialize the serial connection
def init_serial(port, baudrate=9600, timeout=1):
    ser = serial.Serial(port, baudrate, timeout=timeout)
    time.sleep(2)  # Wait for the serial connection to initialize
    return ser

# Send data to the Arduino
def send_data(ser, steering, throttle):
    if ser is None:
        print(f"Debug Mode: Steering: {steering}, Throttle: {throttle}")
        return
    data = f"{steering} {throttle}\n"
    ser.write(data.encode())

# Update the labels and send the data
def update_values(event=None):
    steering = round(steering_scale.get())
    throttle = round(throttle_scale.get())
    steering_label.config(text=f"Steering: {steering}%")
    throttle_label.config(text=f"Throttle: {throttle}%")
    send_data(serial_conn, steering, throttle)

# Reset the values to default
def reset_values(event):
    steering_scale.set(25)
    throttle_scale.set(25)
    update_values()

# List available serial ports
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

# Connect to the selected port
def connect_port():
    global serial_conn
    port = port_var.get()
    if port == "None":
        serial_conn = None
    else:
        serial_conn = init_serial(port)
    connect_button.config(state=tk.DISABLED)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Universal RC Controller")
parser.add_argument('--debug', action='store_true', help="Run the GUI in debug mode without serial connection")
args = parser.parse_args()

# Initialize the GUI
root = tk.Tk()
root.title("Universal RC Controller")

# Create and place the port selection dropdown
port_label = ttk.Label(root, text="Select Serial Port:")
port_label.pack(pady=10)

available_ports = list_serial_ports()
if not available_ports:
    available_ports = ["None"]

port_var = tk.StringVar(value=available_ports[0])
port_menu = ttk.Combobox(root, textvariable=port_var, values=available_ports, state="readonly")
port_menu.pack(pady=10)

connect_button = ttk.Button(root, text="Connect", command=connect_port)
connect_button.pack(pady=10)

# Create and place the steering control
steering_label = ttk.Label(root, text="Steering: 25%")
steering_label.pack(pady=10)

steering_scale = ttk.Scale(root, from_=0, to=100, orient='horizontal', length=300, command=update_values)
steering_scale.set(25)
steering_scale.pack(pady=10)
steering_scale.bind("<ButtonRelease-1>", reset_values)

# Create and place the throttle control
throttle_label = ttk.Label(root, text="Throttle: 25%")
throttle_label.pack(pady=10)

throttle_scale = ttk.Scale(root, from_=0, to=100, orient='horizontal', length=300, command=update_values)
throttle_scale.set(25)
throttle_scale.pack(pady=10)
throttle_scale.bind("<ButtonRelease-1>", reset_values)

# Initialize the serial connection (None for now)
serial_conn = None if args.debug else None

# Start the GUI event loop
root.mainloop()

# Close the serial connection when the program ends
if serial_conn:
    serial_conn.close()
