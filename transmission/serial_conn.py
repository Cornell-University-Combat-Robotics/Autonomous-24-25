import time
import serial

class Serial():

    def __init__(self, baudrate=9600, timeout=1, port=None):
        if port is None:
            port = self.choose_port()
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for the serial connection to initialize

    def choose_port(self):
        def get_ports():
            available_ports = serial.tools.list_ports.comports()
            port_dic = {}
            if len(available_ports) == 0:
                print("No ports found")                
            else:
                print("Choose a port from the options below:")
                for i in range(len(available_ports)):
                    port = available_ports[i]
                    port_dic[str(i+1)] = port.device
                    print(str(i+1) + ":", port)
            print("Choose 0 to refresh your options")

            selection = input("Enter your selection here: ")
            return [selection, port_dic]

        def check_validity(selection):
            while selection != "0" and selection not in port_dic:
                print("Selection invalid. Choose one of the following or 0 to refresh options:",
                      list(port_dic.keys()))
                selection = input("Enter your selection here: ")
            return selection

        selection, port_dic = get_ports()
        selection = check_validity(selection)

        while (selection == '0'):
            selection, port_dic = get_ports()
            selection = check_validity(selection)

        return port_dic[selection]


    def send_data(self, channel, speed):
        data = f"{channel} {speed}\n"
        self.ser.write(data.encode())

    def cleanup(self):
        self.ser.close()
