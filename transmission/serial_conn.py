import time
import serial
import serial.tools.list_ports


class OurSerial():
    """
    The OurSerial module is used to establish connectio witht the arduino over serial communication

    Attributes
    ----------
    ser : serial.Serial
        a serial.Serial object that sets up the connection to the arduino

    Methods
        -------
        choose_port()
            used for user to select port for serial connection

        send_data(channel, speed)
            Sets [channel] to [speed]

        cleanup()
            Closes serial connection

    """

    def __init__(self, baudrate=9600, timeout=1, port=None):
        """
        Parameters
        ----------
        buadrate : int, optional
            the baudrate (data transmission rate)
        timeout : int, optional
            amount of time to wait before raising an error on the serial connection
        port : string, optional
            port arduino is connected to. If None, then calls choose_port for port selection
        """

        if port is None:
            port = self.choose_port()
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for the serial connection to initialize

    def choose_port(self):
        """ 
        Allows user to determine what port the arduino is on

        User Guide: 
        1. Look at port list printed by choose_port
        2. unplug arduino and press '0' to refresh list
        3. Look to see which port is missing
        4. replug arduino and refresh port list
        6. select index of arduino port

        Returns: string port value (ex. "COM3")
        """

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
        """
        Sets [channel] to [speed]

        Parameters
        ----------
        channel: int
            the channel to change
        speed: float
            the value to set the channel  

        """

        data = f"{channel} {speed}\n"
        self.ser.write(data.encode())

    def cleanup(self):
        """Closes serial connection"""
        self.ser.close()
