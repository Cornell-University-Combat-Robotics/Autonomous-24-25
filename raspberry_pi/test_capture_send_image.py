import os
import time
import paramiko  # For SFTP file transfer

# Replace with your laptop's IP address and SSH credentials
LAPTOP_IP = os.environ.get('LAPTOP_IP')
LAPTOP_USER = os.environ.get('LAPTOP_USER')
LAPTOP_PASSWORD = os.environ.get('LAPTOP_PASSWORD')

def capture_image():
    filename = '/home/pi/image.jpg'
    os.system(f'raspistill -o {filename}')
    return filename

def send_image(filename):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(LAPTOP_IP, username=LAPTOP_USER, password=LAPTOP_PASSWORD)
    
    sftp = ssh.open_sftp()
    sftp.put(filename, f'/home/{LAPTOP_USER}/image.jpg')
    sftp.close()
    ssh.close()

if __name__ == '__main__':
    #while True:
    image_file = capture_image()
    send_image(image_file)
        #time.sleep(60)  # Wait for 60 seconds before taking the next picture
