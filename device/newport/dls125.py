

import serial
from time import sleep


class DLS125:
    def __init__(self, port='COM13'):
        self.accel_value = 500
        self.speed_value = 20
        try:
            self.device = serial.Serial(port, baudrate=115200, timeout=0.5)
            print('DLS125 connected!')
            self.homing()
        except IOError:
            print('Connection Error!')

    def homing(self):
        self.device.write(b'IE\r\n')
        sleep(2)
        self.device.write(b'OR\r\n')
        sleep(2)
        self.accel(self.accel_value)
        self.speed(self.speed_value)
        self.ab_pos(0)

    def accel(self, value):
        cmd = b'AC' + str(value).encode('ascii') + b'\r\n'
        self.device.write(cmd)
        sleep(0.5)
        self.device.write(b'AC?\r\n')
        sleep(0.5)
        print('Acceleration is ', end='')
        print(self.device.readline().decode('ascii').strip())

    def speed(self, value):
        cmd = b'VA' + str(value).encode('ascii') + b'\r\n'
        self.device.write(cmd)
        sleep(0.5)
        self.device.write(b'VA?\r\n')
        sleep(0.5)
        print('Speed is ', end='')
        print(self.device.readline().decode('ascii').strip())

    def ab_pos(self, value):
        pos = self.read_pos()
        time = abs(value - pos) / self.speed_value + 0.5
        cmd = b'PA' + str(value).encode('ascii') + b'\r\n'
        self.device.write(cmd)
        sleep(time)

    def rel_pos(self, value):
        cmd = b'PD' + str(value).encode('ascii') + b'\r\n'
        self.device.write(cmd)
        while not self.device.readline():
            pass

    def read_pos(self):
        self.device.write(b'TP\r\n')
        sleep(0.1)
        resp = self.device.readline().decode('ascii').strip()[2:]
        return float(resp)

    def write_cmd(self, cmd=''):
        cmd = cmd.encode('ascii') + b'\r\n'
        self.device.write(cmd)
        sleep(0.1)
        print(self.device.readline().decode('ascii').strip())


if __name__ == "__main__":
    DL = DLS125('COM13')
    for i in range(20, 100, 20):
        DL.ab_pos(i)
        sleep(4)
        print(DL.read_pos())

    for i in range(5):
        DL.rel_pos(-10)
        sleep(4)
        print(DL.read_pos())
