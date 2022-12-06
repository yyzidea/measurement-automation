# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 11:52:44 2015
@author: hera
"""

import serial


class Piezoconcept:
    """ A simple class for the Piezo concept FOC100 nanopositioning system """
    def __init__(self, port='COM9'):
        self.maxvalue = 50E3
        self.minvalue = 0
        self.position = 0
        try:
            self.device = serial.Serial(port, baudrate=115200, timeout=0.5)
            print('pizeoconcept connect!')
        except IOError:
            print('Connection Error!')

    def move_rel(self, value, unit="n"):
        """A command for relative movement, where the default units is nm"""
        if unit == "n":
            multiplier = 1
        elif unit == "u":
            multiplier = 1E3
        else:
            multiplier = 1
        if (value * multiplier + self.position) > self.maxvalue or (value * multiplier + self.position) < self.minvalue:
            print("The value is out of range! 0-50 um (0-5E4 nm)")
        else:
            cmd = b'MOVRX ' + str(value).encode('ascii') + unit.encode('ascii') + b'\n'
            self.device.write(cmd)
            self.position = (value * multiplier + self.position)

    def move(self, value, unit="n"):
        """An absolute movement command, will print an error to the console
        if you moveoutside of the range(50um) default unit is nm"""
        if unit == "n":
            multiplier = 1
        elif unit == "u":
            multiplier = 1E3
        else:
            multiplier = 1
        if value * multiplier > self.maxvalue or value * multiplier < self.minvalue:
            print("The value is out of range! 0-50 um (0-5E4 nm)")
        else:
            cmd = b'MOVEX ' + str(value).encode('ascii') + unit.encode('ascii') + b'\n'
            self.device.write(cmd)
            self.position = value * multiplier

    def center(self):
        """ Moves the stage to the center position"""
        self.move(self.maxvalue/2)

    def write_cmd(self, cmd='INFOS'):
        self.device.write(cmd.encode('ascii') + b'\n')
        return self.device.readlines()

    def disconnect(self):
        self.device.close()
        print('Piezoconcept disconnected !')


if __name__ == "__main__":
    '''Basic test, should open the Z stage and print its info before closing. 
    Obvisouly the comport has to be correct!'''
    Z = Piezoconcept(port="COM9")
    print(Z.write_cmd('INFOS'))
