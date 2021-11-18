import serial
from warnings import warn
import numpy as np
import json


class WaveGenerator:
    def __init__(self, port, timeout=0.5):
        self.port = port
        self.ser = self.open_com(timeout)

    def open_com(self, timeout):
        try:
            # 打开串口，并得到串口对象
            ser = serial.Serial(self.port, 115200, timeout=timeout)
            # 判断是否打开成功
            if ser.is_open is False:
                ser = -1
        except Exception as e:
            print(" Serial open failed:", e)
            ser = -1

        return ser

    def read_com(self):
        res = b''
        while 1:
            res_temp = self.ser.read()
            if res_temp == b'\n':
                break
            elif res_temp == b'':
                warn('Serial read timeout!')
                break
            else:
                print(res)
                res += res_temp

        return res

    def close_com(self):
        self.ser.close()

    def send_command(self, command, data=None, data_len=0):
        if data is not None:
            command = '%s%s\n' % (command, data)
            self.ser.write(bytearray(command, encoding='ascii'))

            # data = (b'%d' % data).zfill(data_len)
            # self.ser.write(b'%s%s\n' % (bytearray(command, encoding='ascii'), data))
        else:
            self.ser.write(b'%s\n' % (bytearray(command, encoding='ascii')))

        return self.read_com()

    def __del__(self):
        self.close_com()

    def set(self, channel, **kwargs):
        if channel == 1:
            channel = 'M'
        elif channel == 2:
            channel = 'F'
        else:
            raise Exception('Illegal channel number!')

        for key, value in kwargs.items():
            if key == 'waveform':
                self.send_command('W%sW' % channel, data=value)
            elif key == 'frequency':
                self.send_command('W%sF' % channel, data=int(value * 1e6))
            elif key == 'amplitude':
                self.send_command('W%sA' % channel, data=value)
            elif key == 'offset':
                self.send_command('W%sO' % channel, data=value)
            elif key == 'duty':
                self.send_command('W%sD' % channel, data='%.1f' % value)
            elif key == 'phase':
                self.send_command('W%sP' % channel, data=value % 360)
            elif key == 'status':
                self.send_command('W%sN' % channel, data=value)
            else:
                pass

    def start(self, channel):
        self.set(channel, status=1)

    def stop(self, channel):
        self.set(channel, status=0)

    def stop_all(self):
        self.stop(1)
        self.stop(2)


class CyclicVoltammetry:
    def __init__(self, port, V_l, V_h, scan_rate=1, channel=1, mode='sweep', start_from_zero=True, timeout=0.5):
        self.port = port
        self.timeout = timeout
        if hasattr(self, 'wave_generator') is False:
            self.wave_generator = WaveGenerator(self.port, timeout=self.timeout)

        self.channel = channel
        self.V_l = V_l
        self.V_h = V_h
        self.scan_rate = scan_rate
        self.amplitude = abs(V_h - V_l)
        self.offset = (V_h + V_l) / 2
        self.frequency = self.scan_rate / self.amplitude / 2
        self.mode = mode
        self.start_from_zero = start_from_zero

        self.stop()

        if self.mode == 'binary':
            if self.V_l < self.V_h:
                self.phase = 180
            else:
                self.phase = 0

            self.wave_generator.set(self.channel, waveform=1, amplitude=self.amplitude,
                                    offset=self.offset, phase=self.phase, frequency=self.frequency)
        else:
            if self.mode != 'sweep':
                warn('Illegal mode, the default mode \'sweep\' will be used.')

            if self.start_from_zero and V_l <= 0 <= V_h:
                self.phase = V_l / self.amplitude * 180
            else:
                self.phase = 0

            self.wave_generator.set(self.channel, waveform=5, amplitude=self.amplitude,
                                    offset=self.offset, phase=self.phase, frequency=self.frequency)

    def set(self, **kwargs):
        V_l = self.V_l
        V_h = self.V_h
        scan_rate = self.scan_rate
        channel = self.channel
        start_from_zero = True
        mode = self.mode

        for key, value in kwargs.items():
            if key == 'V_l':
                V_l = value
            elif key == 'V_h':
                V_h = value
            elif key == 'scan_rate':
                scan_rate = value
            elif key == 'channel':
                channel = value
            elif key == 'start_from_zero':
                start_from_zero = value
            elif key == 'start_from_low':
                start_from_low = value
            elif key == 'mode':
                mode = value
            else:
                pass

        self.__init__(None, V_l, V_h, scan_rate, channel, mode, start_from_zero, None)

    def start(self):
        self.wave_generator.start(self.channel)

    def stop(self):
        self.wave_generator.stop(self.channel)

    def __del__(self):
        self.wave_generator.__del__()


class Chronoamperometry:
    def __init__(self, port, V_base, V_drive, t_base, t_drive, channel=1, timeout=0.5):
        self.port = port
        self.V_base = V_base
        self.V_drive = V_drive
        self.t_base = t_base
        self.t_drive = t_drive
        self.channel = channel
        self.timeout = timeout

        self.port = port
        self.timeout = timeout
        if hasattr(self, 'wave_generator') is False:
            self.wave_generator = WaveGenerator(self.port, timeout=self.timeout)
        self.stop()

        self.amplitude = abs(V_base - V_drive)
        self.offset = (V_base + V_drive) / 2
        self.frequency = 1/(t_base+t_drive)
        if self.V_base < self.V_drive:
            self.duty = t_drive/(t_base+t_drive)*100
        else:
            self.duty = t_base/(t_base+t_drive)*100

        if self.V_base < self.V_drive:
            self.phase = -t_drive/(t_base+t_drive) * 360
        else:
            self.phase = 0

        self.wave_generator.set(self.channel, waveform=1, amplitude=self.amplitude,
                                offset=self.offset, phase=self.phase, frequency=self.frequency, duty=self.duty)

    def set(self, **kwargs):
        V_base = self.V_base
        V_drive = self.V_drive
        t_base = self.t_base
        t_drive = self.t_drive
        channel = self.channel
        timeout = self.timeout

        for key, value in kwargs.items():
            if key == 'V_base':
                V_base = value
            elif key == 'V_drive':
                V_drive = value
            elif key == 't_base':
                t_base = value
            elif key == 't_drive':
                t_drive = value
            elif key == 'channel':
                channel = value
            elif key == 'timeout':
                timeout = value
            else:
                pass

        self.__init__(self.port, V_base, V_drive, t_base, t_drive, channel, timeout)

    def start(self):
        self.wave_generator.start(self.channel)

    def stop(self):
        self.wave_generator.stop(self.channel)

    def __del__(self):
        self.wave_generator.__del__()


def waveform_generate(waveform_type, t, amplitude, frequency, offset, phase, **kwargs):
    t = t - phase / 360 / frequency
    if waveform_type == 'triangle':
        return 2 * amplitude * frequency * np.abs(
            (t + 1 / 2 / frequency) % (1 / frequency) - 1 / 2 / frequency) - amplitude / 2 + offset
    elif waveform_type == 'square':
        return np.int_(np.sin(2 * np.pi * frequency * t) > 0) * amplitude - amplitude / 2 + offset
    else:
        return np.zeros_like(t)
