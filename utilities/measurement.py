import json
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from utilities.data_files_handling import check_data_files_exist, format_filename
from device.wave_generator import waveform_generate
from lifetime_trace import LifetimeTraceGated, LifetimeTraceGatedWithFileWriter, LifetimeTraceWithFileWriter, \
    disable_conditional_filter, enable_conditional_filter
from time_tagger_utility import *
from copy import deepcopy


# Config management
class MeasurementConfig:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def export_measurement_config(self, config_name, obj, attr_keys=None):
        if attr_keys is None:
            attr_keys = obj.__init__.__code__.co_varnames

        params = {}
        for key in attr_keys:
            flag = 0
            if isinstance(obj, dict):
                if key in obj.keys():
                    param = obj[key]
                    flag = 1
            else:
                if key != 'self' and hasattr(obj, key):
                    param = obj.__getattribute__(key)
                    flag = 1

            if flag:
                if param.__class__.__name__ in ['dict', 'list', 'tuple', 'str', 'int',
                                                'float', 'bool', 'NoneType']:
                    params[key] = param
                else:
                    warn('The parameter \'%s\' of type \'%s\' is not JSON serializable and is skipped.' %
                         (key, param.__class__.__name__))

        self.__dict__[config_name] = params

    def save_config(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.__dict__, fp, indent='\t')

    def load_config(self, filename):
        with open(filename, 'r') as fp:
            self.__dict__ = json.load(fp)

        return self

    def copy(self):
        return MeasurementConfig(**deepcopy(self.__dict__))


class MeasurementConfig2(MeasurementConfig):
    def __init__(self, devices_params=None, file_params=None, other_params=None):
        MeasurementConfig.__init__(self)
        if devices_params is None:
            self.devices_params = {}
        else:
            self.devices_params = deepcopy(devices_params)

        if file_params is None:
            self.file_params = {}
        else:
            self.file_params = deepcopy(file_params)

        if other_params is None:
            self.other_params = {}
        else:
            self.other_params = deepcopy(other_params)

        self.other_params['version'] = '2.0'


# Measurement function
def IvsV_measurement(config: MeasurementConfig, smu, start_voltage, stop_voltage, step, other_desc=None,
                     check_filename=True, average_sample_num=10, measure_spacing=0.1, measure_delay=0.1,
                     stop_current=1e-3, compliance_current=1e-3):
    config = config.copy()
    config.main['measurement_type'] = 'iv'
    config.export_measurement_config('iv', locals(), ['start_voltage', 'stop_voltage', 'step',
                                                      'average_sample_num', 'measure_spacing', 'measure_delay'])

    filename = '%(data_dir)s\\%(sample)s_%(measurement_type)s_%(start_voltage)sV_%(stop_voltage)sV_%(other_desc)s' % \
               {'data_dir': config.main['data_dir'], 'sample': config.main['sample'], 'measurement_type': 'iv',
                'start_voltage': start_voltage, 'stop_voltage': stop_voltage, 'other_desc': other_desc}

    if check_filename:
        check_data_files_exist(filename+'.config')

    voltages = np.arange(start_voltage, stop_voltage+step, step)
    currents = np.zeros((voltages.size, average_sample_num))

    smu.apply_voltage(np.max([abs(start_voltage), abs(stop_voltage)]), compliance_current=compliance_current)
    smu.source_voltage = 0
    smu.enable_source()
    smu.measure_current()

    config.main['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    plt.figure()
    fig_num = plt.gcf().number

    for idx, voltage in enumerate(voltages):
        print('Measure at %fV (%d/%d)' % (voltage, idx, voltages.size), end='\r')
        smu.source_voltage = voltage
        time.sleep(measure_delay)
        flag = 0

        for sample_idx in range(0, average_sample_num):
            currents[idx, sample_idx] = smu.current
            if currents[idx, sample_idx] > stop_current:
                flag = 1
                break

            time.sleep(measure_spacing)

        if flag:
            break

        if not plt.fignum_exists(fig_num):
            break

        plt.figure(fig_num)
        plt.clf()
        plt.errorbar(voltages[:idx+1], np.mean(currents[:idx+1, :], axis=1)*1e9,
                     yerr=np.std(currents[:idx+1, :], axis=1)*1e9)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (nA)')
        plt.pause(0.1)

    print('Finished!')
    smu.shutdown()

    np.savetxt(filename+'.csv', np.hstack((np.array([voltages]).transpose(), currents)), delimiter=",")
    config.save_config(filename+'.config')

    return voltages, currents


def spe_measurement(config, pi, frames, exposure_time=5000, other_desc=None, check_filename=True):
    config = config.copy()
    config.main['measurement_type'] = 'spe'

    if other_desc is None:
        other_desc = ''

    filename_spe = format_filename(config.main['sample'], config.main['dot_num'], config.main['state'],
                                   config.main['measurement_type'], config.main['laser_linewidth'], config.main['cwl'],
                                   config.main['power'], exposure_time/1000, other_desc=other_desc, suffix='')
    filename_spe = filename_spe[:-1]
    filename_config = config.main['data_dir']+'\\'+filename_spe+'.config'

    if check_filename:
        check_data_files_exist(filename_spe)

    pi.set_file_path(config.main['data_dir'])

    config.main['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    pi.exposure_time(exposure_time)
    pi.frames(frames)
    pi.file_name(filename_spe)
    pi.acquire()

    config.main['dot_num'] = config.main['dot_num']
    config.main['power'] = config.main['power']

    config.save_config(filename_config)


def cv_measurement(config, pi, smu, V_l, V_h, V_step, cycles, exposure_time=5.0, frames=1, other_desc='',
                   compliance_current=1e-3, check_filename=True):
    # Write basic params into config
    config = config.copy()
    config.main['measurement_type'] = 'cv_spe'
    config.export_measurement_config('cv_spe', locals(), ['V_l', 'V_h', 'V_step', 'cycles',
                                                          'exposure_time', 'frames'])

    # Handling the filename and work directory
    if other_desc is None:
        other_desc = ''
    other_desc = '%dV_%dV_%.0fmVs-1_cv_' % (V_h, V_l, V_step/exposure_time*1e3)+other_desc
    bundle_name = format_filename(config.main['sample'], config.main['dot_num'], config.main['state'], 'cvspe',
                                  config.main['laser_linewidth'], config.main['cwl'], config.main['power'],
                                  config.cv_spe['exposure_time'], other_desc=other_desc, suffix='')
    bundle_name = bundle_name[:-1]

    bundle_config = config.main['data_dir']+'\\'+bundle_name+'.bundle'
    if check_filename:
        check_data_files_exist(bundle_config)

    os.makedirs(config.main['data_dir']+'\\'+bundle_name)
    # Code to manage duplicate folder and files inside should be add here!

    # Generate voltages sequence
    amplitude = abs(V_l-V_h)
    frequency = 1/(2*amplitude/V_step*exposure_time)
    t = np.arange(0, cycles*(2*amplitude/V_step*exposure_time)+exposure_time, exposure_time)
    voltages = waveform_generate('triangle', t, amplitude, frequency, offset=0, phase=-90)

    # Perform spectrum measurement at different voltages
    config.main['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    currents = spe_voltages_measurement(pi, smu, voltages, config.main['data_dir'], bundle_name, exposure_time, frames, compliance_current)

    # Write params into config and save config
    config.cv_spe['t'] = list(t)
    config.cv_spe['voltages'] = list(voltages)
    config.cv_spe['currents'] = list(currents)
    config.main['dot_num'] = config.main['dot_num']
    config.main['power'] = config.main['power']
    config.save_config(bundle_config)


def ca_measurement(config, pi, smu, V_base, V_drive, t_base, t_drive, cycles, exposure_time=5.0, frames=1,
                   other_desc=None, compliance_current=1e-3, check_filename=True):

    # Write basic params into config
    config = config.copy()
    config.main['measurement_type'] = 'ca_spe'
    config.export_measurement_config('ca_spe', locals(), ['V_base', 'V_drive', 't_base', 't_drive', 'cycles',
                                                          'exposure_time', 'frames'])

    # Handling the filename and work directory
    if other_desc is None:
        other_desc = ''
    other_desc = '%dV_%dV_%.1fs_%.1fs_ca_' % (V_base, V_drive, t_base, t_drive)+other_desc
    bundle_name = format_filename(config.main['sample'], config.main['dot_num'], config.main['state'], 'caspe',
                                  config.main['laser_linewidth'], config.main['cwl'], config.main['power'],
                                  config.ca_spe['exposure_time'], other_desc=other_desc, suffix='')
    bundle_name = bundle_name[:-1]

    bundle_config = config.main['data_dir']+'\\'+bundle_name+'.bundle'
    if check_filename:
        check_data_files_exist(bundle_config)

    os.makedirs(config.main['data_dir']+'\\'+bundle_name)
    # Code to manage duplicate folder and files inside should be add here!

    # Generate voltages sequence
    frames_base = int(np.around(t_base/exposure_time))
    if frames_base*exposure_time != t_base:
        warn('The t_base was around from %.3fs to %.3fs' % (t_base, exposure_time*frames_base))
        t_base = exposure_time*frames_base

    frames_drive = int(np.around(t_drive/exposure_time))
    if frames_drive*exposure_time != t_drive:
        warn('The t_drive was around from %.3fs to %.3fs' % (t_drive, exposure_time*frames_drive))
        t_base = exposure_time*frames_drive

    voltages_unique = np.hstack((np.ones(frames_base)*V_base, np.ones(frames_drive)*V_drive))
    voltages = np.tile(voltages_unique, cycles)

    t = np.arange(0, cycles*(t_base+t_drive), exposure_time, dtype='float_')

    # Perform spectrum measurement at different voltages
    config.main['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    currents = spe_voltages_measurement(pi, smu, voltages, config.main['data_dir'], bundle_name, exposure_time, frames, compliance_current)

    # Write params into config and save config
    config.ca_spe['t'] = list(t)
    config.ca_spe['voltages'] = list(voltages)
    config.ca_spe['currents'] = list(currents)
    config.main['dot_num'] = config.main['dot_num']
    config.main['power'] = config.main['power']
    config.save_config(bundle_config)


def spe_voltages_measurement(pi, smu, voltages, data_dir, bundle_name, exposure_time, frames, compliance_current):
    currents = np.zeros(voltages.shape)
    pi.set_file_path(data_dir+'\\'+bundle_name)

    smu.apply_voltage(np.max(np.abs(voltages)), compliance_current=compliance_current)
    smu.source_voltage = 0
    smu.enable_source()
    smu.measure_current()

    for idx, voltage in enumerate(voltages):
        print('Measure at %fV (%d/%d)' % (voltage, idx, voltages.size), end='\r')

        filename_spe = bundle_name+'_%d' % idx
        smu.source_voltage = voltage

        currents[idx] = smu.current
        if currents[idx] > compliance_current:
            print('Current overload!')
            break

        pi.exposure_time(exposure_time*1000)
        pi.frames(frames)
        pi.file_name(filename_spe)
        pi.acquire()

    smu.source_voltage = 0
    print('Finished!')
    pi.set_file_path(data_dir)
    pi.file_name('untitled')

    return currents


def lifetimetrace_measurement(config, tagger, click_channel, start_channel, binwidth, n_bins, int_time,
                              capture_duration, offset, max_delay, min_delay, other_desc='', check_filename=True):
    config = config.copy()
    config.main['measurement_type'] = 'lifetimetrace'

    exposure_time = '%dm' % (int_time / 1e9)

    filename_base = format_filename(config.main['sample'], config.main['dot_num'], config.main['state'],
                                    config.main['measurement_type'], config.main['laser_linewidth'], config.main['cwl'],
                                    config.main['power'], exposure_time, other_desc=other_desc, suffix='')
    filename_base = filename_base[:-1]

    filename_ttbin = config.main['data_dir'] + '\\' + filename_base + '.ttbin'
    filename_csv = config.main['data_dir'] + '\\' + filename_base + '.csv'
    filename_config = config.main['data_dir'] + '\\' + filename_base + '.config'

    if check_filename:
        check_data_files_exist(filename_ttbin)

    meas = LifetimeTraceWithFileWriter(tagger, click_channel, start_channel, binwidth, n_bins, int_time, filename_ttbin,
                                       offset=offset, max_delay=max_delay, min_delay=min_delay)

    config.main['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # Cyclic voltammetry lifetime trace measurement
    meas.startForSecond(capture_duration)
    plot_lifetime_trace(meas)

    # Save csv data
    save_data(filename_csv, meas)

    config.export_measurement_config('lifetime_trace', meas)
    config.save_config(filename_config)

    del meas


class Measurement:
    def __init__(self, devices, devices_params=None, file_params=None, other_params=None):
        self.devices = devices
        self.devices_params = {}
        self.file_params = {}
        self.other_params = {}
        self.config = MeasurementConfig2()

        self.reset_to_default()

        self.add_params(self.devices_params, deepcopy(devices_params))
        self.add_params(self.file_params, deepcopy(file_params))
        self.add_params(self.other_params, deepcopy(other_params))

        self.data = None

    @staticmethod
    def add_params(old_params, new_params):
        if new_params is None:
            return

        for key in new_params:
            old_params[key] = new_params[key]

    def reset_to_default(self):
        self.devices_params = {}
        self.file_params = {}
        self.other_params = {}

        self.config.devices_params = self.devices_params
        self.config.file_params = self.file_params
        self.config.other_params = self.other_params

        self.data = None

    def start(self, *args, **kwargs):
        pass

    def get_data(self):
        return self.data

    def save_config(self, filename):
        self.config.save_config(filename)

    def load_config(self, filename):
        self.config.load_config(filename)

        self.devices_params = self.config.devices_params
        self.file_params = self.config.file_params
        self.other_params = self.config.other_params


class SpeLifetimetraceVoltagesMeasurement(Measurement):
    def __init__(self, devices, devices_params, file_params, other_params=None):
        Measurement.__init__(self, devices, devices_params, file_params, other_params)
        self.lifetime_meas = None

    def reset_to_default(self):
        self.devices_params = {
            'pi': {
                'exposure_time': 5.0,
                'frames': 1
            },
            'smu': {
                'voltages': [],
                'compliance_current': 1e-3
            },
            'tagger': {
                'click_channel': 1,
                'start_channel': 3,
                'gate_on_channel': 4,
                'gate_off_channel': -4,
                'binwidth': 50,
                'n_bins': 2000,
                'int_time': int(0.1e12),
                'offset': 136000,
                'min_delay': 750,
                'max_delay': 50000,
                'enable_global_delay': True,
                'enable_conditional_filter': True
            }
        }
        self.file_params = {
            'config': {},
            'data_dir': '',
            'bundle_name': 'untitled',
            'other_desc': None,
            'check_filename': True,
            'export_config': True,
            'export_ttbin': True,
        }
        self.other_params = {
            'dot_num': 1,
            'power': '50',
            'sample': 'test',
            'state': 1,
            'laser_linewidth': 'pulse',
            'cwl': '630',
            'laser_frequency': 10e6
        }

        self.config.devices_params = self.devices_params
        self.config.file_params = self.file_params
        self.config.other_params = self.other_params

        self.data = None

    def start(self):
        pi = self.devices['pi']
        smu = self.devices['smu']
        tagger = self.devices['tagger']
        voltages = np.array(self.devices_params['smu']['voltages'])
        bundle_full_path = self.file_params['data_dir']+'\\'+self.file_params['bundle_name']
        # use trigger from the PI spectrometer to mark the start and end of each frame.

        self.devices_params['smu']['currents'] = list(np.zeros(voltages.shape))
        currents = self.devices_params['smu']['currents']
        pi.set_file_path(bundle_full_path)

        smu.apply_voltage(np.max(np.abs(voltages)), compliance_current=self.devices_params['smu']['compliance_current'])
        smu.source_voltage = 0
        smu.enable_source()
        smu.measure_current()

        # lifetime_trace_measurement code here
        if self.devices_params['tagger']['enable_conditional_filter']:
            self.enable_conditional_filter()

        if self.file_params['export_ttbin']:
            self.lifetime_meas = LifetimeTraceGatedWithFileWriter(
                tagger,
                self.devices_params['tagger']['click_channel'],
                self.devices_params['tagger']['start_channel'],
                self.devices_params['tagger']['gate_on_channel'],
                self.devices_params['tagger']['gate_off_channel'],
                self.devices_params['tagger']['binwidth'],
                self.devices_params['tagger']['n_bins'],
                self.devices_params['tagger']['int_time'],
                bundle_full_path+'.ttbin',
                self.devices_params['tagger']['offset'],
                self.devices_params['tagger']['min_delay'],
                self.devices_params['tagger']['max_delay']
            )
        else:
            self.lifetime_meas = LifetimeTraceGated(
                tagger,
                self.devices_params['tagger']['click_channel'],
                self.devices_params['tagger']['start_channel'],
                self.devices_params['tagger']['gate_on_channel'],
                self.devices_params['tagger']['gate_off_channel'],
                self.devices_params['tagger']['binwidth'],
                self.devices_params['tagger']['n_bins'],
                self.devices_params['tagger']['int_time'],
                self.devices_params['tagger']['offset'],
                self.devices_params['tagger']['min_delay'],
                self.devices_params['tagger']['max_delay']
            )

        self.lifetime_meas.stop()
        laser_period = 1e12/self.other_params['laser_frequency']
        if self.devices_params['tagger']['offset'] > laser_period:
            original_delay = tagger.getDelayHardware(self.devices_params['tagger']['start_channel'])
            tagger.setDelayHardware(self.devices_params['tagger']['start_channel'],
                                    original_delay+laser_period*np.floor(self.devices_params['tagger']['offset']/laser_period))

        if self.devices_params['tagger']['enable_global_delay']:
            self.enable_global_software_offset()
            self.lifetime_meas.offset = 0
            if self.file_params['export_ttbin']:
                self.lifetime_meas.lifetime_trace_gated.offset = 0

        self.lifetime_meas.clear()
        self.lifetime_meas.start()
        for idx, voltage in enumerate(voltages):
            print('Measure at %fV (%d/%d)' % (voltage, idx, voltages.size), end='\r')

            filename_spe = self.file_params['bundle_name']+'_%d' % idx
            smu.source_voltage = voltage

            currents[idx] = smu.current
            if currents[idx] > self.devices_params['smu']['compliance_current']:
                print('Current overload!')
                break

            pi.exposure_time(self.devices_params['pi']['exposure_time']*1000)
            pi.frames(self.devices_params['pi']['frames'])
            pi.file_name(filename_spe)
            pi.acquire()

            self.plot_data()

        self.lifetime_meas.stop()
        if self.devices_params['tagger']['enable_global_delay']:
            self.disable_global_software_offset()

        if self.devices_params['tagger']['enable_conditional_filter']:
            self.disable_conditional_filter()

        smu.source_voltage = 0
        print('Finished!')
        pi.set_file_path(self.file_params['data_dir'])
        pi.file_name('untitled')

        self.lifetime_meas.saveData(bundle_full_path+'_lifetime.csv', bundle_full_path+'_hists.csv')
        if self.file_params['export_config']:
            self.save_config(bundle_full_path+'.config')

    def enable_global_software_offset(self):
        laser_period = 1e12/self.other_params['laser_frequency']
        original_delay = self.devices['tagger'].getDelaySoftware(self.devices_params['tagger']['start_channel'])
        self.devices['tagger'].setDelaySoftware(self.devices_params['tagger']['start_channel'],
                                                self.devices_params['tagger']['offset']%laser_period+original_delay)

    def disable_global_software_offset(self):
        laser_period = 1e12/self.other_params['laser_frequency']
        original_delay = self.devices['tagger'].getDelaySoftware(self.devices_params['tagger']['start_channel'])
        self.devices['tagger'].setDelaySoftware(self.devices_params['tagger']['start_channel'],
                                                original_delay-self.devices_params['tagger']['offset']%laser_period)

    def plot_data(self):
        pass

    def enable_conditional_filter(self):
        enable_conditional_filter(self.devices['tagger'],
                                  self.devices_params['tagger']['click_channel'],
                                  self.devices_params['tagger']['start_channel'],
                                  self.other_params['laser_frequency'])

    def disable_conditional_filter(self):
        disable_conditional_filter(self.devices['tagger'], self.devices_params['tagger']['start_channel'])


class SpeLifetimetraceCVMeasurement(SpeLifetimetraceVoltagesMeasurement):
    def __init__(self, devices, devices_params=None, file_params=None, other_params=None):
        SpeLifetimetraceVoltagesMeasurement.__init__(self, devices, devices_params,
                                                     file_params, other_params)

    def reset_to_default(self):
        SpeLifetimetraceVoltagesMeasurement.reset_to_default(self)

        additional_params = {
                                'V_l': -5,
                                'V_h': 5,
                                'V_step': 0.2,
                                'cycles': 1,
                            }
        self.add_params(self.devices_params['smu'], additional_params)

    def start(self):
        # Quick access to certain params
        V_l = self.devices_params['smu']['V_l']
        V_h = self.devices_params['smu']['V_h']
        V_step = self.devices_params['smu']['V_step']
        exposure_time = self.devices_params['pi']['exposure_time']
        cycles = self.devices_params['smu']['cycles']
        export_config = self.file_params['export_config']

        # Handling the filename and work directory
        if self.file_params['other_desc'] is None:
            other_desc = ''
        else:
            other_desc = self.file_params['other_desc']

        other_desc = '%dV_%dV_%.0fmVs-1_cv_' % (V_h, V_l, V_step/exposure_time*1e3)+other_desc
        bundle_name = format_filename(self.other_params['sample'], self.other_params['dot_num'],
                                      self.other_params['state'], 'cvspe', self.other_params['laser_linewidth'],
                                      self.other_params['cwl'], self.other_params['power'],
                                      exposure_time, other_desc=other_desc, suffix='')
        bundle_name = bundle_name[:-1]
        self.file_params['bundle_name'] = bundle_name

        bundle_config = self.file_params['data_dir']+'\\'+bundle_name+'.bundle'
        if self.file_params['check_filename']:
            check_data_files_exist(bundle_config)

        os.makedirs(self.file_params['data_dir']+'\\'+bundle_name)
        # Code to manage duplicate folder and files inside should be added here!

        # Generate voltages sequence
        amplitude = abs(V_l-V_h)
        frequency = 1/(2*amplitude/V_step*exposure_time)
        t = np.arange(0, cycles*(2*amplitude/V_step*exposure_time)+exposure_time, exposure_time)
        voltages = waveform_generate('triangle', t, amplitude, frequency, offset=0, phase=-90)
        self.devices_params['smu']['voltages'] = list(voltages)

        # Perform spectrum measurement at different voltages
        self.other_params['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self.file_params['export_config'] = False
        SpeLifetimetraceVoltagesMeasurement.start(self)
        self.file_params['export_config'] = export_config

        # Write params into config and save config
        self.other_params['measurement_type'] = 'cv_spe'
        self.other_params['t'] = list(t)
        self.other_params['dot_num'] = self.other_params['dot_num']
        self.other_params['power'] = self.other_params['power']

        if export_config:
            self.save_config(bundle_config)


class SpeLifetimetraceCAMeasurement(SpeLifetimetraceVoltagesMeasurement):
    def __init__(self, devices, devices_params=None, file_params=None, other_params=None):
        SpeLifetimetraceVoltagesMeasurement.__init__(self, devices, devices_params,
                                                     file_params, other_params)

    def reset_to_default(self):
        SpeLifetimetraceVoltagesMeasurement.reset_to_default(self)

        additional_params = {
                                'V_base': 0,
                                'V_drive': 5,
                                't_base': 60,
                                't_drive': 60,
                                'cycles': 1,
                            }
        self.add_params(self.devices_params['smu'], additional_params)

    def start(self):
        # Quick access to certain params
        V_base = self.devices_params['smu']['V_base']
        V_drive = self.devices_params['smu']['V_drive']
        t_base = self.devices_params['smu']['t_base']
        t_drive = self.devices_params['smu']['t_drive']
        exposure_time = self.devices_params['pi']['exposure_time']
        cycles = self.devices_params['smu']['cycles']
        export_config = self.file_params['export_config']

        # Handling the filename and work directory
        if self.file_params['other_desc'] is None:
            other_desc = ''
        else:
            other_desc = self.file_params['other_desc']

        other_desc = '%dV_%dV_%.1fs_%.1fs_ca_' % (V_base, V_drive, t_base, t_drive)+other_desc
        bundle_name = format_filename(self.other_params['sample'], self.other_params['dot_num'],
                                      self.other_params['state'], 'caspe', self.other_params['laser_linewidth'],
                                      self.other_params['cwl'], self.other_params['power'],
                                      exposure_time, other_desc=other_desc, suffix='')
        bundle_name = bundle_name[:-1]
        self.file_params['bundle_name'] = bundle_name

        bundle_config = self.file_params['data_dir']+'\\'+bundle_name+'.bundle'
        if self.file_params['check_filename']:
            check_data_files_exist(bundle_config)

        os.makedirs(self.file_params['data_dir']+'\\'+bundle_name)
        # Code to manage duplicate folder and files inside should be added here!

        # Generate voltages sequence
        frames_base = int(np.around(t_base/exposure_time))
        if frames_base*exposure_time != t_base:
            warn('The t_base was around from %.3fs to %.3fs' % (t_base, exposure_time*frames_base))
            t_base = exposure_time*frames_base

        frames_drive = int(np.around(t_drive/exposure_time))
        if frames_drive*exposure_time != t_drive:
            warn('The t_drive was around from %.3fs to %.3fs' % (t_drive, exposure_time*frames_drive))
            t_base = exposure_time*frames_drive

        voltages_unique = np.hstack((np.ones(frames_base)*V_base, np.ones(frames_drive)*V_drive))
        voltages = np.tile(voltages_unique, cycles)

        t = np.arange(0, cycles*(t_base+t_drive), exposure_time, dtype='float_')
        self.devices_params['smu']['voltages'] = list(voltages)

        # Perform spectrum measurement at different voltages
        self.other_params['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self.file_params['export_config'] = False
        SpeLifetimetraceVoltagesMeasurement.start(self)
        self.file_params['export_config'] = export_config

        # Write params into config and save config
        self.other_params['measurement_type'] = 'ca_spe'
        self.other_params['t'] = list(t)
        self.other_params['dot_num'] = self.other_params['dot_num']
        self.other_params['power'] = self.other_params['power']

        if export_config:
            self.save_config(bundle_config)
