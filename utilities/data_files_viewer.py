import matplotlib.pyplot as plt
import numpy as np
import os
from utilities.data_files_handling import *


def spe_bundle_viewer(bundle_filename, count_range=None):
    data_dir, bundle_name = os.path.split(bundle_filename)
    bundle_name = os.path.splitext(bundle_name)
    bundle_name = bundle_name[0]

    files, iterators = list_data_files(data_dir + '\\' + bundle_name, bundle_name, '.csv', r'_(\d*?)')

    data = np.empty(0)
    print('Merging existed bundle files: %s' % bundle_name)
    for idx, file in enumerate(files):
        temp_data = np.loadtxt(file, delimiter=',')

        if data.size == 0:
            data = temp_data
        else:
            data = np.hstack((data, np.array([temp_data[:, 1]]).transpose()))

        if (idx + 1) % 10 == 0:
            print('Progress: %d/%d' % (idx + 1, len(files)), end='\r')

    print('Bundle merge complete! Total subfiles:%d' % len(files))

    plt.figure()
    fig_num = plt.gcf().number
    plt.imshow(data[:, 1:-1], aspect='auto', extent=[0, data.shape[1]-1, data[-1, 0], data[0, 0]])
    if count_range is not None:
        plt.clim(count_range)
    plt.colorbar()
    plt.xlabel('Voltages (V)')
    plt.ylabel('Wavelength (nm)')

    idx += 1
    while plt.fignum_exists(fig_num):
        if os.path.exists(bundle_filename):
            locs, labels = plt.xticks()
            locs = np.array(np.around(locs), dtype='int')
            config = load_config(bundle_filename)

            if 'other_params' in config.keys():
                if locs[-1] >= len(config['devices_params']['smu']['voltages']):
                    locs = locs[:-1]
                plt.xticks(locs + 0.5, np.array(config['devices_params']['smu']['voltages'])[locs])
            else:
                if config['main']['measurement_type'] == 'cv_spe':
                    if locs[-1] >= len(config['cv_spe']['voltages']):
                        locs = locs[:-1]
                    plt.xticks(locs+0.5, np.array(config['cv_spe']['voltages'])[locs])
                else:
                    if locs[-1] >= len(config['ca_spe']['voltages']):
                        locs = locs[:-1]
                    plt.xticks(locs+0.5, np.array(config['ca_spe']['voltages'])[locs])
            break

        data_filename = data_dir + '\\' + bundle_name + '\\' + bundle_name + '_%d.csv' % idx
        if os.path.exists(data_filename):
            try:
                temp_data = np.loadtxt(data_filename, delimiter=',')
                data = np.hstack((data, np.array([temp_data[:, 1]]).transpose()))
                idx += 1
            except ValueError:
                pass
            except IndexError:
                pass
            finally:
                pass

        plt.figure(fig_num)
        plot_spe(data, count_range)


def plot_spe(data, count_range):
    plt.clf()
    plt.imshow(data[:, 1:-1], aspect='auto', extent=[0, data.shape[1], data[-1, 0], data[0, 0]])

    if count_range is not None:
        plt.clim(count_range)

    plt.colorbar()

    plt.pause(0.2)
    plt.xlabel('Voltages (V)')
    plt.ylabel('Wavelength (nm)')



