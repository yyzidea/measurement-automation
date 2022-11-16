import matplotlib.pyplot as plt
import numpy as np
import os
from utilities.data_files_handling import *
from warnings import warn

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


def read_spe_csv(filename, mode='all', noise_level=0):
    raw_data = np.loadtxt(filename, delimiter=',')

    if raw_data.shape[1] == 2:
        date_length = raw_data.shape[0]
        frame_size = 1340
        while date_length % frame_size:
            frame_size = frame_size - 1

        data = np.hstack((np.atleast_2d(raw_data[0:frame_size, 0]).transpose(),
                          np.reshape(raw_data[:, 1], (int(date_length/frame_size), frame_size)).transpose()))
        frameLen = date_length / frame_size
    # elif raw_data.shape[1] == 3:
    #     frames = unique(raw_data(:, 3))'
    #     data = raw_data(raw_data(:, 3) == frames(1), 1: 2)
    #     for i = frames(2:end)
    #     data(:, end + 1) = raw_data(raw_data(:, 3) == i, 2)
    #     frameLen = length(frames)
    else:
        data = raw_data
        frameLen = raw_data.shape[1] - 1

    if mode == 'acc':
        data = np.vstack((data[:, 0], np.sum(data[:, 1:-1], 2)))
    elif mode == 'mean':
        data = np.vstack((data[:, 0], np.sum(data[:, 1: -1], 2)))
        data[:, 2] = data[:, 2] / frameLen
    elif mode == 'raw':
        data = raw_data
    elif mode != 'all':
        warn('Illegal mode: %s, and output with default setting.', mode)

    if noise_level == 'auto':
        noise_level = np.mean(data[-101:-1, 1])

    data[:, 1:-1] = data[:, 1:-1]-noise_level

    return data


def find_spec_peak(data: np.ndarray, spike_limit=500, spike_width=4):
    peak = 0
    peak_idx = 0
    if data.ndim == 2:
        frame_dim = np.argmin(data.shape)
        frame_len = data.shape[frame_dim]
        peak_frame_idx = 0
        for frame in np.arange(0, frame_len):
            if frame_dim:
                temp_peak, temp_peak_idx = find_spec_peak(data[:, frame], spike_limit, spike_width)
            else:
                temp_peak, temp_peak_idx = find_spec_peak(data[frame, :], spike_limit, spike_width)
            if peak < temp_peak:
                peak = temp_peak
                peak_idx = temp_peak_idx
                peak_frame_idx = frame
        peak_idx = (peak_frame_idx, peak_idx)
    else:
        data = data.copy()
        while 1:
            peak = np.max(data)
            peak_idx = np.argmax(data)
            if peak > spike_limit:
                data[int(np.floor(peak_idx-spike_width/2)):int(np.ceil(peak_idx+spike_width/2))] = 0
            else:
                break

    return peak, peak_idx


def plot_spe_from_file(filename, count_range=None):
    spe = read_spe_csv(filename)
    if count_range is None:
        count_range = [0, find_spec_peak(spe[:, 1:-1])[0]]
    plot_spe(spe, count_range)
