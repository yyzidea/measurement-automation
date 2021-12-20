from functools import partial
import os
import re
import numpy as np
from warnings import warn


def format_filename(sample, dot_num, state, measurement_type, laser_linewidth, cwl, power, exposure_time,
                    other_desc='', suffix='csv'):
    filename = '%(sample)s_%(dot_num)d_%(state)d_%(measurement_type)s_%(laser_linewidth)s_%(cwl)snm_%(power)snw_' \
            '%(exposure_time)ss_%(other_desc)s.%(suffix)s' % {'sample': sample, 'dot_num': dot_num, 'state': state,
                                                             'measurement_type': measurement_type,
                                                             'laser_linewidth': laser_linewidth, 'cwl': cwl,
                                                             'power': power,
                                                             'exposure_time': exposure_time,
                                                             'other_desc': other_desc, 'suffix': suffix}
    filename = filename.replace('__', '_').replace('_.', '.')

    return filename


def format_spec_filename(sample, dot_num, state, measurement_type, laser_linewidth, cwl, power, exposure_time,
                         other_desc='', suffix='spec'):
    return format_filename(sample, dot_num, state, measurement_type, laser_linewidth, cwl, power, exposure_time,
                           other_desc, suffix)


def list_data_files(data_dir, prefix, file_suffix, iterator=r'([\d\.]+).*?'):
    # data_dir = '/Users/yyzidea/Desktop/untitled folder';
    # prefix = 'lll_1_0.4nm_685_2s_';
    # file_suffix = '.spe';

    data_iterators = np.empty(0)

    dir_files = os.listdir(data_dir)
    data_filename = prefix+iterator+file_suffix
    data_files = []

    for file in dir_files:
        res = re.findall(data_filename, file)
        if len(res) != 0:
            if '(' in iterator and ')' in iterator:
                data_iterators = np.append(data_iterators, int(res[0]))
            data_files.append(file)

    if '(' in iterator and ')' in iterator:
        I = np.argsort(data_iterators)
        data_iterators = np.sort(data_iterators)
        temp = []
        if os.name is 'posix':
            for i in range(0, I.size):
                temp.append(data_dir+'/'+data_files[I[i]])
        else:
            for i in range(0, I.size):
                temp.append(data_dir+'\\'+data_files[I[i]])

        data_files = temp

    return data_files, data_iterators


def check_data_files_exist(filename):
    if os.path.exists(filename):
        while 1:
            choice = input('File existed: %s \nDo you want to overwrite it? (Y/N)' % filename)
            if choice in 'Yy':
                warn('Data file have been overwritten: %s' % filename)
                return True
            elif choice in 'Nn':
                raise Exception('File existed, please change the filename.')
    else:
        return False


_rename_history = []


def batch_rename(old_files, old_str, new_str):
    global _rename_history
    new_files = []
    file_map = []

    for idx, old_file in enumerate(old_files):
        new_files.append(old_file.replace(old_str, new_str))
        os.rename(old_file, new_files[-1])
        file_map.append([old_file, new_files[-1]])

    _rename_history.append(file_map)

    return file_map


def redo_batch_rename(count=1):
    global _rename_history
    while count > 0 and len(_rename_history) > 0:
        file_map = _rename_history.pop()
        for i in np.arange(len(file_map)-1, -1, -1):
            os.rename(file_map[i][1], file_map[i][0])

        count -= 1
