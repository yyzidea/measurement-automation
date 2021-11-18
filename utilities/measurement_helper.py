import json
from warnings import warn

# def init_from_config(meas_cls, config: dict):
#     arg_str = ''
#
#     for key, value in config.items():
#         arg_str = key+'='+value


def export_measurement_config(obj, attr_keys=None):
    if attr_keys is None:
        attr_keys = obj.__init__.__code__.co_varnames

    params = {}
    for key in attr_keys:
        if key != 'self':
            if isinstance(obj, dict):
                param = obj[key]
            else:
                param = obj.__getattribute__(key)

            if param.__class__.__name__ in ['dict', 'list', 'tuple', 'str', 'int',
                                            'float', 'bool', 'NoneType']:
                params[key] = param
            else:
                warn('The parameter \'%s\' of type \'%s\' is not JSON serializable and is skipped.' %
                     (key, param.__class__.__name__))

    return params


def save_config(config, filename):
    with open(filename, 'w') as fp:
        json.dump(config, fp, indent='\t')


def load_config(filename):
    with open(filename, 'r') as fp:
        config = json.load(fp)

    return config
