import sys
import os
import yaml
import pickle
from typing import Callable
from ppit.src.exception import CustomException
from ppit.src.logger import logging
from dataclasses import dataclass
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from functools import wraps
import time

def timing(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper
@dataclass
class AttrDict(dict):
    """
    Dictionary subclass whose entries can be accessed by attributes (as well
        as normally).
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


def WritePickle(data, file_path=''):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def WriteYAML(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
    return

import time

def writer(
        result: dict, output_path: str
) -> Callable[[str], None]:
    output_format = os.path.basename(output_path).split(".")[-1]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writers = {
        "p": WritePickle,
        "yml": WriteYAML,
    }
    return writers[output_format](result, output_path)


def reader(
        input_path: str
) -> Callable[[str], None]:
    input_path = os.path.normpath(input_path)
    input_format = os.path.basename(input_path).split(".")[-1]
    readers = {
        "p": ReadPickle,
        "yml": ReadYAML,
        "csv": ReadCSV,
    }
    return readers[input_format](input_path)

def ReadCSV(file_path: str):
    df = pd.read_csv(file_path)
    return df

def ReadYAML(file_path: str):
    with open(file_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            logging.info(f"Successfully load data from {file_path}")
            return data
        except yaml.YAMLError as e:
            raise CustomException(e, sys)


def ReadPickle(file_path: str):
    with open(file_path, 'rb') as stream:
        try:
            data = pickle.load(stream)
            logging.info(f"Successfully load data from {file_path}")
            return data
        except yaml.YAMLError as e:
            raise CustomException(e, sys)

def join_path(*args):
    paths = [os.path.normpath(arg) for arg in args]
    element_in_path = []
    for path in paths:
        for element in path.split(os.sep):
            element_in_path.append(element)
    local_path = os.path.join(*element_in_path)
    return local_path

def none_or_str(value):
    if value is None:
        return value
    if value.lower() == 'none':
        return None
    return value

def _check_file_unique_exist(file_path):
    import glob
    for file in glob.iglob(file_path):
        try:
            return file
        except:
            raise FileNotFoundError('Check the file path %s' % file_path)

def scan_folder(path, keyword="model"):
    if os.path.isdir(path):
        import glob
        return glob.glob(os.path.join(path, "*"+keyword+"*"))
    else:
        return [path]


def basename(file_path):
    return os.path.basename(file_path).split(".")[0]

def load_config(configuration_path, folder="model"):
    config_filename = basename(configuration_path)
    config = AttrDict.from_nested_dicts(reader(configuration_path))
    sub_folder_path = os.path.join(os.path.dirname(configuration_path), folder, config_filename)
    os.makedirs(sub_folder_path, exist_ok=True)
    logging.info(f"Creating {folder} folder {config_filename}")
    return (config_filename, config, sub_folder_path)


def has_nan(data):
    if isinstance(data, list):
        for item in data:
            if has_nan(item):
                return True
        return False
    elif isinstance(data, dict):
        for key, value in data.items():
            if has_nan(value):
                return True
        return False
    else:
        return data is None or (isinstance(data, float) and np.isnan(data))

if __name__=="__main__":
    pass