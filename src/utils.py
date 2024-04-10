import numpy as np
import sys
import os
import yaml
import pickle
from typing import Callable
from src.exception import CustomException
from src.logger import logging

ROOT = os.path.dirname(os.getcwd())


def WritePickle(data, file_path=''):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def WriteYAML(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
    return


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
    input_format = os.path.basename(input_path).split(".")[-1]
    readers = {
        "p": ReadPickle,
        "yml": ReadYAML,
    }
    return readers[input_format](input_path)


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


def reformat_pressure_map(df):
    """
    Find the data structure in the df. The index should be time
    reshape the df to array (time_frame, row, col)
    """
    data_structure = {
        'pressure_map': (
            len(set(col_name.split('_')[0]
                    for col_name in df.columns)),
            len(set(col_name.split('_')[1]
                    for col_name in df.columns))
        )
    }

    data = {}
    for data_type in data_structure:
        data[data_type] = np.zeros((len(df),) + data_structure[data_type])
        for iy, ix in np.ndindex(data_structure[data_type]):
            data[data_type][:, iy, ix] = df['row%d_col%d_%s' % (iy, ix, data_type)]
    # The shape here is (time_frame, row, col)
    return data


def reformat_sensor(df):
    """
    Find the data structure in the df. The index should be time
    reshape the df to array (time_frame, row, col)
    """
    data_structure = {
        data_type: (
            len(set(col_name.split('_')[0]
                    for col_name in df.columns
                    if data_type in col_name)),
            len(set(col_name.split('_')[1]
                    for col_name in df.columns
                    if data_type in col_name))
        ) for data_type in set(col_name.split('_', 2)[-1]
                               for col_name in df.columns)
    }

    data = {}
    for data_type in data_structure:
        data[data_type] = np.zeros((len(df),) + data_structure[data_type])
        for iy, ix in np.ndindex(data_structure[data_type]):
            data[data_type][:, iy, ix] = df['row%d_col%d_%s' % (iy, ix, data_type)]
    # The shape here is (time_frame, row, col)
    return data


def reformat_keypoint(df):
    """
    change the keypoint dataframe to numpy array (time frames, joints, xyz)
    """
    num_joint = int(len(df.columns) / 3)
    data = {'keypoint': np.empty((num_joint, len(df), 3))}
    for idx in range(1, num_joint + 1):
        joint_df = df.filter(regex='^' + str(idx) + ' ')
        df = df.drop(joint_df.columns, axis=1)
        joint_data = np.hstack([joint_df.filter(regex=axis + '$').to_numpy() for axis in ['x', 'y', 'z']])
        data['keypoint'][idx - 1, :, :] = joint_data
        # The shape here is (joints, time frames, xyz)
    data['keypoint'] = np.swapaxes(data['keypoint'], 0, 1)
    # The shape here is (time frames, joints, xyz)
    return data


def _check_file_unique_exist(file_path):
    import glob
    for file in glob.iglob(file_path):
        try:
            return file
        except:
            raise FileNotFoundError('Check the file path %s' % file_path)
