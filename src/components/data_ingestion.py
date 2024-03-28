import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("path")
    val_data_path: str = os.path.join("new path")
    raw_data_path: str = os.path.join("another path")


class DataIngestion:
    def __init__(self):
        super().__init__()
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            logging.info("Download dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  #

            pass
        except Exception as e:
            logging.info("")
            raise CustomException(e, sys)


class ReadMat():
    def __init__(self, file_path, loc_ele_map=None, pressure_ele_map=None, ui_grid=None):
        """
                Initializes an instance of the ReadMat class.

                Args:
                    file_path (str): The path to the file containing the data to be read.
                    loc_ele_map (array-like): An array-like object containing the mapping of location electrodes.
                    pressure_ele_map (array-like): An array-like object containing the mapping of pressure electrodes.
                    ui_grid (tuple): A tuple containing the dimensions of the grid used to read the data.

                Attributes:
                    file_path (str): The path to the file containing the data to be read.
                    file_name (str): The name of the file containing the data to be read.
                    df (pandas.DataFrame): A DataFrame containing the data read from the file.
                    ui_grid (tuple): A tuple containing the dimensions of the grid used to read the data.
                    time_stamp (numpy.ndarray): A numpy array containing the time stamp of the data.
                    loc_ele_map (numpy.ndarray): An array containing the mapping of location electrodes.
                    pressure_ele_map (numpy.ndarray): An array containing the mapping of pressure electrodes.
                    sensor_num (int): The total number of sensors used.

        """
        self.file_path = file_path
        self.file_name = os.path.basename(os.path.splitext(self.file_path)[0]).split(' ')[-1]
        try:
            self.df = pd.read_csv(self.file_path, skiprows=7, index_col=0)
            logging.info("read csv file")
        except Exception as e:
            logging.info("pd.read_csv file error")
            raise CustomException(e, sys)
        self.ui_grid = ui_grid  # (row in one module, column number in one module, number of module), MCU at left
        self.df = self.df.iloc[range(0, len(self.df), self.ui_grid[
            -1])]  # The df only updates one module in each row, only choose data every 6 rows
        self.df.index = pd.to_datetime(self.df.index,
                                       format='%m_%d_%Y_%H_%M_%S_%f')  # change time columnn to the correct format
        self.time_stamp = self.df.index.to_numpy()
        self.loc_ele_map = np.array(loc_ele_map)
        self.pressure_ele_map = np.array(pressure_ele_map)
        self.sensor_num = int(np.prod(self.loc_ele_map.shape) + np.prod(self.pressure_ele_map.shape))

    def extract_pressure_data(self):
        """
        Extracts pressure data from the data read from the file.

        Returns:
            numpy.ndarray: A numpy array containing the pressure data.

        """
        pressure_map = []
        num_module = self.ui_grid[-1]
        num_row = self.ui_grid[0]
        num_col_per_module = self.ui_grid[1]
        grid_in_module = self.ui_grid[0] * self.ui_grid[1]
        for i in range(num_module):
            data = self.df.iloc[:, i * grid_in_module: i * grid_in_module + grid_in_module]
            data.iloc[:, 0] = data.iloc[:, 0].str.replace('[', '', regex=True)
            data.iloc[:, -1] = data.iloc[:, -1].str.replace(']', '', regex=True)
            data = data.astype(float).to_numpy()  # The data here has shape of (time_frame, 78)
            pressure_map.append(data.reshape(len(data), num_row, num_col_per_module))

        pressure_map = np.dstack(pressure_map)

        return pressure_map

    def _signal_range(self):
        bounds = {}
        f = open(self.file_path, 'r')
        for i in range(self.ui_grid[-1]):
            line = f.readline()
            bounds[i] = eval(line.split(':', 1)[1])
        f.close()
        for key in bounds:
            bounds[key]['range'] = np.subtract(bounds[key]['maxs'], bounds[key]['mins'])
        return bounds

    def extract_sensor_data(self, normalise=True):
        """
        Two data output:
            1. location electrode data array of (time_frame, row, column) for whole mat, MCU on the left side.
            2. pressure electrode data array of (time_frame, row, column) for whole mat, MCU on the left side.
        """

        # create empty arrays with the shape of (num_module, time_frame, row, column)
        loc_ele_data = np.zeros((self.ui_grid[-1], len(self.df), *self.loc_ele_map.shape))
        pressure_ele_data = np.zeros((self.ui_grid[-1], len(self.df), *self.pressure_ele_map.shape))

        grid_num = np.prod(self.ui_grid)
        if normalise:
            bounds = self._signal_range()
        for i in range(self.ui_grid[-1]):
            data = self.df.iloc[:,
                   grid_num + i * self.sensor_num: grid_num + (i + 1) * self.sensor_num]
            data.iloc[:, 0] = data.iloc[:, 0].str.replace('[', '', regex=True)
            data.iloc[:, -1] = data.iloc[:, -1].str.replace(']', '', regex=True)
            data = data.astype(int)
            data = data.to_numpy()  # in the sequence of electrode number from 0 to 40
            if normalise:
                data = np.subtract(data, bounds[i]['mins']) / bounds[i]['range']

            for iy, ix in np.ndindex(loc_ele_data.shape[-2:]):
                loc_ele_data[i, :, iy, ix] = data[:, self.loc_ele_map[iy, ix]]

            for iy, ix in np.ndindex(pressure_ele_data.shape[-2:]):
                pressure_ele_data[i, :, iy, ix] = data[:, self.pressure_ele_map[iy, ix]]

        loc_ele_data = np.dstack(loc_ele_data)
        pressure_ele_data = np.dstack(pressure_ele_data)

        return loc_ele_data, pressure_ele_data

    def add_noise(self, data, ratio=0.01):
        """
        Accept numpy array, add Gaussian noise
        """
        data_range = np.max(data) - np.min(data)
        noise = np.random.normal(0, data_range * ratio, data.shape)
        return data + noise

    def _array2df(self, data):
        df = pd.DataFrame(data.reshape((len(data), -1)))
        col_name = []
        for iy, ix in np.ndindex(data.shape[-2:]):
            col_name.append('row%d_col%d' % (iy, ix))
        df.columns = col_name
        df.index = self.time_stamp
        return df

    def save_data(self, data, export_path):
        """
        Distinguish the pressure map data and electrode data by the type of data.
        If it is array, it is pressure map data.
        If it is tuple, it is electrode data.
        """
        if type(data) == tuple:
            locaction_df = self._array2df(data[0]).add_suffix('_location')
            pressure_df = self._array2df(data[1]).add_suffix('_pressure')
            df = pd.concat([locaction_df, pressure_df], axis=1)
        else:
            df = self._array2df(data).add_suffix('_pressure_map')

        if os.path.exists(os.path.dirname(export_path)):
            None
        else:
            os.mkdir(os.path.dirname(export_path))

        df.to_csv(export_path)
