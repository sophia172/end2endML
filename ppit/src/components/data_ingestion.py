from ppit.src.exception import CustomException
from ppit.src.logger import logging
from ppit.src.utils import reader
import sys
import pandas as pd
import numpy as np

JOINTS = ["lefthip",
          "leftknee",
          "leftfoot",
          "righthip",
          "rightknee",
          "rightfoot",
          "leftshoulder",
          "leftelbow",
          "leftwrist",
          "rightshoulder",
          "rightelbow",
          "rightwrist",
          "righttoe",
          "lefttoe",
          "hips",
          "neck"]
XYZ = ['x', 'y', 'z']


class DataLoader():
    """
    Designed for PPIT project
    """

    def __init__(self, data_path, ):
        self.data: pd.DataFrame = reader(data_path)
        logging.info(f"Data Preview \n {self.data.head()}" )
    def __call__(self):
        try:
            inputs, outputs = self.load_train_file(self.data)
            logging.info(f"generate X and y with data shape of {inputs.shape}, {outputs.shape}")
            return inputs, outputs
        except Exception as e:
            raise CustomException(e, sys)

    def reformat_pressure_map(self, df):
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

    def reformat_sensor(self, df):
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

    def reformat_keypoint(self, df):
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

    def load_train_file(self,
                        file: str,
                        input_shape=(10, 13, 24),
                        output_shape=(16, 3)):
        """
        Open the tabular data file and process data to the input and output format

        Parameters
        ----------
        file: str
            The csv data file saved in Google Cloud Storage.

        input_shape: Tuple
            The shape of input for training model

        output_shape: Tuple
            The shape of output for training model

        Returns
        -------
        input: A NumPy array containing the full input data

        output: A NumPy array containing the full output data
        """

        keypoint_col_names = [col for col in self.data.columns if 'JointAngle' in col]
        pressure_col_names = [col for col in self.data.columns if 'pressure' in col]
        location_col_names = [col for col in self.data.columns if 'location' in col]
        keypoint_data = self.keypoint_df_to_array(self.data[keypoint_col_names])
        logging.info(f"Transformed keypoint data to array")
        pressure_data = self.sensor_df_to_array(self.data[pressure_col_names])
        logging.info(f"Transformed pressure data to array")
        location_data = self.sensor_df_to_array(self.data[location_col_names])
        logging.info(f"Transformed sensor data to array")
        num_sample = len(self.data)
        del self.data

        # reshape sensor data
        sensor_data = []
        for i in range(num_sample):
            pressure = pressure_data[i]
            location = location_data[i]
            sensor_data.append(self.restack_electrode_data(pressure, location))

        logging.info(f"Combined pressure sensor data and location sensor data")
        sensor_data = np.array(sensor_data)
        del pressure_data, location_data

        # layout input data
        input_data = []
        for i in range(num_sample - input_shape[0] + 1):
            input_data.append(sensor_data[i: i + input_shape[0]])
        input_data = np.array(input_data)

        # layout output data
        output_data = keypoint_data[input_shape[0] - 1:]

        return input_data, output_data

    def restack_electrode_data(self, pressure, location):
        frame_data = np.zeros((13, 24))
        for i in range(frame_data.shape[0]):
            for j in range(frame_data.shape[1]):
                # i.   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
                # loc  0,  , 1,  , 2,  , 3,  , 4,  ,  5,   ,  6
                # pres. , 0,  , 1,  , 2,  , 3,  , 4,   ,  5,

                # j.   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ..., 22, 23
                # loc  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ..., 22, 23
                # pres.0, 0, 1, 1, 2, 2, 3, 3, 4, 4,  5,  5,  6,      11, 11
                source_i = i // 2
                source_j = j
                source_data = location
                if i % 2 != 0:  # pressure
                    source_data = pressure
                    source_j = j // 2
                frame_data[i, j] = source_data[source_i, source_j]
        return frame_data

    def sensor_df_to_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts the dataframe to numpy array

        For pressure and location, convert by row and col
        Parameters
        ----------
        df: pd.DataFrame
            The dataframe to convert

        Returns
        -------
        A NumPy array
        """

        num_sample = len(df)
        num_col = max([int(col_name.split('_')[-1]) for col_name in df.columns])
        num_row = max([int(col_name.split('_')[-2]) for col_name in df.columns])
        input_shape = (num_sample, num_row + 1, num_col + 1)

        init_array = np.zeros(input_shape)

        def fill_num_to_array(col: pd.Series):
            _, row_name, col_name = col.name.split('_')
            init_array[:, int(row_name), int(col_name)] = col.values
            return

        df.apply(lambda col: fill_num_to_array(col), axis=0)

        return init_array

    def keypoint_df_to_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts the dataframe to numpy array

        For keypoint, convert by x, y, z coordinates
        Parameters
        ----------
        df: pd.DataFrame
            The dataframe to convert

        Returns
        -------
        A NumPy array
        """

        num_sample = len(df)
        num_col = len(XYZ)
        num_row = len(JOINTS)
        output_shape = (num_sample, num_row, num_col)

        init_array = np.zeros(output_shape)

        def fill_num_to_array(col: pd.Series):
            _, joint_name, xyz = col.name.split('_')
            if joint_name not in JOINTS:
                return
            init_array[:, JOINTS.index(joint_name), XYZ.index(xyz)] = col.values
            return

        df.apply(lambda col: fill_num_to_array(col), axis=0)

        return init_array


if __name__ == "__main__":
    pass
