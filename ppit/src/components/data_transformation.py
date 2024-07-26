import sys
from ppit.src.exception import CustomException
from ppit.src.logger import logging
from ppit.src.utils import _check_file_unique_exist, reader, load_config
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from dataclasses import dataclass
from ppit.src.utils import writer
import glob


@dataclass
# class DataTransformationConfig:
#     # TODO
#     preprocessor_obj_file_path: str = os.path.join()

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()
#
#     def get_data_transformer_object(self):
#         try:
#             num_pipeline = Pipeline(
#                 steps=[
#                         ("imputer", SimpleImputer(strategy="medium")),
#                         ("scaler", StandardScaler()),
#                 ],
#             )
#             cat_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="most_frequent")),
#                     ("one_hot_encoder", OneHotEncoder()),
#                     ("scaler", StandardScaler()),
#                 ],
#             )
#             logging.info("Numerical encoding completed")
#
#             preprocessor=ColumnTransformer(
#                 [
#                     ("num_pipeline", num_pipeline, numerical_columns),
#                     ("cat_pipeline", cat_pipeline, categorical_columns)
#                 ]
#             )
#             return preprocessor
#         except Exception as e:
#             raise CustomException(e, sys)
#
#     def initiate_data_transformation(self,train_path, test_path):
#         try:
#             # get data
#             preprocessing_obj = self.get_data_transformer_object()
#             target_column = []
#             numerical_columns = ["",""]
#
#             input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
#             # TODO
#             # write function to extract train and test
#
#             save_pickle(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj,
#             )
#             return (train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path)
#
#         except Exception as e:
#             raise CustomException(e, sys)

class DataProcessor:
    """
        A class for loading data from a directory of mat and keypoint files.
    """

    def __init__(self, config_path=None, **kwargs):

        self.__dict__ = kwargs
        self.export_dir = None
        self.sample = None
        self.df_file = {}
        self.sensor = []
        self.pressure_map = []
        self.keypoint = []
        self.config_filename, self.config, self.db_dir = load_config(config_path, folder="raw_data")
        self.locate_file_info()
        return

    def __call__(self, save_mode='pickle'):
        """
            Load data from mat and keypoint files and save it as pickle files.
        """

        for sample_info in self.sample:
            logging.info("start processing file ", sample_info)
            file_name = '_'.join([sample_info['date_folder'], sample_info['individual'], sample_info['set']])

            self.find_mat_data(sample_info, file_name + '.csv')
            logging.info("found mat data in " + file_name + 'csv')

            self.find_keypoint_data(sample_info, file_name + '.csv')
            logging.info("found keypoint data in " + file_name + 'csv')

            data_aligner = AlignData(self.df_file)
            data = data_aligner(frequency=self.frequency)
            logging.info("Aligned data in sensor and keypoint")

            # transfer keypoint coordinates to 3D heatmap
            # processor = keypoint_to_heatmap(heatmap_shape=self.config['heatmap_shape'],
            #                                 axis_range=self.config['axis_range'])
            # data['heatmap'] = processor(data['keypoint'])
            # logging.info("Created heatmap")

            self.update_link_limit(data['keypoint'])
            logging.info("Changed link_limit")
            writer(data, os.path.join('data', file_name + '.p'))
        return os.path.join('data', '*.p')

    def update_link_limit(self, data):
        """
        Load keypoint data and calculate the link limit based on linked joints from self.config['link_limit']
        """
        for idx, link_limit in enumerate(self.config['link_limit']):
            link, limit = link_limit
            joint1 = data[:, link[0] - 1, :]
            joint2 = data[:, link[1] - 1, :]
            length = np.linalg.norm(np.nan_to_num(joint2 - joint1, nan=np.inf), axis=1)
            length = np.where(length > 1e6, np.nan, length)
            max_length = np.nanmax(length).item()
            min_length = np.nanmin(length).item()
            if limit is None:
                self.config['link_limit'][idx][-1] = [min_length, max_length]

            else:
                self.config['link_limit'][idx][-1] = [min([min_length, self.config['link_limit'][idx][-1][0]]),
                                                      max([max_length, self.config['link_limit'][idx][-1][1]])]

        return

    def locate_file_info(self):
        """
            Find the mat recording file, keypoint matlab recording file for each sample
        """
        self.sample = []
        for date_path in glob.glob(os.path.join(self.db_dir, "*")):
            date_folder = os.path.basename(date_path)
            if date_folder in self.date_folder_list or self.date_folder_list is None:
                mat_info = reader(os.path.join(date_path, 'mat_info.yml'))
                self.config.update(mat_info)
                for individual_path in glob.glob(os.path.join(date_path, "*")):
                    if os.path.isfile(individual_path): continue
                    individual_folder = os.path.basename(individual_path)
                    if individual_folder in self.individual_list or self.individual_list is None:

                        info = reader(os.path.join(individual_path, 'info.yml'))
                        info['date_folder'] = date_folder
                        info['individual'] = individual_folder
                        info['marker_set_path'] = os.path.join(individual_path, 'keypoint_recording',
                                                               'marker set list.csv')

                        mat_recording_files = glob.glob(os.path.join(individual_path, 'mat_recording', '*.csv'))
                        keypoint_recording_files = glob.glob(
                            os.path.join(individual_path, 'keypoint_recording', '*.mat'))
                        mat_recording = [os.path.basename(path).split('.')[0].replace(" ", "_").lower() for path in
                                         mat_recording_files]
                        keypoint_recording = [os.path.basename(path).split('.')[0].replace(" ", "_").lower() for path in
                                              keypoint_recording_files]

                        set_name = set(mat_recording).intersection(set(keypoint_recording))
                        for one_set in list(set_name):
                            info['set'] = one_set
                            info['mat_recording'] = os.path.join(individual_path, 'mat_recording', one_set + '.csv')
                            info['keypoint_recording'] = os.path.join(individual_path, 'keypoint_recording',
                                                                      one_set + '.mat')
                            self.sample.append(info.copy())
        return

    def find_mat_data(self, sample_info, file_name):
        """
        In each sample, load the mat corner coordinates.
        """

        mat_processor = ReadMat(sample_info['mat_recording'],
                                loc_ele_map=self.config['location_ele_map'],
                                pressure_ele_map=self.config['pressure_ele_map'],
                                ui_grid=self.config['ui_grid'])
        sensor_data = mat_processor.extract_sensor_data()
        export_path = os.path.join('data', 'sensor', file_name)
        mat_processor.save_data(sensor_data, export_path)
        self.df_file['sensor'] = export_path

        pressure_data = mat_processor.extract_pressure_data()

        export_path = os.path.join('data', 'pressure_map', file_name)
        mat_processor.save_data(pressure_data, export_path)
        self.df_file['pressure_map'] = export_path
        return

    def find_keypoint_data(self, sample_info, file_name):
        mocap_processor = ReadMoCap(sample_info['keypoint_recording'],
                                    joints=self.joints,
                                    marker_set_path=sample_info['marker_set_path'],
                                    merge_point_label=self.config['merge_point'],
                                    axis_range=self.config['axis_range'])
        centre = mocap_processor.mat_centre()
        theta = mocap_processor.rotated_angle()
        data = mocap_processor.skeleton_data()
        data = mocap_processor.axis_transform(data, centre, theta)
        data = mocap_processor.normalise(data)
        # data = data_processor.cut_data(data, percentage=0.5)

        export_path = os.path.join('data', 'keypoint', file_name)
        mocap_processor.save_data(data, export_path)

        self.df_file['keypoint'] = export_path
        return

    def augment_data(self):

        # TODO
        return


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
                    ui_grid (tuple): A tuple containing the dimensions of the grid used to read the data.
                    loc_ele_map (numpy.ndarray): An array containing the mapping of location electrodes.
                    pressure_ele_map (numpy.ndarray): An array containing the mapping of pressure electrodes.

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
            data.iloc[:, 0] = data.iloc[:, 0].str.replace('[', '')
            data.iloc[:, -1] = data.iloc[:, -1].str.replace(']', '')
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
        for i in range(self.ui_grid[-1]):
            data = self.df.iloc[:,
                   grid_num + i * self.sensor_num: grid_num + (i + 1) * self.sensor_num]
            data.iloc[:, 0] = data.iloc[:, 0].astype(str).str.replace('[', '')
            data.iloc[:, -1] = data.iloc[:, -1].astype(str).str.replace(']', '')
            data = data.astype(int)
            data = data.to_numpy()  # in the sequence of electrode number from 0 to 40
            if normalise:
                bounds = self._signal_range()
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
        assert type(data) == np.ndarray, "Please check input data format as numpy array"
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
            logging.info("Save sensor data in both location and pressure")
            locaction_df = self._array2df(data[0]).add_suffix('_location')
            pressure_df = self._array2df(data[1]).add_suffix('_pressure')
            df = pd.concat([locaction_df, pressure_df], axis=1)
        else:
            logging.info("Save pressure map UI grid data")
            df = self._array2df(data).add_suffix('_pressure_map')

        if os.path.exists(os.path.dirname(export_path)):
            None
        else:
            os.mkdir(os.path.dirname(export_path))

        df.to_csv(export_path)


class ReadMoCap():
    def __init__(self, file_path, joints=None, marker_set_path=None, merge_point_label=None, axis_range=32767):
        self.file_path = file_path
        self.axis_range = axis_range
        self.dir_name = os.path.dirname(self.file_path)
        mat_file = loadmat(file_path)
        self.temp_file = mat_file
        self.file_name = list(mat_file.keys())[-1]
        self.mat_file = mat_file[self.file_name][0, 0]
        #     self.frequency = self.mat_file[4][0,0]
        #     self.record_time = pd.to_datetime(self.mat_file[1][0].split('\t')[0])
        self._marker_set(marker_set_path)
        self.run_once = {}

        self.skeleton_label = joints
        self.merge_point_label = merge_point_label

        return

    def _marker_set(self, marker_set_path):
        try:
            df = pd.read_csv(marker_set_path, header=0)
            df[['location1', 'location2']] = df['location'].str.split('/', expand=True)
            df[['right', 'keyword']] = df['location2'].str.split(' ', n=1, expand=True)
            df['location1'] = df['location1'].fillna('') + ' ' + df['keyword'].fillna('')
            location = df['location1'].tolist() + df['location2'].tolist()
            marker = df['marker1'].tolist() + df['marker2'].tolist()
            self.marker_set = {location[i]: marker[i] for i in range(len(marker))}
        except:
            raise Exception('Marker set list file not found')

    def record_time(self):
        # Add time delta here to adjust the offset between Kingston time and TG0 time
        return pd.to_datetime(self.mat_file['Timestamp'][0].split('\t')[0]) + pd.Timedelta('0.7s')

    def frames(self):
        return self.mat_file['Frames'][0, 0]

    def time_stamp(self):
        if 'time_stamp' not in self.run_once:
            time_index = pd.date_range(self.record_time(),
                                       periods=self.frames(),
                                       freq=pd.Timedelta(seconds=1 / self.frame_rate()))
            self.run_once['time_stamp'] = time_index
        return self.run_once['time_stamp']

    def frame_rate(self):
        return self.mat_file['FrameRate'][0, 0]

    def labels(self):
        if 'labels' not in self.run_once:
            label_cell = self.mat_file['Trajectories'][0, 0]['Labeled'][0, 0]['Labels']
            self.run_once['labels'] = [label_cell[0][i][0] for i in range(self.num_label())]
        return self.run_once['labels']

    def num_label(self):
        return int(self.mat_file['Trajectories'][0, 0]['Labeled'][0, 0]['Count'][0, 0])

    def data(self):
        """
        In the shape of (num_label, coordinates_xyz, frames)
        """
        if 'data' not in self.run_once:
            self.run_once['data'] = self.mat_file['Trajectories'][0, 0]['Labeled'][0, 0]['Data'][:, :3, :]
        return self.run_once['data']

    def estimated_error(self):
        return self.mat_file['Trajectories'][0, 0]['Labeled'][0, 0]['Data'][:, -1, :]

    def trajectory_type(self):
        # Camera Observation
        return self.mat_file['Trajectories'][0, 0]['Labeled'][0, 0]['TrajectoryType'][0, 0]

    def skeleton_data(self):
        """
        In the shape of (num_joints, coordinates_xyz, frames)
        """
        label = self.labels()
        full_data = self.data()
        data = []
        for joint in self.skeleton_label:
            joint_data = []
            for marker in self.merge_point_label[joint]:
                landmark = self.marker_set[marker]
                try:
                    idx = label.index(landmark)  # In some data, the coordinates points does not exist
                    joint_data.append(full_data[idx])
                except:
                    logging.info('marker %s cannot be found' % landmark)
            logging.info("joint", joint, ". \n The number of nan value", np.sum(np.isnan(joint_data)),
                  ". \n The data shape for this joint",
                  np.array(joint_data).shape,
                  ". \n The data shape after mean calculation for this joint",
                  np.nanmean(joint_data, axis=0).shape,
                  ". \n The number of nan value after mean treatment",
                  np.sum(np.isnan(np.nanmean(joint_data, axis=0))))
            data.append(np.nanmean(joint_data, axis=0))
        logging.info("full data", np.array(data).shape, np.sum(np.isnan(np.array(data))))
        return np.array(data)

    def fillna(self, data):
        """
        In the shape of (num_position, coordinates_xyz, frames)
        """
        return

    def normalise(self, data):
        """
        In the shape of (num_position, coordinates_xyz, frames)
        Normalise the data within byte_range
        """

        ratio = self.axis_range / 1500  # This 1500 is the approximate ratio between Kingston axis to byte range 32767
        if ratio >= 1:
            ratio = int(ratio)  # If axis_range is smaller than 1500, All data will be float
        data = np.swapaxes(data, 1, 2) * ratio  # expand the axis so they would utilise byte range as much as possible

        # remove infinite number or NaN
        data = self._rm_nan(data)

        data = data.astype(type(ratio))

        # change infinite number to NaN
        data = self._add_nan(data)

        normalised_data = np.swapaxes(data, 1, 2)
        return normalised_data

    def cut_data(self, data, percentage=0.3):
        """
        This function is added to choose part of data due to MCU memeory limitation
        Data has the shape of (num_position, coordinates_xyz, frames)
        """
        num_frame = data.shape[-1]
        return data[:, :, :-int(num_frame * percentage)]

    def _add_nan(self, data):
        """
        accept data in arrays
        Change infinate data to None
        """
        data = np.where(((data > 10 ** 6) | (data < -10 ** 6)), None, data)
        return data

    def _rm_nan(self, data):
        """
        accept data in arrays

        """
        data = np.where(np.isnan(data), np.inf, data)
        # data = np.where(((data > 10 ** 6) | (data < -10 ** 6)), 32767, data)
        return data

    def _reduce_frequency(self, df, frequency=100):
        """
        Accept data frame only.
        Each row is one time frame.
        """
        gap = 200 // frequency
        df = df.iloc[range(0, len(df), gap)]
        return df

    def save_data(self, data, export_path, frequency=100):
        """
        In the shape of (num_position, coordinates_xyz, frames)
        """
        header = []
        for marker in self.skeleton_label:
            for pos in ['x', 'y', 'z']:
                header.append(' '.join((str(self.skeleton_label[marker]), marker, pos)))
        reshaped_data = np.hstack([joint.T for joint in data])
        df = pd.DataFrame(reshaped_data, columns=header)
        df.index = self.time_stamp()  # Add time stamp calcuated from matlab file to the DataFrame

        # reduce frequency
        # df = self._reduce_frequency(df, frequency=frequency)

        if os.path.exists(os.path.dirname(export_path)):
            None
        else:
            os.mkdir(os.path.dirname(export_path))

        format = export_path.split('.')[-1]

        if format == 'csv':
            df.to_csv(export_path)
        else:
            np.asarray(df.values).tofile(export_path)
        return

    def _mat_data_exist(self):
        if '_mat_data_exist' not in self.run_once:
            self.mat_label = ['MattBackR',
                              'MattFrontR',
                              'MattBackL',
                              'MattFrontL']
            if sum([label in self.labels() for label in self.mat_label]) == 4:
                self.run_once['_mat_data_exist'] = True
            else:
                self.run_once['_mat_data_exist'] = False
        return self.run_once['_mat_data_exist']

    def _mat_data(self):

        """
        In the shape of (4_corners, coordinates_xyz, frames)
        """
        if self._mat_data_exist():
            label = self.labels()
            full_data = self.data()
            data = []
            for mat_corner in self.mat_label:
                idx = label.index(mat_corner)
                data.append(full_data[idx])
            return np.array(data)

    def mat_centre(self):
        """
        In the shape of (coordinates_xyz)
        """
        if 'mat_centre' not in self.run_once:
            if self._mat_data_exist():
                data = np.average(self._mat_data(), axis=-1)
                self.run_once['mat_centre'] = np.average(data, axis=0)
                return self.run_once['mat_centre']
            else:
                self.run_once['mat_centre'] = None
        return self.run_once['mat_centre']

    def rotated_angle(self):
        # Find the mat x axis first, calculate angle afterwards
        if 'rotation_angle' not in self.run_once:
            if self._mat_data_exist():
                label = self.labels()
                full_data = self.data()
                data = []
                for left_corner in ['MattBackR', 'MattFrontR']:
                    idx = label.index(left_corner)
                    data.append(full_data[idx])
                data = np.average(data, axis=-1)
                data = np.average(data, axis=0)
                x_axis = data - self.mat_centre()
                dot_product = np.dot(np.array([1, 0, 0]), x_axis)
                self.run_once['rotation_angle'] = np.arccos(dot_product / np.linalg.norm(x_axis))
            else:
                self.run_once['rotation_angle'] == None
        return self.run_once['rotation_angle']

    def rotate_z(self, coordinates, theta=0):
        """
        In the shape of (num_points, coordinates_xyz, frames)
        """
        R = np.identity(3)
        R[0, 0] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)
        R[1, 1] = np.cos(theta)

        data = np.empty(coordinates.shape)
        for point in range(coordinates.shape[0]):
            for frame in range(coordinates.shape[-1]):
                data[point, :, frame] = np.dot(R, coordinates[point, :, frame])
        return np.array(data)

    def axis_transform(self, coordinates, centre=0, theta=0, opt=0):
        """
        The coordinates and transformed_coords are both un the shape of (num_joints, coordinates_xyz, frames)
        """
        if theta is None or centre is None:
            theta, centre = (0, 0)
        # Giving a few options to rotate for another 90 degree
        theta += opt * np.pi / 2
        coordinates = np.swapaxes(coordinates, 1, 2)
        recentred_coords = coordinates - centre
        recentred_coords = np.swapaxes(recentred_coords, 1, 2)

        transformed_coords = self.rotate_z(recentred_coords, theta)
        return transformed_coords

    def _signal_range(self, data):

        """
        For analysis only, it is used to define the appropriate value in normalise
        """
        upper_bound = np.nanmax(np.nanmax(data, axis=-1), axis=0)
        lower_bound = np.nanmin(np.nanmin(data, axis=-1), axis=0)
        print((lower_bound, upper_bound))

class ReadTracker():
    def __init__(self):
        return


def reformat_file(df):
    if 'row' in df.columns[-2]:  # I chose a random column name here to check which file it is.
        print(' This file contains sensor data')
        return reformat_sensor(df)
    else:
        print(' This file contains keypoint data')
        return reformat_keypoint(df)


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


class read_csv():

    def __init__(self):
        return

    def __call__(self, move_set, folder=''):
        try:
            csv_file = _check_file_unique_exist(os.path.join(folder, '*[sS]%s*.csv' % move_set[1:]))
            self.df = pd.read_csv(csv_file, index_col=0)
            logging.info("read file: " + csv_file)
        except Exception as e:
            logging.info("move_set list: " + ",".join(move_set))
            raise CustomException(e, sys)
        return reformat_file(self.df)


class AlignData:
    def __init__(self, file_dict):
        """
        File should be a dictionary with key as data type: chooose among sensor, pressure_map and keypoint
        """
        # Check if there is corresponding file

        self.df = {}
        self.df_new = {}
        self.data = {}

        for key in file_dict:
            file = _check_file_unique_exist(file_dict[key])
            self.df[key] = pd.read_csv(file, index_col=0)
            self.df[key].index = pd.to_datetime(self.df[key].index)

    def __call__(self, frequency=100):
        time_delta = str(int(1000 / frequency)) + 'ms'

        try:
            for key in self.df:
                self.df_new[key] = self.df[key].resample(time_delta).median().interpolate(limit_direction='forward')
            intersection = self.df_new['sensor'].index.intersection(self.df_new['keypoint'].index)

            logging.info("interpret data with fixed time frame")
        except Exception as e:
            logging.info("df_new: ")
            raise CustomException(e, sys)

        try:
            for key in self.df_new:
                self.df_new[key] = self.df_new[key].loc[intersection]
                self.data.update(eval('reformat_%s(self.df_new[key])' % key))
            logging.info("reformat data")
        except Exception as e:
            logging.info("intersection: " + ",".join(intersection))
            raise CustomException(e, sys)
        return self.data


class data_augmentation():
    def __init__(self):
        # TODO
        return

    def flip_direction(self, data, axis=1):
        """
        axis 0,1,2 corresponds to x, y, z.
        x axis is along the longer direction of mat, y axis is along the shorter direction of mat.
        """

        # TODO
        return

    def add_noise(self, data, ratio=0.1):
        """
        The noise ratio is based on the maximum signal of the pressure map, sensor or keypoint data.
        """

        # TODO
        return


class keypoint_to_heatmap():
    """
    It creates a heatmap (frame, joints, 3Dheatmap_resolution) from keypoint (frame, joints, xyz)
    The 3D heatmap_resolution is (20, 20, 18) in the MIT model.
    xyz_range is the maximum value in the axis range (positive number only). It goes from -32767 to +32767 by default
    """

    def __init__(self, heatmap_shape=[20, 20, 18], axis_range=32767):
        # heatmap_shape should be numpy array
        self.heatmap_shape = np.array(heatmap_shape)
        y, x, z = [np.linspace(0., 1., int(heatmap_shape[i])) for i in range(3)]
        self.pos_xyz = np.meshgrid(x, y, z)
        self.axis_range = axis_range

    def __call__(self, keypoints):
        """
        keypoints is numpy array, it has a shape of (frame, joints, xyz)
        It has to be within axis range. No cap is applied here
        """
        frame_num, joint_num = keypoints.shape[:2]

        heatmap = np.zeros((frame_num, joint_num, *self.heatmap_shape))

        keypoints = self.normalise(keypoints, self.heatmap_shape)

        for i in range(frame_num):
            frame = keypoints[i, :, :]  # each frame has a shape of (joint_num, xyz)

            for k in range(joint_num):
                dis = np.sqrt(np.sum([(self.pos_xyz[j] - frame[k, j]) ** 2 for j in range(3)], axis=0))
                g = self.gaussian(dis, 0.0001, 0.2)
                # heatmap[i,k,:,:,:] = np.round(self.softmax(g),2)
                heatmap[i, k, :, :, :] = self.softmax(
                    g) / 0.25  # 1:0.25; 0.5:0.8 This parameter is used in MIT model. Not sure what it is doing

        return heatmap

    def gaussian(self, dis, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(dis - mu) ** 2 / (2 * sigma ** 2))

    def softmax(self, data):
        return np.exp(data) / np.sum(np.exp(data))

    def normalise(self, data, heatmap_shape):
        # data has shape of (frame, joint, xyz)
        resolution = [
            self.axis_range * 2 / heatmap_shape[i]
            for i in range(3)
        ]
        data = np.divide((data + self.axis_range), resolution)
        data = np.divide(data.astype(int), heatmap_shape - 1)
        return data


class heatmap_to_keypoint():
    """
    It creates  a keypoint (frame, joints, xyz) from a heatmap (frame, joints, 3Dheatmap_resolution)
    Output keypoint should be in the range of (-32767,32767) as the input in keypoint_to_heatmap
    The 3Dheatmap_resolution is (20, 20, 18) in the MIT model.
    """

    def __init__(self, axis_range=32767):
        self.axis_range = axis_range

    def __call__(self, heatmap, heatmap_shape=[20, 20, 18]):
        """
        heatmap is numpy array, it has a shape of (frame, joints, 3Dheatmap_resolution_in_xyz_axis)
        return data of size (frame, joints, xyz), data range is [0,1] before reverse_normalise,
        [-32767,32767) after reverse_normalise
        """
        self.pos_xyz = np.meshgrid(
            *[np.linspace(0., 1., int(heatmap_shape[i]))
              for i in range(3)]
        )
        heatmap = np.reshape(heatmap, (*heatmap.shape[:-3], *heatmap_shape))

        eps = 1e-6
        expected_xyz = [
            np.sum(self.pos_xyz[i].reshape(1, -1) * heatmap, axis=-1)
            / (np.sum(heatmap, axis=-1) + eps)
            for i in range(3)
        ]

        expected_xyz = np.swapaxes(expected_xyz, 0, 1)
        expected_xyz = np.swapaxes(expected_xyz, 1, 2)

        xyz_float = self.reverse_normalise(expected_xyz)
        return xyz_float.astype(int)

    def reverse_normalise(self, data):
        """
        data has the shape of (frame, joints, xyz)
        """
        return data * self.axis_range * 2 - self.axis_range
