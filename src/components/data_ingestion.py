
from src.exception import CustomException
from src.logger import logging
import yaml
import glob
from src.components.data_transformation import *
from utils import save_pickle, save_log


class DataLoader:
    """
        A class for loading data from a directory of mat and keypoint files.
    """

    def __init__(self, config_path):
        self.export_dir = None
        self.sample = None
        self.df_file = {}
        self.sensor = []
        self.pressure_map = []
        self.keypoint = []
        self.load_config(config_path)
        self.locate_file_info()
        return

    def __call__(self, save_mode='pickle'):
        """
            Load data from mat and keypoint files and save it as pickle files.
        """
        # self.data = []
        self.full_log = {'samples': {}}

        for sample_info in self.sample:

            file_name = '_'.join([sample_info['date_folder'], sample_info['individual'], sample_info['set']])  #
            self.find_mat_data(sample_info, file_name + '.csv')
            logging.info("find mat data in " + file_name + 'csv')
            self.find_keypoint_data(sample_info, file_name + '.csv')
            logging.info("find keypoint data in " + file_name + 'csv')

            data_aligner = AlignData(self.df_file)
            data = data_aligner(frequency=self.frequency)
            logging.info("Aligned data in sensor and keypoint")

            # transfer keypoint coordinates to 3D heatmap
            processor = keypoint_to_heatmap(heatmap_shape=self.config['heatmap_shape'],
                                            axis_range=self.config['axis_range'])
            data['heatmap'] = processor(data['keypoint'])
            logging.info("Created heatmap")

            self.update_link_limit(data['keypoint'])
            logging.info("Changed link_limit")

            data['log'] = sample_info
            self.full_log['samples'][file_name] = sample_info

            export_path = os.path.join(self.config['export_dir'], 'log', file_name + '.yml')
            save_log(data['log'], export_path)

            if save_mode == "pickle":
                export_path = os.path.join(self.config['export_dir'], 'data', file_name + '.p')

                save_pickle(data, export_path)

            # self.data.append(data.copy())
        self.full_log.update(self.config)

        save_log(self.full_log, os.path.join(self.config['export_dir'], 'process_log.yml'))
        # save_pickle(self.data, os.path.join(self.config['export_dir'], 'data.p'))

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

    def load_config(self, config_path):
        """
        Load configuration from config file:
        - db_dir
        - date_folder_list: include all if it's empty.
        - individual_list: include all if it's empty.
        - joints
        - frequency
        - add_noise: False or ratio float
        - flip: False or axis int
        - export_dir: The folder where configuration file sits
        """

        # Open the YAML file
        with open(config_path, 'r') as file:
            # Load the YAML data
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        for key, value in self.config.items():
            # Assign configuration to self variable
            setattr(self, key, value)

        # Define the raw data is under the same folder
        if self.db_dir == "":
            self.db_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(config_path))), "raw")
            print(self.db_dir)

        self.config['export_dir'] = os.path.dirname(config_path)

        return

    def locate_file_info(self):
        """
            Find the mat recording file, keypoint matlab recording file for each sample
        """
        self.sample = []
        for date_path in glob.glob(os.path.join(self.db_dir, "*")):
            date_folder = os.path.basename(date_path)
            if date_folder in self.date_folder_list or self.date_folder_list is None:
                with open(os.path.join(date_path, 'mat_info.yml'), 'r') as file:
                    mat_info = yaml.load(file, Loader=yaml.FullLoader)
                    self.config.update(mat_info)
                for individual_path in glob.glob(os.path.join(date_path, "*")):
                    if os.path.isfile(individual_path): continue
                    individual_folder = os.path.basename(individual_path)
                    if individual_folder in self.individual_list or self.individual_list is None:

                        with open(os.path.join(individual_path, 'info.yml'), 'r') as file:
                            info = yaml.load(file, Loader=yaml.FullLoader)
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

        processor = ReadMat(sample_info['mat_recording'],
                            loc_ele_map=self.config['location_ele_map'],
                            pressure_ele_map=self.config['pressure_ele_map'],
                            ui_grid=self.config['ui_grid'])
        sensor_data = processor.extract_sensor_data()
        export_path = os.path.join(self.config['export_dir'], 'sensor', file_name)
        processor.save_data(sensor_data, export_path)
        self.df_file['sensor'] = export_path

        pressure_data = processor.extract_pressure_data()

        export_path = os.path.join(self.config['export_dir'], 'pressure_map', file_name)
        processor.save_data(pressure_data, export_path)
        self.df_file['pressure_map'] = export_path
        return

    def find_keypoint_data(self, sample_info, file_name):
        data_processor = ReadKeypoint(sample_info['keypoint_recording'],
                                      joints=self.joints,
                                      marker_set_path=sample_info['marker_set_path'],
                                      merge_point_label=self.config['merge_point'],
                                      axis_range=self.config['axis_range'])
        centre = data_processor.mat_centre()
        theta = data_processor.rotated_angle()
        data = data_processor.skeleton_data()
        data = data_processor.axis_transform(data, centre, theta)
        data = data_processor.normalise(data)
        # data = data_processor.cut_data(data, percentage=0.5)

        export_path = os.path.join(self.config['export_dir'], 'keypoint', file_name)
        data_processor.save_data(data, export_path)

        self.df_file['keypoint'] = export_path
        return

    def augment_data(self):

        # TODO
        return

