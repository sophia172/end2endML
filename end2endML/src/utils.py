import sys
import os
import yaml
import pickle
from typing import Callable
from end2endML.src.exception import CustomException
from end2endML.src.logger import logging
from dataclasses import dataclass
from sklearn.metrics import r2_score
import pandas as pd
def _root():
    for path in os.walk(os.getcwd()):
        path = os.path.split(path[0])
        if path[-1] == "src":
            src_path = os.path.join(os.getcwd(), *path)
            return os.path.dirname(src_path)

    dir_path = os.getcwd()
    for _ in range(3):
        dir_path = os.path.dirname(dir_path)

        for path in os.walk(dir_path):
            path = os.path.split(path[0])
            if path[-1] == "src":
                src_path = os.path.join(*path)
                return os.path.dirname(src_path)
    logging.info(f"Cannot locate root file.")
    raise CustomException(Exception, sys)


ROOT = _root()


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

def load_config(configuration_path, folder="model"):
    config_filename = os.path.basename(configuration_path).split(".")[0]
    config = AttrDict.from_nested_dicts(reader(configuration_path))
    dir = os.path.join(ROOT, folder, config_filename)
    os.makedirs(dir, exist_ok=True)
    logging.info(f"Creating {folder} folder {config_filename}")
    return (config_filename, config, dir)


def evaluate_models(X_train, X_test, y_train, y_test, models={}, metric=r2_score):
    try:
        report = {}
        trained_model = {}

        for name, model in models.items():
            logging.info(f"Started training and testing model {name}")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = metric(y_train, y_train_pred)
            test_model_score = metric(y_test, y_test_pred)
            report[name] = test_model_score
            trained_model[name] = model
            logging.info(f"Finished training and testing model {name} with train score {train_model_score} \
            and test score {test_model_score}")
        return report, trained_model

    except CustomException as e:
        logging.error(e)
        raise CustomException(e, sys)



if __name__=="__main__":
    print(ROOT)
    pass