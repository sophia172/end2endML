import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import ROOT, writer
import os
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join()


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def __call__(self, X_train, y_train, X_test, y_test):
        try:
            logging.info(f"Import data")
            # TODO
        except CustomException as e:
            logging.error(e)
            raise CustomException(e, sys)
