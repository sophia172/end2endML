import argparse
from src.exception import CustomException
from src.logger import logging
import sys
from src.components.data_ingestion import DataLoader

def cli():
    parser = argparse.ArgumentParser(description='Process parameters')
    parser.add_argument('--model_parameter', type=str, default='./config/example.yml', help='parameter file path')
    parser.add_argument('--data_processor_config', type=str, default=None, help='Process file from raw data')
    parser.add_argument('--data_path', type=str, default=None, help='Clean data path')

    args = parser.parse_args().__dict__

    try:
        # TODO
        result = transcribe()
        logging.info("Finished training pipeline")
    except Exception as e:
        raise CustomException(e, sys)

def transcribe():
    try:
        # TODO
        data = DataLoader()
        logging.info(f"Finished loading clean data {data.shape}")
        result = ModelTrainer(data)
        logging.info(f"Finished training model")

    except Exception as e:
        raise CustomException(e, sys)


if __name__=="__main__":
    cli()