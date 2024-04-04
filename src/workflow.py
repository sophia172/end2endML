import argparse
import os

from src.exception import CustomException
from src.logger import logging
import sys
from src.components.data_ingestion import DataLoader
def cli():
    """
    Sort out parameters here
    :return:
    """
    parser = argparse.ArgumentParser(description='Process parameters')
    parser.add_argument('--model_parameter', type=str, default='./config/data_processor_example.yml', help='parameter file path')
    parser.add_argument('--data_config_path', type=str, default=None, help='Process file from raw data')
    parser.add_argument('--data_path', type=str, default="gs://cloud-ai-platform-37c994cb-e090-4da1-ab7d-599262814bd1/vertexAI_PPIT_data.csv", help='Clean data path')

    args = parser.parse_args().__dict__
    process_raw_data: bool = args["data_config_path"] is not None
    data_config_path: str = args.pop("data_config_path")
    if process_raw_data:
        data_path = DataLoader(config_path=data_config_path)()
        logging.info("clean data saved", data_path)

    else:
        data_path: str = args.pop("data_path")
        if data_path.startswith("gs://"):
            try:
                os.makedirs("data", exist_ok=True)
                os.system(f"gsutil cp {data_path} data/")
                logging.info("Downloaded data from online")
            except Exception as e:
                raise CustomException(e, sys)


    try:
        # TODO
        logging.info("Finished training pipeline")
    except Exception as e:
        raise CustomException(e, sys)




if __name__=="__main__":
    cli()