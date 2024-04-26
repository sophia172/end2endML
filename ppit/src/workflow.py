import argparse
import os

from ppit.src.exception import CustomException
from ppit.src.logger import logging
import sys
from ppit.src.components.data_transformation import DataProcessor
from ppit.src.components.data_ingestion import DataLoader
from ppit.src.components.model_trainer import BaselineSearch
from ppit.src.components.models import CNN
from ppit.src.utils import none_or_str


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def cli():
    """
    Sort out parameters here
    :return:
    """
    parser = argparse.ArgumentParser(description='Process parameters')
    parser.add_argument('--model_parameter', type=str, default='config/model_example.yml',
                        help='parameter file path')
    parser.add_argument('--data_config_path', type=str, default=None,
                        help='Process file from raw data')
    parser.add_argument('--raw_data_path', type=str, default="raw_data/20230626",
                        help='Process file from raw data')
    parser.add_argument('--data_path', type=str, default="data/vertexAI_PPIT_data.csv", help='Clean data path')
    parser.add_argument('--project_id', type=str, default="gs://cloud-ai-platform-37c994cb-e090-4da1-ab7d-599262814bd1",
                        help='Clean data dir')
    parser.add_argument('--baseline_search', type=bool, default=False,
                        help='Looking for baseline model')


    args = parser.parse_args().__dict__


    process_raw_data: bool = none_or_str(args["data_config_path"]) is not None
    data_config_path: str = args.pop("data_config_path")
    raw_data_path: str = args.pop("raw_data_path")
    project_id: str = args.pop("project_id")
    baseline_search: bool = args.pop("baseline_search")

    def download_if_not_exist(path=""):
        path = os.path.normpath(path)
        local_path = os.path.join(*path.split(os.sep))
        remote_path = '/'.join([project_id, *path.split(os.sep)])
        path_exist = os.path.exists(local_path)
        try:
            if path_exist:
                logging.info(f"File(s) exists at local {local_path}")
            else:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                os.system(f"gsutil -m cp -r {remote_path} {os.path.dirname(local_path)}")
                os.system(f"gsutil -m cp {remote_path} {os.path.dirname(local_path)}")
                logging.info(f"Downloaded file(s) from project bucket {remote_path} to local {local_path}")
        except Exception as e:
            raise CustomException(e, sys)
        return local_path

    def upload(path="config/data_processor_example.yml"):
        path = os.path.normpath(path)
        local_path = os.path.join(*path.split(os.sep))
        remote_path = '/'.join([project_id, *path.split(os.sep)[:-1]])
        try:
            os.system(f"gsutil -m cp -r {local_path} {remote_path}")
            os.system(f"gsutil -m cp {local_path} {remote_path}")
        except Exception as e:
            raise CustomException(e, sys)
        logging.info(f"Upload file(s) from local{local_path} to project bucket {remote_path}")
        return

    if process_raw_data:
        data_config_path = download_if_not_exist(path=data_config_path)
        download_if_not_exist(path=raw_data_path)
        processor = DataProcessor(config_path=data_config_path)
        data_path = processor()
        upload(data_path)
        logging.info(f"Clean data saved and uploaded: {data_path}", )

    else:
        data_path: str = args.pop("data_path")
        data_path = download_if_not_exist(path=data_path)
        logging.info(f"Clean data path located: {data_path}")

        data_loader = DataLoader(data_path)
        data = data_loader()

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(*data)

    try:
        if baseline_search:
            baseline_search = BaselineSearch()
            baseline_search(
                X_train.reshape(X_train.shape[0], -1),
                X_test.reshape(X_test.shape[0], -1),
                y_train.reshape(y_train.shape[0], -1),
                y_test.reshape(y_test.shape[0], -1)
            )
            logging.info("Finished searching for baseline model")
    except Exception as e:
        raise CustomException(e, sys)

    try:
        model = CNN("config/model_params_example.yml")
        model.build()
        model.debug_compile_fit(X_train[:64], X_test[:64], y_train[:64], y_test[:64])
        # model.compile()
        # model.fit(X_train, X_test, y_train, y_test)
        # model.save()
        logging.info("Finished training pipeline")
    except Exception as e:
        raise CustomException(e, sys)





if __name__=="__main__":
    cli()
