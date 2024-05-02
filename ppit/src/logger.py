import logging
import os
from datetime import datetime

LOG_FOLDER = f"{datetime.now().strftime('%m_%d_%Y')}.log"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join("logs", LOG_FOLDER)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(filename)s %(module)s %(funcName)s %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

