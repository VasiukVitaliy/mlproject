import logging
from datetime import datetime
import os

now = datetime.now()
LOG_NAME = f"{now.day}_{now.month}-{now.hour}-{now.minute}-{now.second}.log"
os.makedirs("logs", exist_ok=True)
log_file_path = os.path.join("logs", LOG_NAME)

log_format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    format=log_format,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)