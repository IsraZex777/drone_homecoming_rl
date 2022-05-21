import os
import sys
import logging
from logging.handlers import RotatingFileHandler

LOGGING_FOLDER_NAME = "logs"
LOGGING_FOLDER_PATH = os.path.join(os.path.dirname(__file__), LOGGING_FOLDER_NAME)


def create_stdout_logger(level=logging.DEBUG):
    logger = logging.getLogger("stdout_logger")
    logger.handlers = []

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def create_general_logger(logger_file_name: str, level=logging.DEBUG):
    logger = logging.getLogger(logger_file_name)
    logger.handlers = []

    logger.setLevel(level)
    logger.propagate = False

    file_path = os.path.join(LOGGING_FOLDER_PATH, f"{logger_file_name}.txt")
    handler = RotatingFileHandler(file_path)
    handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
