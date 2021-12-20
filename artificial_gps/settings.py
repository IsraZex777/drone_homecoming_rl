import os
import tensorflow as tf
from flight_recording import INPUT_DATA_COLUMNS

BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER_NAME = "data"
DATA_FOLDER_PATH = os.path.join(BASE_DIR, DATA_FOLDER_NAME)
MODELS_FOLDER_NAME = "models"
MODELS_FOLDER_PATH = os.path.join(BASE_DIR, MODELS_FOLDER_NAME)

INPUT_SEQUENCE_LEN = 10

OUTPUT_DATA_COLUMNS = ["position_x", "position_y", "position_z"]
INPUT_DATA_COLUMNS = [column for column in INPUT_DATA_COLUMNS if column not in OUTPUT_DATA_COLUMNS]


FORCE_CPU_RUN = False
if FORCE_CPU_RUN:
    print("###########################")
    tf.config.set_visible_devices([], 'GPU')
