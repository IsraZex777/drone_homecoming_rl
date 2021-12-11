import os

INPUT_SEQUENCE_LEN = 10
OUTPUT_DATA_COLUMNS = ["gps_altitude", "gps_latitude", "gps_longitude"]

DATA_FOLDER_NAME = "data"
GLOBAL_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), DATA_FOLDER_NAME)
