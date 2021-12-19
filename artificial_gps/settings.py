import os
from flight_recording import INPUT_DATA_COLUMNS

INPUT_SEQUENCE_LEN = 10

OUTPUT_DATA_COLUMNS = ["gps_altitude", "gps_latitude", "gps_longitude"]
INPUT_DATA_COLUMNS = [column for column in INPUT_DATA_COLUMNS if column not in OUTPUT_DATA_COLUMNS]

DATA_FOLDER_NAME = "data"
GLOBAL_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), DATA_FOLDER_NAME)
