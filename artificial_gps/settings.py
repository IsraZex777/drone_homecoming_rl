import os
from flight_recording import INPUT_DATA_COLUMNS

INPUT_SEQUENCE_LEN = 10

                      
OUTPUT_DATA_COLUMNS = [ "position_x", "position_y", "position_z"]

DATA_FOLDER_NAME = "data"
GLOBAL_DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), DATA_FOLDER_NAME)
