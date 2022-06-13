import os

BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER_NAME = "data"
DATA_FOLDER_PATH = os.path.join(BASE_DIR, DATA_FOLDER_NAME)
MODELS_FOLDER_NAME = "models"
MODELS_FOLDER_PATH = os.path.join(BASE_DIR, MODELS_FOLDER_NAME)
TUNERS_FOLDER_NAME = "tuners"
TUNERS_FOLDER_PATH = os.path.join(BASE_DIR, TUNERS_FOLDER_NAME)

ORIENTATION_SENSORS = ["orientation_x", "orientation_y", "orientation_z", "orientation_w"]
ORIENTATION_DIFF_COLUMNS = [f"{field}_diff" for field in ORIENTATION_SENSORS]

LINEAR_ACCELERATION_SENSORS = ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",]
ANGULAR_ACCELERATION_SENSORS = [ "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"]
ANGULAR_VELOCITY_SENSORS = [ "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]
BAROMETER_SENSORS = ["barometer_altitude", "barometer_pressure"]
TIMESTAMP_COLUMN = "motor_state_timestamp"

INPUT_1_CLOUMNS = [*ORIENTATION_SENSORS,
                   *ORIENTATION_DIFF_COLUMNS,
                   *LINEAR_ACCELERATION_SENSORS,
                   *ANGULAR_ACCELERATION_SENSORS]
INPUT_2_CLOUMNS = [TIMESTAMP_COLUMN]
INPUT_3_CLOUMNS = [*BAROMETER_SENSORS]

INPUT_CLOUMNS = [*ORIENTATION_SENSORS,
                 *LINEAR_ACCELERATION_SENSORS,
                 *ANGULAR_ACCELERATION_SENSORS,
                 TIMESTAMP_COLUMN,
                 *BAROMETER_SENSORS
]

OUTPUT_COLUMNS = ["position_x", "position_y", "position_z"]

FORCE_CPU_RUN = False
if FORCE_CPU_RUN:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
