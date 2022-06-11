import os
import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER_NAME = "data"
DATA_FOLDER_PATH = os.path.join(BASE_DIR, DATA_FOLDER_NAME)
MODELS_FOLDER_NAME = "models"
MODELS_FOLDER_PATH = os.path.join(BASE_DIR, MODELS_FOLDER_NAME)
TUNERS_FOLDER_NAME = "tuners"
TUNERS_FOLDER_PATH = os.path.join(BASE_DIR, TUNERS_FOLDER_NAME)

INPUT_SEQUENCE_LEN = 10

OUTPUT_DATA_COLUMNS = ["position_x", "position_y", "position_z"]
INPUT_DATA_COLUMNS = ["angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z",
                      # "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                      "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
                      # "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
                      "orientation_x", "orientation_y", "orientation_z", "orientation_w", "motor_state_timestamp",
                      "barometer_altitude", "barometer_pressure", "barometer_qnh", "barometer_timestamp",
                      "magnetometer_magnetic_field_body_x", "magnetometer_magnetic_field_body_y",
                      "magnetometer_magnetic_field_body_x", "magnetometer_timestamp",
                      "rotor_a_speed", "rotor_a_thrust", "rotor_a_torque_scaler",
                      "rotor_b_speed", "rotor_b_thrust", "rotor_b_torque_scaler",
                      "rotor_c_speed", "rotor_c_thrust", "rotor_c_torque_scaler",
                      "rotor_d_speed", "rotor_d_thrust", "rotor_d_torque_scaler",
                      "rotor_timestamp"]

INPUT_SEQUENCE_LENGTH = 10
INPUT_SEQUENCE_COLUMNS = ["angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z",
                          "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
                          "orientation_x", "orientation_y", "orientation_z", "orientation_w", "motor_state_timestamp",
                          "barometer_altitude", "barometer_pressure", "barometer_qnh", "barometer_timestamp",
                          "magnetometer_magnetic_field_body_x", "magnetometer_magnetic_field_body_y",
                          "magnetometer_magnetic_field_body_x", "magnetometer_timestamp",
                          "rotor_a_speed", "rotor_a_thrust", "rotor_a_torque_scaler",
                          "rotor_b_speed", "rotor_b_thrust", "rotor_b_torque_scaler",
                          "rotor_c_speed", "rotor_c_thrust", "rotor_c_torque_scaler",
                          "rotor_d_speed", "rotor_d_thrust", "rotor_d_torque_scaler",
                          "rotor_timestamp"]
OUTPUT_SEQUENCE_COLUMNS = ["position_x", "position_y", "position_z"]

FORCE_CPU_RUN = False
if FORCE_CPU_RUN:
    print("###########################")
    tf.config.set_visible_devices([], 'GPU')
