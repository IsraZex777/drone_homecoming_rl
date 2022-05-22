import os
from vincenty import vincenty
RECORD_COLUMNS = [
    "gps_altitude", "gps_latitude", "gps_longitude",
    "position_x", "position_y", "position_z",
    "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w", "motor_state_timestamp",
    "barometer_altitude", "barometer_pressure", "barometer_qnh", "barometer_timestamp",
    "magnetometer_magnetic_field_body_x", "magnetometer_magnetic_field_body_y",
    "magnetometer_magnetic_field_body_z", "magnetometer_timestamp",
    "rotor_a_speed", "rotor_a_thrust", "rotor_a_torque_scaler",
    "rotor_b_speed", "rotor_b_thrust", "rotor_b_torque_scaler",
    "rotor_c_speed", "rotor_c_thrust", "rotor_c_torque_scaler",
    "rotor_d_speed", "rotor_d_thrust", "rotor_d_torque_scaler",
    "rotor_timestamp"
]

OBSERVATION_COLUMNS = [
    "gps_altitude", "gps_latitude", "gps_longitude",
    "position_x", "position_y", "position_z",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    "barometer_altitude", "barometer_pressure", "has_collided",
    "distance", "motor_state_timestamp",
]

TIMESTAMP_COLUMNS = [
    "motor_state_timestamp",
    "barometer_timestamp",
    "magnetometer_timestamp",
    "rotor_timestamp"
]
MAIN_TIMESTAMP_COLUMN = "motor_state_timestamp"

MIN_X_POS = -150
MAX_X_POS = 150
MIN_Y_POS = -150
MAX_Y_POS = 150
MIN_Z_POS = 50
MAX_Z_POS = 200
MIN_VELOCITY = 1
MAX_VELOCITY = 20
MIN_TIMEOUT = 5
MAX_TIMEOUT = 20

POSITION_NUM = 0
RUN_HOURS = 5
RUN_MINUTES = RUN_HOURS * 60 + 2
RUN_SECONDS = RUN_MINUTES * 60

RECORDS_FOLDER = os.path.join(os.path.dirname(__file__), "recordings")

IS_SIM_CLOCK_FASTER = False
