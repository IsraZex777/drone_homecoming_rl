import os
import threading
import time
import numpy as np
import pandas as pd
import airsim

from functools import wraps

RECORDS_FOLDER = os.path.join(os.path.dirname(__file__), "recordings")

def start_recording(stop_recording: threading.Event, flight_name: str, drone_name: str = "") -> pd.DataFrame:
    """
    Records flight sensors data while flight and returns it

    :param stop_recording: Threading event to stop recording
    :param flight_name: Flight name
    :param drone_name: Drone name to collect data of
    :return: Sensors data as pandas dataframe
    """
    DATA_COLUMNS = ["gps_altitude", "gps_latitude", "gps_longitude",
                    "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z",
                    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
                    "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
                    "orientation_x", "orientation_y", "orientation_z", "orientation_w", "motor_state_timestamp",
                    "barometer_altitude", "barometer_pressure", "barometer_qnh", "barometer_timestamp",
                    "magnetometer_magnetic_field_body_x", "magnetometer_magnetic_field_body_y",
                    "magnetometer_magnetic_field_body_x", "agnetometer_timestamp"]
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    all_records = pd.DataFrame(columns=DATA_COLUMNS)

    while not stop_recording.is_set():
        multi_rotor_state = client.getMultirotorState()
        barometer_state = client.getBarometerData()
        magnetometer_state = client.getMagnetometerData()
        data = np.array([
            multi_rotor_state.gps_location.altitude,
            multi_rotor_state.gps_location.latitude,
            multi_rotor_state.gps_location.longitude,
            multi_rotor_state.kinematics_estimated.angular_acceleration.x_val,
            multi_rotor_state.kinematics_estimated.angular_acceleration.y_val,
            multi_rotor_state.kinematics_estimated.angular_acceleration.z_val,
            multi_rotor_state.kinematics_estimated.angular_velocity.x_val,
            multi_rotor_state.kinematics_estimated.angular_velocity.y_val,
            multi_rotor_state.kinematics_estimated.angular_velocity.z_val,
            multi_rotor_state.kinematics_estimated.linear_acceleration.x_val,
            multi_rotor_state.kinematics_estimated.linear_acceleration.y_val,
            multi_rotor_state.kinematics_estimated.linear_acceleration.z_val,
            multi_rotor_state.kinematics_estimated.linear_velocity.x_val,
            multi_rotor_state.kinematics_estimated.linear_velocity.y_val,
            multi_rotor_state.kinematics_estimated.linear_velocity.z_val,
            multi_rotor_state.kinematics_estimated.orientation.x_val,
            multi_rotor_state.kinematics_estimated.orientation.y_val,
            multi_rotor_state.kinematics_estimated.orientation.z_val,
            multi_rotor_state.kinematics_estimated.orientation.w_val,
            multi_rotor_state.timestamp,
            barometer_state.altitude,
            barometer_state.pressure,
            barometer_state.qnh,
            barometer_state.time_stamp,
            magnetometer_state.magnetic_field_body.x_val,
            magnetometer_state.magnetic_field_body.y_val,
            magnetometer_state.magnetic_field_body.z_val,
            magnetometer_state.time_stamp,
        ])
        record = pd.DataFrame(data.reshape(1, -1), columns=DATA_COLUMNS)
        all_records = all_records.append(record, ignore_index=True)
        time.sleep(0.010)
    print(os.path.join(RECORDS_FOLDER,f"{flight_name}_record_data.xlsx"))
    all_records.to_excel(os.path.join(RECORDS_FOLDER,f"{flight_name}_record_data.xlsx"))


def create_flight_recording(flight_name: str):
    """
    Decorator to record flight data while the inner function executes
    """
    def decorator(func):
        def inner(*args, **kwargs):
            stop_event = threading.Event()
            record_thread = threading.Thread(target=start_recording, args=[stop_event, flight_name])
            record_thread.start()
            func(*args, **kwargs)
            stop_event.set()
            record_thread.join()
        return inner
    return decorator

def record_for_seconds(flight_name: str, seconds: int):
    """
    Records flight info for the input amount of seconds

    :param flight_name: Name of the flight
    :param seconds: The amount of seconds to record
    """
    stop_event = threading.Event()
    record_thread = threading.Thread(target=start_recording, args=[stop_event, flight_name])

    record_thread.start()
    time.sleep(seconds)

if __name__ == "__main__":
    # record_for_seconds("test_flight", 20)
    coolll()




