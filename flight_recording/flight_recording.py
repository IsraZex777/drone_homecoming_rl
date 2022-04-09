import os
import datetime
import threading
import time
import numpy as np
import pandas as pd
import airsim

from .settings import (
    RECORD_COLUMNS,
    RECORDS_FOLDER,
    IS_SIM_CLOCK_FASTER
)


def start_recording(stop_recording: threading.Event,
                    flight_name: str = "flight",
                    drone_name: str = ""):
    """
    Records flight sensors data while flight and returns it

    :param stop_recording: Threading event to stop recording
    :param flight_name: Flight name
    :param drone_name: Drone name to collect data of
    :return: Sensors data as pandas dataframe
    """

    client = airsim.MultirotorClient()
    client.confirmConnection()
    # client.enableApiControl(True)

    all_records = pd.DataFrame(columns=RECORD_COLUMNS)

    while not stop_recording.is_set():
        multi_rotor_state = client.getMultirotorState(vehicle_name=drone_name)
        barometer_state = client.getBarometerData(vehicle_name=drone_name)
        magnetometer_state = client.getMagnetometerData(vehicle_name=drone_name)
        rotor_state = client.getRotorStates(vehicle_name=drone_name)

        data = np.array([
            multi_rotor_state.gps_location.altitude,
            multi_rotor_state.gps_location.latitude,
            multi_rotor_state.gps_location.longitude,
            multi_rotor_state.kinematics_estimated.position.x_val,
            multi_rotor_state.kinematics_estimated.position.y_val,
            multi_rotor_state.kinematics_estimated.position.z_val,
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
            rotor_state.rotors[0]["speed"],
            rotor_state.rotors[0]["thrust"],
            rotor_state.rotors[0]["torque_scaler"],
            rotor_state.rotors[1]["speed"],
            rotor_state.rotors[1]["thrust"],
            rotor_state.rotors[1]["torque_scaler"],
            rotor_state.rotors[2]["speed"],
            rotor_state.rotors[2]["thrust"],
            rotor_state.rotors[2]["torque_scaler"],
            rotor_state.rotors[3]["speed"],
            rotor_state.rotors[3]["thrust"],
            rotor_state.rotors[3]["torque_scaler"],
            rotor_state.timestamp,
        ])
        record = pd.DataFrame(data.reshape(1, -1), columns=RECORD_COLUMNS)
        all_records = all_records.append(record, ignore_index=True)

        if not IS_SIM_CLOCK_FASTER:
            time.sleep(0.015)
    all_records.to_csv(os.path.join(RECORDS_FOLDER, f"{flight_name}_record.csv"))


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


def record_flight_for_seconds(seconds: int,
                              flight_name: str = f"flight_{datetime.datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}", ):
    """
    Records flight info for the input amount of seconds

    :param flight_name: Name of the flight
    :param seconds: The amount of seconds to record
    """
    stop_event = threading.Event()
    record_thread = threading.Thread(target=start_recording, args=[stop_event, flight_name])

    record_thread.start()
    time.sleep(seconds)
    stop_event.set()
    record_thread.join()


class FlightRecorder:
    def __init__(self, flight_name: str):
        self._flight_name = flight_name
        self._session_no = 1
        self._recording_thread = None
        self._recording_event = None

    def start_flight_recording(self):
        if self.is_recording_running():
            raise ValueError("There is a recording running in the background")

        record_file_name = f"{self._flight_name}_{self._session_no}"
        self._session_no += 1
        self._recording_event = threading.Event()
        self._recording_thread = threading.Thread(target=start_recording,
                                                  args=[self._recording_event, record_file_name])
        self._recording_thread.start()

    def stop_and_save_recording_data(self):
        if self._recording_thread and self._recording_event:
            self._recording_event.set()
            self._recording_thread.join()

    def is_recording_running(self):
        return self._recording_thread and self._recording_thread.is_alive()
