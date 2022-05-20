import os
import datetime
import threading
import time
import numpy as np
import pandas as pd
import airsim

from state_collector import StateCollector

from .settings import (
    RECORD_COLUMNS,
    RECORDS_FOLDER,
    IS_SIM_CLOCK_FASTER
)


def record_sensors(state_collector: StateCollector,
                   stop_recording: threading.Event,
                   drone_name: str = ""):
    """
    Records flight sensors data while flight and returns it

    @param state_collector: Objects the collects the input state in a thread safe way
    @param stop_recording: Threading event to stop recording
    @param drone_name: Drone name to collect data of
    """

    client = airsim.MultirotorClient()
    client.confirmConnection()

    while not stop_recording.is_set():
        multi_rotor_state = client.getMultirotorState(vehicle_name=drone_name)
        barometer_state = client.getBarometerData(vehicle_name=drone_name)

        state_collector.append_new_state(
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
            barometer_state.pressure)

        if not IS_SIM_CLOCK_FASTER:
            time.sleep(0.015)


class ActorObserver:
    def __init__(self, flight_name: str):
        self._flight_name = flight_name
        self._state_collector = StateCollector()

        self._recording_thread = None
        self._recording_event = None

    def start_flight_recording(self):
        if self.is_recording_running():
            raise ValueError("There is a recording running in the background")

        self._recording_event = threading.Event()
        self._recording_thread = threading.Thread(target=record_sensors,
                                                  args=[self._state_collector,
                                                        self._recording_event,
                                                        self._flight_name])
        self._recording_thread.start()

    def get_and_reset_recording_data(self) -> np.array:
        states = self._state_collector.get_observed_state()
        self._state_collector.empty_observed_state()
        return states

    def stop_recording_process(self):
        if self._recording_thread and self._recording_event:
            self._recording_event.set()
            self._recording_thread.join()

    def is_recording_running(self):
        return self._recording_thread and self._recording_thread.is_alive()
