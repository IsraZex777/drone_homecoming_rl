import threading
import time
import numpy as np
import airsim

from .state_collector import StateCollector

from rl_global.constants import simulator_time_factor


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
        distance_state = client.getDistanceSensorData(distance_sensor_name="Distance", vehicle_name=drone_name)

        state_collector.append_new_state(
            multi_rotor_state.gps_location.altitude,
            multi_rotor_state.gps_location.latitude,
            multi_rotor_state.gps_location.longitude,
            multi_rotor_state.kinematics_estimated.position.x_val,
            multi_rotor_state.kinematics_estimated.position.y_val,
            multi_rotor_state.kinematics_estimated.position.z_val,
            multi_rotor_state.kinematics_estimated.orientation.x_val,
            multi_rotor_state.kinematics_estimated.orientation.y_val,
            multi_rotor_state.kinematics_estimated.orientation.z_val,
            multi_rotor_state.kinematics_estimated.orientation.w_val,
            multi_rotor_state.kinematics_estimated.angular_acceleration.x_val,
            multi_rotor_state.kinematics_estimated.angular_acceleration.y_val,
            multi_rotor_state.kinematics_estimated.angular_acceleration.z_val,
            multi_rotor_state.kinematics_estimated.angular_velocity.x_val,
            multi_rotor_state.kinematics_estimated.angular_velocity.y_val,
            multi_rotor_state.kinematics_estimated.angular_velocity.z_val,
            multi_rotor_state.kinematics_estimated.linear_acceleration.x_val,
            multi_rotor_state.kinematics_estimated.linear_acceleration.y_val,
            multi_rotor_state.kinematics_estimated.linear_acceleration.z_val,
            barometer_state.altitude,
            barometer_state.pressure,
            multi_rotor_state.collision.has_collided,
            distance_state.distance,
            multi_rotor_state.timestamp,
        )

        if not simulator_time_factor == 1:
            time.sleep(0.015)


class ActorObserver:
    def __init__(self, drone_name: str):
        self._drone_name = drone_name
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
                                                        self._drone_name])
        self._recording_thread.start()

    def get_recording_data(self) -> np.array:
        states = self._state_collector.get_observed_state()
        return states

    def reset_recording_data(self) -> None:
        self._state_collector.empty_observed_state()

    def stop_recording_process(self):
        if self._recording_thread and self._recording_event:
            self._recording_event.set()
            self._recording_thread.join()

    def is_recording_running(self):
        return self._recording_thread and self._recording_thread.is_alive()
