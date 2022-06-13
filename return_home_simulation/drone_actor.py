import os
import time
import numpy as np
import random
import threading
import pandas as pd
import tensorflow as tf

from flight_recording.actor_observer import ActorObserver
from position_prediction import (
    PositionPredictor
)
from position_prediction.settings import MODELS_FOLDER_PATH
from rl_global.return_home_actor import ReturnHomeActor
from rl_dqn.actor import make_actor_action
from gps_anomaly_detection import GpsAnomalyDetector
from drone_interface import (
    DroneActions,
    AgentDroneController
)


def run_drone_actor(position_predictor: PositionPredictor,
                    return_home_actor: ReturnHomeActor,
                    q_model: tf.keras.Model,
                    should_create_anomaly: threading.Event,
                    drone_name: str = "drone1"):
    observer = ActorObserver(drone_name=drone_name)
    anomaly_detector = GpsAnomalyDetector()
    init_observation = None
    is_under_attack = False
    reached_target = False

    # listens to input until detects gps anomaly
    observer.start_flight_recording()
    time.sleep(.5)
    while not is_under_attack:
        # print(is_under_attack)
        init_observation = observer.get_recording_data()
        pos_obs = init_observation[["position_x", "position_y", "position_z"]]
        gps_obs = init_observation[["gps_latitude", "gps_longitude"]]

        # creates anomaly
        if should_create_anomaly.is_set():
            new_gps_values = gps_obs.iloc[-1] * (1 + np.random.randn(2) * 0.1)
            gps_obs = gps_obs.append(new_gps_values, ignore_index=True)

        x_offset, y_offset, z_offset = position_predictor.predict_position_offset(init_observation)

        real_x_offset = pos_obs["position_x"].iloc[-1] - pos_obs["position_x"].iloc[0]
        real_y_offset = pos_obs["position_y"].iloc[-1] - pos_obs["position_y"].iloc[0]
        real_z_offset = pos_obs["position_z"].iloc[-1] - pos_obs["position_z"].iloc[0]
        print(
            f"last received GPS: {gps_obs.iloc[-1].to_numpy()}, "
            f"Predicted vs Real offset: ({x_offset :.2f}, {y_offset :.2f}, {z_offset:.2f}) - "
            f"({real_x_offset :.2f}, {real_y_offset :.2f}, {real_z_offset:.2f}) ")

        gps_points = [gps_obs.iloc[0].to_numpy(), gps_obs.iloc[-1].to_numpy()]
        # pos_points = [pos_obs.iloc[0].to_numpy(), pos_obs.iloc[-1].to_numpy()]
        pos_points = [np.array([0, 0]), np.array([x_offset, y_offset])]
        is_under_attack = anomaly_detector.is_under_attack(gps_points, pos_points)

    print(f"Anomaly was detected! returns drone back to base")
    controller = AgentDroneController(drone_name=drone_name)
    controller.handle_action(DroneActions.STOP, duration=2, stop_duration=0)

    return_home_actor.reset_forwarding_info_with_sensors(init_observation)
    observer.reset_recording_data()
    time.sleep(.1)

    while not reached_target:
        curr_observation = observer.get_recording_data()
        observer.reset_recording_data()
        state = return_home_actor.observation_to_normalized_state(curr_observation)
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)

        action = make_actor_action(q_model, state_tensor, epsilon=0)
        action_type, action_duration = action

        print(f"Drone Action: {action_type.name}")
        controller.handle_action(action_type, duration=action_duration, stop_duration=3)

        if state[0] < 0.05:
            reached_target = True

    print("Drone reached target")
    controller.handle_action(DroneActions.STOP, duration=4, stop_duration=0)

#
