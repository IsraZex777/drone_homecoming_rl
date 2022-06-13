import os
import pandas as pd
import tensorflow as tf

from flight_recording.actor_observer import ActorObserver
from position_prediction import (
    PositionPredictor
)
from position_prediction.settings import MODELS_FOLDER_PATH
from rl_global.return_home_actor import ReturnHomeActor
from gps_anomaly_detection import GpsAnomalyDetector


def run_drone_actor(position_predictor: PositionPredictor,
                    return_home_actor: ReturnHomeActor,
                    q_model: tf.keras.Model,
                    drone_name: str = "drone1"):
    observer = ActorObserver(drone_name=drone_name)
    anomaly_detector = GpsAnomalyDetector()
    is_under_attack = False

    # listens to input until detects gps anomaly
    observer.start_flight_recording()
    while not is_under_attack:
        curr_observation = observer.get_recording_data()

#
