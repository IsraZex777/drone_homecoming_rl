import os
import pandas as pd
import tensorflow as tf

from flight_recording.actor_observer import ActorObserver
from position_prediction import (
    PositionPredictor
)
from return_home_simulation import (
    run_drone_actor
)
from position_prediction.settings import MODELS_FOLDER_PATH
from rl_global.return_home_actor import ReturnHomeActor
from gps_anomaly_detection import GpsAnomalyDetector

from position_prediction import load_model_with_scalers_binary

if __name__ == "__main__":
    model_name = "ann_pos_13Jun_next_generation"
    prediction_model_path = os.path.join(MODELS_FOLDER_PATH, model_name)
    print(prediction_model_path)
    model, scaler_x, scaler_y = load_model_with_scalers_binary(prediction_model_path)
    pos_predictor = PositionPredictor(model, scaler_x, scaler_y)

    forward_path_name = "forward_path-turn_left_forward_record.csv"
    return_home_actor = ReturnHomeActor(forward_path_name, model_name)

    run_drone_actor(pos_predictor, return_home_actor)
