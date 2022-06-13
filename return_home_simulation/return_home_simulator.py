import os
import threading
import numpy as np
import pandas as pd

from tensorflow.keras import Model
from position_prediction import (
    PositionPredictor,
    load_model_with_scalers_binary,
    load_model,
    MODELS_FOLDER_PATH,
)

from rl_global import ReturnHomeActor
from .drone_actor import run_drone_actor
from drone_interface import DroneKeyboardInterface


def run_drone_return_to_home_simulation(pos_prediction_model: str,
                                        q_model_name: str):
    prediction_model_path = os.path.join(MODELS_FOLDER_PATH, pos_prediction_model)
    model, scaler_x, scaler_y = load_model_with_scalers_binary(prediction_model_path)
    pos_predictor = PositionPredictor(model, scaler_x, scaler_y)

    q_model = load_model(os.path.join(MODELS_FOLDER_PATH, q_model_name))
    q_model = Model(q_model.inputs[0], q_model.layers[-4].output)

    # return_home_actor = ReturnHomeActor(pos_prediction_model_name=pos_prediction_model)
    return_home_actor = ReturnHomeActor()

    print("You have the control.")
    print("Press escape to create the gps anomaly")

    should_create_anomaly_event = threading.Event()
    actor_thread = threading.Thread(target=run_drone_actor,
                                    args=[pos_predictor, return_home_actor, q_model, should_create_anomaly_event])

    actor_thread.start()

    client_controller = DroneKeyboardInterface()
    client_controller.fly_by_keyboard()
    should_create_anomaly_event.set()
    actor_thread.join()
