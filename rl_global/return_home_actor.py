import logging
import math
import os.path
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf

from .constants import (
    max_distance
)

from position_prediction import (
    PositionPredictor
)
from position_prediction.settings import (
    MODELS_FOLDER_PATH,
    DATA_FOLDER_PATH
)
from position_prediction.utils import (
    load_model_with_scalers_binary
)
from .utils import calculate_yaw_diff


class ReturnHomeActor:
    def __init__(self,
                 forward_path_csv_path: str,
                 pos_prediction_model_name: str = "",
                 logger: logging.Logger = logging.getLogger("dummy")):

        self.logger = logger
        self.forward_path_csv_path = ""
        self.forward_sensors = None
        self.init_x_position = None
        self.init_y_position = None
        self.init_z_position = None

        # In the future predict this change
        self.curr_x_position = None
        self.curr_y_position = None
        self.curr_z_position = None

        self.reset_forwarding_info(forward_path_csv_path)

        self.pos_prediction_model_name = pos_prediction_model_name
        self.model = ""
        self.scaler_x = ""
        self.scaler_y = ""
        self.position_predictor = ""

        if pos_prediction_model_name:
            prediction_model_path = os.path.join(MODELS_FOLDER_PATH, pos_prediction_model_name)
            self.model, self.scaler_x, self.scaler_y = load_model_with_scalers_binary(prediction_model_path)
            self.position_predictor = PositionPredictor(self.model, self.scaler_x, self.scaler_y)

    def reset_forwarding_info(self, forward_path_csv_path=""):
        if forward_path_csv_path != self.forward_path_csv_path:
            self.forward_path_csv_path = forward_path_csv_path
            self.forward_sensors = pd.read_csv(forward_path_csv_path)

        self.init_x_position = self.forward_sensors.iloc[0]["position_x"]
        self.init_y_position = self.forward_sensors.iloc[0]["position_y"]
        self.init_z_position = self.forward_sensors.iloc[0]["position_z"]

        # In the future predict this change
        self.curr_x_position = self.forward_sensors.iloc[-1]["position_x"]
        self.curr_y_position = self.forward_sensors.iloc[-1]["position_y"]
        self.curr_z_position = self.forward_sensors.iloc[-1]["position_z"]

        self.real_x_position = self.forward_sensors.iloc[-1]["position_x"]
        self.real_y_position = self.forward_sensors.iloc[-1]["position_y"]
        self.real_z_position = self.forward_sensors.iloc[-1]["position_z"]

    def observation_to_normalized_state(self, obs_data: pd.DataFrame, save_observation_data=False):
        # if using the prediction model

        if save_observation_data:
            obs_data.to_csv(os.path.join(DATA_FOLDER_PATH,
                                         f"observation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"))

        if self.pos_prediction_model_name:
            x_offset, y_offset, z_offset = self.position_predictor.predict_position_offset(obs_data)

            pred_vs_real = f"pos_x_offset: {x_offset} - {obs_data.iloc[-1]['position_x'] - self.real_x_position}\n" \
                           f"pos_y_offset: {y_offset} - {obs_data.iloc[-1]['position_y'] - self.real_y_position}\n" \
                           f"pos_z_offset: {z_offset} - {obs_data.iloc[-1]['position_z'] - self.real_z_position}\n"

            self.curr_x_position += x_offset
            self.curr_z_position += y_offset
            self.curr_y_position += z_offset

            print(pred_vs_real)
            self.logger.debug(f"pred_model_vs_real\n{pred_vs_real}")

        else:
            self.curr_x_position = obs_data.iloc[-1]["position_x"]
            self.curr_y_position = obs_data.iloc[-1]["position_y"]
            self.curr_z_position = obs_data.iloc[-1]["position_z"]

        self.real_x_position = obs_data.iloc[-1]["position_x"]
        self.real_y_position = obs_data.iloc[-1]["position_y"]
        self.real_z_position = obs_data.iloc[-1]["position_z"]

        yaw_diff = calculate_yaw_diff(np.array([obs_data.iloc[-1]["orientation_x"],
                                                obs_data.iloc[-1]["orientation_y"],
                                                obs_data.iloc[-1]["orientation_z"],
                                                obs_data.iloc[-1]["orientation_w"]]),
                                      np.array((self.curr_x_position, self.curr_y_position, self.curr_z_position)),
                                      np.array((self.init_x_position, self.init_y_position, self.init_z_position)))

        if "distance" in obs_data.iloc[-1]:
            distance = float(obs_data.iloc[-1]["distance"])
        else:
            distance = max_distance

        horizontal_distance_scalar = math.sqrt((self.curr_x_position - self.init_x_position) ** 2 +
                                               (self.curr_y_position - self.init_y_position) ** 2)
        vertical_distance_scalar = abs(self.curr_z_position - self.init_z_position)

        state = tf.convert_to_tensor(np.array([horizontal_distance_scalar / 100,
                                               vertical_distance_scalar / 50,
                                               abs(yaw_diff / 180),
                                               yaw_diff > 0,
                                               yaw_diff < 0,
                                               distance / max_distance]))

        return state
