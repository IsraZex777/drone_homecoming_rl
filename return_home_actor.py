import math
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as ScipyRotation

from constants import (
    max_distance
)

from utils import calculate_yaw_diff


class ReturnHomeActor:
    def __init__(self, forward_path_csv_path: str):
        self.forward_sensors = pd.read_csv(forward_path_csv_path)
        self.init_x_position = None
        self.init_y_position = None
        self.init_z_position = None

        # In the future predict this change
        self.curr_x_position = None
        self.curr_y_position = None
        self.curr_z_position = None

        self.reset_forwarding_info()

    def reset_forwarding_info(self):
        self.init_x_position = self.forward_sensors.iloc[0]["position_x"]
        self.init_y_position = self.forward_sensors.iloc[0]["position_y"]
        self.init_z_position = self.forward_sensors.iloc[0]["position_z"]

        # In the future predict this change
        self.curr_x_position = self.forward_sensors.iloc[-1]["position_x"]
        self.curr_y_position = self.forward_sensors.iloc[-1]["position_y"]
        self.curr_z_position = self.forward_sensors.iloc[-1]["position_z"]

    def observation_to_state(self, obs_data: pd.DataFrame):
        # In the future predict this change
        self.curr_x_position = obs_data.iloc[-1]["position_x"]
        self.curr_y_position = obs_data.iloc[-1]["position_y"]
        self.curr_z_position = obs_data.iloc[-1]["position_z"]

        yaw_diff = calculate_yaw_diff(np.array([obs_data.iloc[-1]["orientation_x"],
                                                obs_data.iloc[-1]["orientation_y"],
                                                obs_data.iloc[-1]["orientation_z"],
                                                obs_data.iloc[-1]["orientation_w"]]),
                                      np.array((self.curr_x_position, self.curr_y_position, self.curr_z_position)),
                                      np.array((self.init_x_position, self.init_y_position, self.init_z_position)))

        if "distance" in obs_data.iloc[-1]:
            distance = int(obs_data.iloc[-1]["distance"])
        else:
            distance = max_distance

        state = tf.convert_to_tensor(np.array([self.curr_x_position - self.init_x_position,
                                               self.curr_y_position - self.init_y_position,
                                               self.curr_z_position - self.init_z_position,
                                               yaw_diff,
                                               distance / max_distance]))

        return state
