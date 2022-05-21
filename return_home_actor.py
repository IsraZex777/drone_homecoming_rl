import pandas as pd
import numpy as np
import tensorflow as tf


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
        self.init_x_position = self.forward_sensors.iloc[0, "x_position"]
        self.init_y_position = self.forward_sensors.iloc[0, "y_position"]
        self.init_z_position = self.forward_sensors.iloc[0, "z_position"]

        # In the future predict this change
        self.curr_x_position = self.forward_sensors.iloc[-1, "x_position"]
        self.curr_y_position = self.forward_sensors.iloc[-1, "y_position"]
        self.curr_z_position = self.forward_sensors.iloc[-1, "z_position"]

    def observation_to_state(self, obs_data: pd.DataFrame):
        # In the future predict this change
        self.curr_x_position = obs_data.iloc[-1, "x_position"]
        self.curr_y_position = obs_data.iloc[-1, "y_position"]
        self.curr_z_position = obs_data.iloc[-1, "z_position"]

        state = tf.Tensor(np.array([self.curr_x_position - self.init_x_position,
                                    self.curr_y_position - self.init_y_position,
                                    self.curr_z_position - self.init_z_position,
                                    obs_data.iloc[-1, "distance"]]))

        return state
