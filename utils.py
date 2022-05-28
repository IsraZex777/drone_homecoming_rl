import math
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as ScipyRotation

from constants import (
    max_distance
)


def calculate_yaw_diff(curr_orientation: np.array,
                       curr_position: np.array,
                       init_position: np.array) -> float:
    """
    calculates yaw diff (normalizes the degrees to a value between -1 and 1)

    @param curr_orientation: Current quaternion representation
    @param curr_position: Current position vector
    @param init_position: Initial position vector
    @return: Normalized diff
    """
    drone_orientation = ScipyRotation.from_quat(np.array(curr_orientation))
    current_yaw_degree = drone_orientation.as_euler('zyx')[0] * (180 / math.pi)

    x_delta = curr_position[0] - init_position[0]
    y_delta = curr_position[1] - init_position[1]
    yaw_to_look_at_degree = math.atan2(-y_delta, -x_delta) * (180 / math.pi)

    # have same sigh
    yaw_dir_right = (current_yaw_degree - yaw_to_look_at_degree) < 0

    yaw_diff = abs(current_yaw_degree - yaw_to_look_at_degree)
    if abs(yaw_diff > 180):
        yaw_diff = 360 - abs(yaw_diff)

    print("#########################")
    print(current_yaw_degree)
    print(yaw_to_look_at_degree)
    print(yaw_dir_right)
    print(yaw_diff)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~`")

    # yaw dir
    yaw_diff *= 1 if yaw_dir_right else -1

    # yaw normalization
    yaw_diff /= 180
    return yaw_diff
