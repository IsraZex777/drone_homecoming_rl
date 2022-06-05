import os
import math
import pickle
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as ScipyRotation

from .constants import (
    RL_REPLAY_MEMORY_FOLDER_PATH
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

    # yaw dir
    yaw_diff *= 1 if yaw_dir_right else -1

    return yaw_diff


def save_replay_memory_to_file(file_name: str, memory) -> None:
    """
    Saves replay memory to file input file_name

    @param file_name: target file name
    @param memory: replay memory object
    @return: None
    """
    dump = pickle.dumps(memory)
    file_path = os.path.join(RL_REPLAY_MEMORY_FOLDER_PATH, f"{file_name}.bin")

    with open(file_path, "bw") as file:
        file.write(bytes(dump))


def load_replay_memory_from_file(file_name: str):
    """
    Loads replay memory from file
    @param file_name: source file name
    @return: The replay memory
    """
    file_path = os.path.join(RL_REPLAY_MEMORY_FOLDER_PATH, f"{file_name}.bin")

    with open(file_path, "rb") as file:
        dump = file.read()
        memory = pickle.loads(dump)
        return memory


def is_replay_memory_file_exist(file_name: str) -> bool:
    """
    checks weather replay memory file exists
    @param file_name: replay memory file name
    @return: True if exists, else False
    """
    file_path = os.path.join(RL_REPLAY_MEMORY_FOLDER_PATH, f"{file_name}.bin")

    return os.path.isfile(file_path)


def save_model(model, model_name):
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"{model_name}.h5")


def load_model(model_name):
    with open(f'{model_name}.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{model_name}.h5")

    return loaded_model


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
