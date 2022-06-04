import os
import math
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from ddpg.settings import (
    RL_REPLAY_MEMORY_FOLDER_PATH
)
from replay_memory import ReplayMemory


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


def save_replay_memory_to_file(file_name: str, memory: ReplayMemory) -> None:
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


def load_replay_memory_from_file(file_name: str) -> ReplayMemory:
    """
    Loads replay memory from file
    @param file_name: source file name
    @return:
    """
    file_path = os.path.join(RL_REPLAY_MEMORY_FOLDER_PATH, f"{file_name}.bin")

    with open(file_path, "rb") as file:
        dump = file.read()
        print(dump)
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


