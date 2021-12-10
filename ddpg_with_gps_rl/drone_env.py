import airsim
import numpy as np
import math

from typing import Tuple

from gym import (
    spaces,
    Env
)

from observation import Observation


class AirSimDroneEnv(Env):

    def __init__(self):
        self.action_space = spaces.Discrete(7)
        self.observation_space = 

        self._target_position = (40, 15, -10)
        self._last_state = None

        self._drone = airsim.MultirotorClient(ip="127.0.0.1")
        self.reset()

    def __del__(self):
        self._drone.reset()

    def _get_obs(self):
        multi_rotor_state = self._drone.getMultirotorState()
        return Observation(multi_rotor_state)

    def _do_action(self, action: np.array):
        quad_vel = self._drone.getMultirotorState().kinematics_estimated.linear_velocity
        self._drone.moveByVelocityAsync(
            quad_vel.x_val + action[0],
            quad_vel.y_val + action[1],
            quad_vel.z_val + action[2],
            5,
        ).join()

    def _compute_reward(self, obs: Observation) -> Tuple[float, bool]:
        factor = 100

        if obs.collision["is_collied"]:
            reward = -1
        else:
            current_position = obs.position

            euclidean_dist = np.linalg.norm(current_position - self._target_position)
            reward = factor / euclidean_dist

        is_done = reward < 0 or reward > factor * 10

        return reward, is_done

    def step(self, action: np.array):
        self._do_action(action)
        obs = self._get_obs()
        last_timestamp = self._last_state.multi_rotor_state_timestamp
        self._last_state = obs
        obs.multi_rotor_state_timestamp -= last_timestamp

        reward, is_done = self._compute_reward(obs)

        return obs, reward, is_done, {}

    def reset(self):
        self._drone.reset()
        self._drone.enableApiControl(True)
        self._drone.armDisarm(True)

        # Set home position and velocity
        self._drone.takeoffAsync()
        self._drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

        obs = self._get_obs()
        self._last_state = obs
        return obs
