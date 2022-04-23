import time
import airsim
import numpy as np

from enum import Enum, unique
from pynput import keyboard
from scipy.spatial.transform import Rotation as ScipyRotation


@unique
class DroneActions(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    UP = "up"
    DOWN = "down"
    SPEED_LEVEL_1 = "speed_level_1"
    SPEED_LEVEL_2 = "speed_level_2"
    SPEED_LEVEL_3 = "speed_level_3"
    STOP = "stop"


class DroneController:
    """
    High level drone controller with 6 discrete actions.
    """

    def __init__(self, initial_height: int = None, drone_name: str = ""):

        self._speed_level_to_speed = {
            DroneActions.SPEED_LEVEL_1: 3,
            DroneActions.SPEED_LEVEL_2: 5,
            DroneActions.SPEED_LEVEL_3: 10,
        }
        self._base_speed = self._speed_level_to_speed[DroneActions.SPEED_LEVEL_2]

        self.acceleration = 3.0
        self.angular_velocity = 90.0
        self.duration = .4

        self.desired_velocity = np.zeros(3, dtype=np.float32)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True, vehicle_name=drone_name)
        self._client.simEnableWeather(False)

        # Take off to the input height
        if initial_height:
            self._client.moveToZAsync(-initial_height, 20).join()

    def move(self, velocity, yaw_rate):
        self._client.moveByVelocityAsync(velocity[0].item(), velocity[1].item(), velocity[2].item(), self.duration,
                                         drivetrain=airsim.DrivetrainType.ForwardOnly,
                                         yaw_mode=airsim.YawMode(True, yaw_rate))

    def handle_action(self, action: DroneActions):
        if action in [DroneActions.SPEED_LEVEL_1, DroneActions.SPEED_LEVEL_2, DroneActions.SPEED_LEVEL_3]:
            self._base_speed = self._speed_level_to_speed[action]
        else:
            drone_orientation = ScipyRotation.from_quat(self._client.simGetVehiclePose().orientation.to_numpy_array())
            yaw = drone_orientation.as_euler('zyx')[0]
            forward_direction = np.array([np.cos(yaw), np.sin(yaw), 0])
            left_direction = np.array([np.cos(yaw - np.deg2rad(90)), np.sin(yaw - np.deg2rad(90)), 0])

            action_to_velocity = {
                DroneActions.FORWARD: forward_direction * self.duration * self._base_speed,
                DroneActions.BACKWARD: -1 * forward_direction * self.duration * self._base_speed,
                DroneActions.UP: drone_orientation.apply(np.array([0.0, 0.0, -1.0])) * self.duration * self._base_speed,
                DroneActions.DOWN: -1 * drone_orientation.apply(
                    np.array([0.0, 0.0, -1.0])) * self.duration * self._base_speed,
                DroneActions.TURN_LEFT: drone_orientation.apply(np.zeros(3)),
                DroneActions.TURN_RIGHT: drone_orientation.apply(np.zeros(3)),
                DroneActions.STOP: drone_orientation.apply(np.zeros(3)),
            }

            action_to_yaw_rate = {
                DroneActions.FORWARD: 0.0,
                DroneActions.BACKWARD: 0.0,
                DroneActions.UP: 0.0,
                DroneActions.DOWN: 0.0,
                DroneActions.TURN_LEFT: 0.0 - self.angular_velocity,
                DroneActions.TURN_RIGHT: 0.0 + self.angular_velocity,
                DroneActions.STOP: 0.0,
            }

            self.desired_velocity = action_to_velocity[action]
            self.move(self.desired_velocity, action_to_yaw_rate[action])
