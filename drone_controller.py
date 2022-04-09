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


class DroneController:
    """
    High level drone controller with 6 discrete actions.
    """

    def __init__(self, initial_location=""):

        self._speed_level_to_speed = {
            DroneActions.SPEED_LEVEL_1: 3,
            DroneActions.SPEED_LEVEL_2: 5,
            DroneActions.SPEED_LEVEL_3: 10,
        }
        self._base_speed = self._speed_level_to_speed[DroneActions.SPEED_LEVEL_2]

        self.acceleration = 3.0
        self.angular_velocity = 90.0
        self.duration = .5

        self.desired_velocity = np.zeros(3, dtype=np.float32)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.simEnableWeather(False)
        self._client.takeoffAsync()

    def move(self, velocity, yaw_rate):
        self._client.moveByVelocityAsync(velocity[0].item(), velocity[1].item(), velocity[2].item(), self.duration,
                                         drivetrain=airsim.DrivetrainType.ForwardOnly,
                                         yaw_mode=airsim.YawMode(True, yaw_rate))

    def handle_action(self, action: DroneActions):
        if action in [DroneActions.SPEED_LEVEL_1, DroneActions.SPEED_LEVEL_2, DroneActions.SPEED_LEVEL_3]:
            print(action)
            self._base_speed = self._speed_level_to_speed[action]
        else:
            drone_orientation = ScipyRotation.from_quat(self._client.simGetVehiclePose().orientation.to_numpy_array())
            yaw = drone_orientation.as_euler('zyx')[0]
            forward_direction = np.array([np.cos(yaw), np.sin(yaw), 0])

            action_to_velocity = {
                DroneActions.FORWARD: forward_direction * self.duration * self._base_speed,
                DroneActions.BACKWARD: -1 * forward_direction * self.duration * self._base_speed,
                DroneActions.UP: drone_orientation.apply(np.array([0.0, 0.0, -1.0])) * self.duration * self._base_speed,
                DroneActions.DOWN: -1 * drone_orientation.apply(
                    np.array([0.0, 0.0, -1.0])) * self.duration * self._base_speed,
                DroneActions.TURN_LEFT: np.dot(self.desired_velocity, forward_direction) * forward_direction,
                DroneActions.TURN_RIGHT: np.dot(self.desired_velocity, forward_direction) * forward_direction,
            }

            action_to_yaw_rate = {
                DroneActions.FORWARD: 0.0,
                DroneActions.BACKWARD:  0.0,
                DroneActions.UP:  0.0,
                DroneActions.DOWN:  0.0,
                DroneActions.TURN_LEFT: 0.0 - self.angular_velocity,
                DroneActions.TURN_RIGHT:  0.0 + self.angular_velocity,
            }

            self.move(action_to_velocity[action], action_to_yaw_rate[action])

        #
        # if self._active_commands["forward"] or self._active_commands["backward"]:
        #     forward_increment = forward_direction * self.duration * self.acceleration
        #     if self._active_commands["forward"]:
        #         self.desired_velocity += forward_increment
        #     else:
        #         self.desired_velocity -= forward_increment
        # else:
        #     forward_component = np.dot(self.desired_velocity, forward_direction) * forward_direction
        #     self.desired_velocity -= self.friction * forward_component
        #
        # if self._active_commands["up"] or self._active_commands["down"]:
        #     vertical_component =
        #     vertical_component *= *self.acceleration
        #     if self._active_commands["up"]:
        #         self.desired_velocity += vertical_component
        #     else:
        #         self.desired_velocity -= vertical_component
        # else:
        #     self.desired_velocity[2] *= self.friction
        #
        # if self._active_commands["left"] or self._active_commands["right"]:
        #     lateral_increment = left_direction * self.duration * self.acceleration
        #     if self._active_commands["left"]:
        #         self.desired_velocity += lateral_increment
        #     else:
        #         self.desired_velocity -= lateral_increment
        # else:
        #     left_component = np.dot(self.desired_velocity, left_direction) * left_direction
        #     self.desired_velocity -= self.friction * left_component
        #
        # speed = np.linalg.norm(self.desired_velocity)
        # if speed > self.max_speed:
        #     self.desired_velocity = self.desired_velocity / speed * self.max_speed

        # yaw_rate =
        # if self._active_commands["turn left"]:
        #     yaw_rate = -self.angular_velocity
        # elif self._active_commands["turn right"]:
        #     yaw_rate = self.angular_velocity

        # self.move(self.desired_velocity, yaw_rate)


if __name__ == "__main__":
    controller = DroneController()
    controller.fly_by_keyboard()
