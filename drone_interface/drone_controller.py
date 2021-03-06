import airsim
import numpy as np

from airsim.types import Quaternionr
from scipy.spatial.transform import Rotation as ScipyRotation

from drone_interface import DroneActions


class DroneController:
    """
    High level drone controller with 6 discrete actions.
    """

    def __init__(self, initial_height: int = None, drone_name: str = ""):
        # self._speed_level_to_speed = {
        #     DroneActions.SPEED_LEVEL_1: 3,
        #     DroneActions.SPEED_LEVEL_2: 5,
        #     DroneActions.SPEED_LEVEL_3: 5,
        # }
        # self._base_speed = self._speed_level_to_speed[DroneActions.SPEED_LEVEL_2]
        self._base_speed = 5

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
        # if action in [DroneActions.SPEED_LEVEL_1, DroneActions.SPEED_LEVEL_2, DroneActions.SPEED_LEVEL_3]:
        #     self._base_speed = self._speed_level_to_speed[action]
        # else:
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

    def reset(self,
              position_x: float = 0.0,
              position_y: float = 0.0,
              position_z: float = 0.0,
              quat_x: float = 0.0,
              quat_y: float = 0.0,
              quat_z: float = 0.0,
              quat_w: float = 0.0,
              ) -> None:
        self._client.reset()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

        position = airsim.Vector3r(position_x,
                                   position_y,
                                   position_z)

        heading = Quaternionr()
        heading.x_val = quat_x
        heading.y_val = quat_y
        heading.z_val = quat_z
        heading.w_val = quat_w

        pose = airsim.Pose(position, heading)
        self._client.simSetVehiclePose(pose, True)
