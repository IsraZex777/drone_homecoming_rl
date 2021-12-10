import numpy as np
from airsim import (
    MultirotorState,
    BarometerData,
    MagnetometerData,
)


class Observation:
    def __init__(self, multi_rotor_state: MultirotorState,
                 barometer_data: BarometerData = None,
                 magnetometer_data: MagnetometerData = None):
        
        self.collision = {
            "has_collied": multi_rotor_state.collision.has_collided,
            "collision_power": multi_rotor_state.collision.normal.to_numpy_array()
        }
        self.position = multi_rotor_state.kinematics_estimated.position.to_numpy_array()
        self.angular_acceleration = multi_rotor_state.kinematics_estimated.angular_acceleration.to_numpy_array()
        self.angular_velocity = multi_rotor_state.kinematics_estimated.angular_velocity.to_numpy_array()
        self.linear_acceleration = multi_rotor_state.kinematics_estimated.angular_acceleration.to_numpy_array()
        self.linear_velocity = multi_rotor_state.kinematics_estimated.angular_acceleration.to_numpy_array()
        self.orientation = multi_rotor_state.kinematics_estimated.orientation.to_numpy_array()
        self.multi_rotor_state_timestamp = multi_rotor_state.timestamp

    def to_numpy(self):
        collision_array = np.concatenate((np.array([self.collision["has_collied"]]), self.collision["collision_power"]))
        numpy_array = np.concatenate((collision_array,
                                      self.position,
                                      self.angular_acceleration,
                                      self.angular_velocity,
                                      self.linear_acceleration,
                                      self.linear_velocity,
                                      self.orientation,
                                      np.array([self.multi_rotor_state_timestamp]),))
        return numpy_array
