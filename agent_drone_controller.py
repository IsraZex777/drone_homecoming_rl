import time
from drone_controller import (
    DroneActions,
    DroneController
)


class AgentDroneController:
    def __init__(self, drone_name: str = ""):
        self._controller = DroneController(drone_name=drone_name)

    def apply_action_for_seconds(self, action: DroneActions, duration: float = 1):
        curr_timestamp = time.time()
        while time.time() - curr_timestamp < duration:
            self._controller.handle_action(action)
            time.sleep(.2)

    def handle_action(self, action: DroneActions, duration: float = 1, stop_duration: float = 5.5):
        stop_actions = [DroneActions.UP, DroneActions.DOWN, DroneActions.FORWARD, DroneActions.BACKWARD]
        self.apply_action_for_seconds(action, duration)

        # Waits till drone completely stops
        if action in stop_actions:
            time.sleep(stop_duration)

    def reset(self,
              position_x: float = 0.0,
              position_y: float = 0.0,
              position_z: float = 0.0) -> None:
        self._controller.reset(position_x=position_x,
                               position_y=position_y,
                               position_z=position_z)
