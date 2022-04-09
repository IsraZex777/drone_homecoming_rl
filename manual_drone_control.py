import time

from pynput import keyboard

from drone_controller import (
    DroneController,
    DroneActions
)


class DroneKeyboardInterface:
    """
    High level drone controller for manual drone navigation using a regular keyboard.
    """

    def __init__(self):
        self.drone_controller = DroneController()
        self.duration = .5

        self._key_command_mapping = {
            keyboard.Key.up: DroneActions.FORWARD,
            keyboard.Key.down: DroneActions.BACKWARD,
            keyboard.Key.left: DroneActions.TURN_LEFT,
            keyboard.Key.right: DroneActions.TURN_RIGHT,
            keyboard.KeyCode.from_char("w"): DroneActions.UP,
            keyboard.KeyCode.from_char("s"): DroneActions.DOWN,
            keyboard.KeyCode.from_char("1"): DroneActions.SPEED_LEVEL_1,
            keyboard.KeyCode.from_char("2"): DroneActions.SPEED_LEVEL_2,
            keyboard.KeyCode.from_char("3"): DroneActions.SPEED_LEVEL_3,
        }
        self._current_action = None

    def fly_by_keyboard(self):
        """
        Listens to keyboard events and makes the proper action
        @return:
        """
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as keyboard_listener:
            keyboard_listener.wait()
            while keyboard_listener.running:
                if self._current_action:
                    self.drone_controller.handle_action(self._current_action)
                time.sleep(self.duration / 1)
            keyboard_listener.join()

    def _on_press(self, key):
        """
        Updates the pressed key.
        NOTE - One key can be pressed at a time
        @param key: Keyboard key
        @return: None
        """
        if key in self._key_command_mapping.keys():
            self._current_action = self._key_command_mapping[key]

    def _on_release(self, key):
        """
        NOTE - One key can be pressed at a time
        @param key: Keyboard key
        @return: None
        """
        if key in self._key_command_mapping.keys() and \
                self._current_action == self._key_command_mapping[key]:
            self._current_action = None


if __name__ == "__main__":
    controller = DroneKeyboardInterface()
    controller.fly_by_keyboard()
