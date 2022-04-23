import random
import logging
import time
import numpy as np

from typing import List

from project_logging import create_general_logger
from flight_recording import FlightRecorder
from drone_controller import (
    DroneController,
    DroneActions,
)


class DroneBot:
    def __init__(self,
                 bot_name: str,
                 actions: List[DroneActions],
                 intervals: List[int],
                 logger: logging.Logger = logging.getLogger("dummy")):
        """
        initializes the drone bot
        @param bot_name: Bot name, used for the recording file name
        @param actions: Actions that cam be taken by the bot
        @param intervals: Interval in which the bot applies the randomized actions
        @param logger: Bot Operation logger
        """
        self.logger = logger
        self.recording_name = bot_name

        self.logger.info(f"{self.recording_name} | Connects and takes off")
        self.controller = DroneController(initial_height=100, drone_name="")
        self.logger.info(f"{self.recording_name} | Connected and took off")

        self.recorder = FlightRecorder(self.recording_name)
        self.actions = actions
        self.intervals = intervals
        self.recovery_time = 2

    def apply_action_for_seconds(self, action: DroneActions, period: float = 1):
        curr_timestamp = time.time()
        while time.time() - curr_timestamp < period:
            self.controller.handle_action(action)
            time.sleep(.2)

    def start_flight(self, steps: int = 5):
        self.apply_action_for_seconds(DroneActions.STOP, self.recovery_time)

        self.logger.info(f"{self.recording_name} | Started recording")
        self.recorder.start_flight_recording()
        for step in range(steps):
            action = random.choice(self.actions)
            interval = random.choice(self.intervals) if action not in [DroneActions.TURN_RIGHT,
                                                                       DroneActions.TURN_LEFT] else .5
            self.logger.info(
                f"{self.recording_name} | Step: {step + 1}/{steps} Action: {action.value}, Interval: {interval: .2f}")

            self.apply_action_for_seconds(action, interval)

        self.apply_action_for_seconds(DroneActions.STOP, self.recovery_time)
        self.recorder.stop_and_save_recording_data()
        self.logger.info(f"{self.recording_name} | Stopped recording")

    def start_flight_with_breaks(self, steps: int = 5, break_time: float = 1):
        self.apply_action_for_seconds(DroneActions.STOP, self.recovery_time)
        self.logger.info(f"{self.recording_name} | Started recording")
        self.recorder.start_flight_recording()
        for step in range(steps):
            action = random.choice(self.actions)
            interval = random.choice(self.intervals) if action not in [DroneActions.TURN_RIGHT,
                                                                       DroneActions.TURN_LEFT] else .5
            self.logger.info(
                f"{self.recording_name} | Step: {step + 1}/{steps} Action: {action.value}, Interval: {interval: .2f}")

            self.apply_action_for_seconds(action, interval)
            self.logger.info(f"{self.recording_name} | Goes for a break, break_time: {break_time: .2f}")
            self.apply_action_for_seconds(DroneActions.STOP, break_time)

        self.apply_action_for_seconds(DroneActions.STOP, self.recovery_time)
        self.recorder.stop_and_save_recording_data()
        self.logger.info(f"{self.recording_name} | Stopped recording")

    def start_flight_in_zig_zag_with_breaks(self, steps: int = 5, break_time: float = 2):
        self.apply_action_for_seconds(DroneActions.STOP, self.recovery_time)
        self.logger.info(f"{self.recording_name} | Started recording")
        self.recorder.start_flight_recording()
        for step in range(steps):
            action = random.choice(self.actions)
            interval = random.choice(self.intervals) if action not in [DroneActions.TURN_RIGHT,
                                                                       DroneActions.TURN_LEFT] else .5
            self.logger.info(
                f"{self.recording_name} | Step: {step + 1}/{steps} Action: {action.value}, Interval: {interval: .2f}")

            self.apply_action_for_seconds(action, interval)
            self.logger.info(f"{self.recording_name} | Goes for a break, break_time: {break_time: .2f}")
            self.apply_action_for_seconds(DroneActions.STOP, break_time)

            turn_action = random.choice([DroneActions.TURN_RIGHT, DroneActions.TURN_LEFT])
            self.logger.info(
                f"{self.recording_name} | Turns: {'left' if turn_action == DroneActions.TURN_LEFT else 'right'}")
            self.apply_action_for_seconds(turn_action, .5)

        self.apply_action_for_seconds(DroneActions.STOP, self.recovery_time)
        self.recorder.stop_and_save_recording_data()
        self.logger.info(f"{self.recording_name} | Stopped recording")


def activate_bot_simplified_1(logger: logging.Logger = logging.getLogger("dummy")):
    actions = [DroneActions.FORWARD, DroneActions.BACKWARD, DroneActions.DOWN, DroneActions.UP]
    intervals = list(np.arange(1, 10, 0.4))

    bot = DroneBot(bot_name="bot-simplified-1", actions=actions, intervals=intervals, logger=logger)

    steps = random.choice(list(range(5, 10)))
    bot.start_flight(steps)


def activate_bot_simplified_2(logger: logging.Logger = logging.getLogger("dummy")):
    actions = [DroneActions.FORWARD,
               DroneActions.FORWARD,
               DroneActions.FORWARD,
               DroneActions.BACKWARD,
               DroneActions.BACKWARD,
               DroneActions.BACKWARD,
               DroneActions.DOWN,
               DroneActions.UP]
    intervals = list(np.arange(1, 10, 0.4))

    bot = DroneBot(bot_name="bot-simplified-2", actions=actions, intervals=intervals, logger=logger)

    steps = random.choice(list(range(5, 10)))
    bot.start_flight(steps)


def activate_bot_simplified_3(logger: logging.Logger = logging.getLogger("dummy")):
    actions = [DroneActions.FORWARD,
               DroneActions.BACKWARD,
               DroneActions.DOWN,
               DroneActions.UP]
    intervals = list(np.arange(1, 10, 0.4))

    bot = DroneBot(bot_name="bot-simplified-3", actions=actions, intervals=intervals, logger=logger)

    steps = random.choice(list(range(5, 10)))
    bot.start_flight_with_breaks(steps)


def activate_bot_simplified_4(logger: logging.Logger = logging.getLogger("dummy")):
    actions = [DroneActions.FORWARD,
               DroneActions.BACKWARD]
    intervals = list(np.arange(1, 10, 0.4))

    bot = DroneBot(bot_name="bot-simplified-4", actions=actions, intervals=intervals, logger=logger)

    steps = random.choice(list(range(5, 10)))
    bot.start_flight_with_breaks(steps)


def activate_bot_expert_1(logger: logging.Logger = logging.getLogger("dummy")):
    actions = [DroneActions.FORWARD,
               DroneActions.BACKWARD,
               DroneActions.TURN_LEFT,
               DroneActions.TURN_RIGHT,
               DroneActions.DOWN,
               DroneActions.UP,
               DroneActions.STOP]
    intervals = list(np.arange(1, 10, 0.2))

    bot = DroneBot(bot_name="bot-expert-1", actions=actions, intervals=intervals, logger=logger)

    steps = random.choice(list(range(3, 7)))
    bot.start_flight(steps)


def activate_bot_expert_2(logger: logging.Logger = logging.getLogger("dummy")):
    actions = [DroneActions.FORWARD,
               DroneActions.BACKWARD,
               DroneActions.TURN_LEFT,
               DroneActions.TURN_RIGHT,
               DroneActions.STOP]
    intervals = list(np.arange(1, 10, 0.2))

    bot = DroneBot(bot_name="bot-expert-2", actions=actions, intervals=intervals, logger=logger)

    steps = random.choice(list(range(5, 10)))
    bot.start_flight(steps)


def activate_bot_expert_3(logger: logging.Logger = logging.getLogger("dummy")):
    actions = [DroneActions.FORWARD,
               DroneActions.BACKWARD,
               ]
    intervals = list(np.arange(1, 10, 0.2))

    bot = DroneBot(bot_name="bot-expert-3", actions=actions, intervals=intervals, logger=logger)

    steps = random.choice(list(range(5, 10)))
    bot.start_flight_in_zig_zag_with_breaks(steps)


def activate_bot_train_1(logger: logging.Logger = logging.getLogger("dummy")):
    actions = [DroneActions.FORWARD,
               DroneActions.BACKWARD,
               ]
    intervals = list(np.arange(1, 10, 0.2))

    bot = DroneBot(bot_name="bot-train-3", actions=actions, intervals=intervals, logger=logger)

    steps = random.choice(list(range(5, 10)))
    bot.start_flight_in_zig_zag_with_breaks(steps, break_time=5)


def main():
    bots = [
        # activate_bot_simplified_1,
        # activate_bot_simplified_2,
        # activate_bot_simplified_3,
        # activate_bot_simplified_4,
        # activate_bot_expert_1,
        # activate_bot_expert_2,
        # activate_bot_expert_3,
        activate_bot_train_1
    ]

    logger = create_general_logger("bots_recording")

    while True:
        bot = random.choice(bots)
        bot(logger=logger)


if __name__ == "__main__":
    main()
