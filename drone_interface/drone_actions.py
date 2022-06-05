from enum import Enum, unique


@unique
class DroneActions(Enum):
    FORWARD = 0
    BACKWARD = 5
    TURN_LEFT = 1
    TURN_RIGHT = 2
    UP = 3
    DOWN = 4
    STOP = 6
    # SPEED_LEVEL_1 = 6
    # SPEED_LEVEL_2 = 6
    # SPEED_LEVEL_3 = 6
