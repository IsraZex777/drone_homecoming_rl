import random
import time
import airsim
import datetime
import threading
from .flight_recording import (
    FlightRecorder,
)

from .settings import (
    MIN_X_POS,
    MAX_X_POS,
    MIN_Y_POS,
    MAX_Y_POS,
    MIN_Z_POS,
    MAX_Z_POS,
    MIN_VELOCITY,
    MAX_VELOCITY,
    MIN_TIMEOUT,
    MAX_TIMEOUT,
    RUN_SECONDS
)


def _reset_session(client):
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)

    z = - (MAX_Z_POS + MIN_Z_POS) / 2
    velocity = random.randrange(MIN_VELOCITY, MAX_VELOCITY)

    client.moveToZAsync(z, velocity).join()


def generate_and_save_flight_data(flight_name: str = f"flight_{datetime.datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}"):
    flight_recorder = FlightRecorder(flight_name)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    iteration = 1

    flight_recorder.start_flight_recording()
    _reset_session(client)

    t_end = time.time() + RUN_SECONDS
    t_start = time.time()
    while time.time() < t_end:
        x_pos = random.randrange(MIN_X_POS, MAX_X_POS)
        y_pos = random.randrange(MIN_Y_POS, MAX_Y_POS)
        z_pos = -random.randrange(MIN_Z_POS, MAX_Z_POS)
        velocity = random.randrange(MIN_VELOCITY, MAX_VELOCITY)
        timeout = random.randrange(MIN_TIMEOUT, MAX_TIMEOUT)

        t_offset = int((time.time() - t_start))
        print(f"{iteration} ({t_offset} seconds out of {RUN_SECONDS}):\n"
              f"x_pos:{x_pos}\n"
              f"y_pos:{y_pos}\n"
              f"z_pos:{z_pos}\n"
              f"velocity:{velocity}\n"
              f"timeout:{timeout}\n\n")
        client.moveToPositionAsync(x_pos, y_pos, z_pos, velocity, timeout).join()

        # In collision reset session
        state = client.getMultirotorState()
        # If under the minimum height (the height is negative)
        if -state.kinematics_estimated.position.z_val < MIN_Z_POS:
            if flight_recorder.is_recording_running():
                flight_recorder.stop_and_save_recording_data()
            _reset_session(client)
            flight_recorder.start_flight_recording()

        iteration += 1

    if flight_recorder.is_recording_running():
        flight_recorder.stop_and_save_recording_data()
    client.armDisarm(False)
    client.enableApiControl(False)


def generate_and_save_flight_intervals(
        flight_name: str = f"recording_{datetime.datetime.now().strftime('%d%b_%H:%M')}",
        min_recording_length: int = 35):
    flight_recorder = FlightRecorder(flight_name)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    iteration = 1

    flight_recorder.start_flight_recording()
    _reset_session(client)

    t_end = time.time() + RUN_SECONDS
    t_start = time.time()
    while time.time() < t_end:
        destinations = random.randrange(1, 3)
        t_offset = int((time.time() - t_start))
        print(f"{iteration} ({t_offset} seconds out of {RUN_SECONDS}):\n"
              f"{destinations} intervals")

        for index in range(destinations):
            x_pos = random.randrange(MIN_X_POS, MAX_X_POS)
            y_pos = random.randrange(MIN_Y_POS, MAX_Y_POS)
            z_pos = -random.randrange(MIN_Z_POS, MAX_Z_POS)
            velocity = random.randrange(MIN_VELOCITY, MAX_VELOCITY)
            print(f"{(time.time() - t_start - t_offset)} - x_pos: {x_pos}, y_pos: {y_pos}, z_pos: {z_pos}, "
                  f"velocity: {velocity}\n")

            # Move until stop, so each recording will start from velocity 0
            client.moveToPositionAsync(x_pos, y_pos, z_pos, velocity).join()
            time.sleep(1)

        print(f"time: {time.time() - t_start - t_offset}")
        # If drone arrived to the destination in less than min_
        # recording_length seconds, wait in order to make equal
        # length recordings
        while (time.time() - t_start - t_offset) < destinations * min_recording_length:
            time.sleep(5)

        # Save and start new recording
        if flight_recorder.is_recording_running():
            flight_recorder.stop_and_save_recording_data()

        # In collision reset session
        state = client.getMultirotorState()
        # If under the minimum height (the height is negative)
        if -state.kinematics_estimated.position.z_val < MIN_Z_POS:
            _reset_session(client)
        # every 15 records reset the flight
        elif iteration % 15 == 0:
            _reset_session(client)

        flight_recorder.start_flight_recording()
        time.sleep(.5)

        iteration += 1

    if flight_recorder.is_recording_running():
        flight_recorder.stop_and_save_recording_data()
    client.armDisarm(False)
    client.enableApiControl(False)
