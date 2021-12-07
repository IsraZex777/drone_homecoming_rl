import random
import time
import airsim
import datetime
from .flight_recording import create_flight_recording

POSITION_NUM = 3
RUN_HOURS = 0
RUN_MINUTES = RUN_HOURS * 60
RUN_SECONDS = RUN_MINUTES * 60 + 30


def _reset_session(client):
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)

    z = -10
    client.moveToZAsync(z, 10).join()


@create_flight_recording(f"flight_{datetime.datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}")
def generate_and_save_flight_data():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    iteration = 1

    z = -10
    client.moveToZAsync(z, 10).join()

    t_end = time.time() + RUN_SECONDS
    while time.time() < t_end:
        x_pos = random.randrange(-1000, 1000)
        y_pos = random.randrange(-1000, 1000)
        z_pos = random.randrange(-200, -30)
        velocity = random.randrange(1, 80)
        timeout = random.randrange(5, 20)

        print(f"{iteration}:\nx_pos:{x_pos}\ny_pos:{y_pos}\nz_pos:{z_pos}\nvelocity:{velocity}\ntimeout:{timeout}\n\n")
        client.moveToPositionAsync(x_pos, y_pos, z_pos, velocity, timeout).join()

        # In collision reset session
        state = client.getMultirotorState()
        if state.collision.has_collided:
            _reset_session(client)

        iteration += 1

    client.armDisarm(False)
    client.enableApiControl(False)
