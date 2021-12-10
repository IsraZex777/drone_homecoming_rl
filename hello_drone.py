import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

from time import sleep

if __name__ == "__main__":
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    #
    state = client.getMultirotorState()
    # print(state)
    # print(dir(state))
    # print(type(state))
    # print(dict(state))
    # print(type(dict(state)))
    s = pprint.pformat(state)
    print("state: %s" % s)

    # state = client.getRotorStates()
    # s = pprint.pformat(state)
    # print("getRotorStates: %s" % s)
    # #
    # imu_data = client.getImuData()
    # s = pprint.pformat(imu_data)
    # print("imu_data: %s" % s)

    barometer_data = client.getDistanceSensorData()
    s = pprint.pformat(barometer_data)
    print("distance: %s" % s)

    # barometer_data = client.getBarometerData()
    # s = pprint.pformat(barometer_data)
    # print("barometer_data: %s" % s)
    #
    # magnetometer_data = client.getMagnetometerData()
    # s = pprint.pformat(magnetometer_data)
    # print("magnetometer_data: %s" % s)
    #
    # gps_data = client.getGpsData()
    # s = pprint.pformat(gps_data)
    # print("gps_data: %s" % s)

    airsim.wait_key('Press any key to takeoff')
    print("Taking off...")
    client.armDisarm(True)
    client.takeoffAsync().join()

    airsim.wait_key('Press any key to reset to original state')

    client.reset()
    client.armDisarm(False)

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)
