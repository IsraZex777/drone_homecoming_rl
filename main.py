import datetime
from flight_recording import record_flight_for_seconds
from bots import DroneBot

if __name__ == "__main__":
    record_flight_for_seconds(1000, f"manual_train_{datetime.datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}")