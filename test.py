from flight_recording import (
    generate_and_save_flight_data,
    record_flight_for_seconds
)
from artificial_gps.experiment1 import train_static_model
from artificial_gps.data import load_preprocessed_sequences

if __name__ == "__main__":
    pass
    # x, y = load_preprocessed_sequences()
    # generate_and_save_flight_data()
    record_flight_for_seconds(30)
#