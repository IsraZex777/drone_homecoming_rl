import numpy as np

from .data import load_sequences_from_file
from .utils import load_model_with_scalers_binary
from .settings import (
    INPUT_DATA_COLUMNS
)
from flight_recording.settings import (
    TIMESTAMP_INPUT_COLUMNS,
    MAIN_TIMESTAMP_COLUMN
)


def test_model_predictions(model_name: str, data_csv_name: str):
    flight_x_df, flight_y_df = load_sequences_from_file(data_csv_name)
    data_x = flight_x_df.to_numpy()
    real_y = flight_y_df.to_numpy()

    try:
        model, scaler_x, scaler_y = load_model_with_scalers_binary(model_name)
    except FileNotFoundError:
        print(f"There is no model in name: {model_name}")
        return

    time_intervals = flight_x_df[MAIN_TIMESTAMP_COLUMN].to_numpy().reshape(-1, 1)
    data_x = scaler_x.transform(data_x)

    predicted_y = model.predict(data_x)
    predicted_y = scaler_y.inverse_transform(predicted_y)

    time_offset = np.add.accumulate(time_intervals)
    predicted_offset = np.add.accumulate(predicted_y)
    real_offset = np.add.accumulate(real_y)

    time_stamps_to_print = list(range(10))
    time_stamps_to_print.extend([15, 20, 25, 30, 40, 50, 100])
    index = 0

    for predicted, real, time_offset, real_tmp in zip(predicted_offset, real_offset, time_offset, real_y):
        if index < len(time_stamps_to_print) and \
                int(time_offset) == time_stamps_to_print[index]:
            print(
                f"position:\nreal offset:      {real}\npredicted offset: {predicted}\ntime offset(seconds): {time_offset}\n\n")
            index += 1
    #
    # time_intervals = flight_data_x[:, time_interval_index]
    # print(time_intervals)
