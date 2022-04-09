import numpy as np

from .data import load_flight_steps_from_file
from .utils import load_model_with_scalers_binary
from .settings import (
    INPUT_DATA_COLUMNS
)
from flight_recording.settings import (
    TIMESTAMP_COLUMNS,
    MAIN_TIMESTAMP_COLUMN
)


def test_model_predictions(model_name: str, data_csv_name: str, input_columns: list, output_columns: list):
    flight_x_df, flight_y_df = load_flight_steps_from_file(data_csv_name, input_columns, output_columns)
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

    for predicted, real, time_offset in zip(predicted_offset, real_offset, time_offset):
        if index < len(time_stamps_to_print) and \
                int(time_offset) == time_stamps_to_print[index]:
            print(
                f"position:\nreal offset:      {real}\npredicted offset: {predicted}\ntime offset(seconds): {time_offset}\n\n")
            index += 1

    return predicted_offset, real_offset, time_offset


def test_lstm_model_predictions(model_name: str,
                                data_csv_name: str,
                                input_columns: list,
                                output_columns: list,
                                sequence_length: int):
    flight_x_df, flight_y_df = load_flight_steps_from_file(data_csv_name, input_columns, output_columns)
    data_x = flight_x_df.to_numpy()
    real_y = flight_y_df.to_numpy()

    try:
        model, scaler_x, scaler_y = load_model_with_scalers_binary(model_name)
    except FileNotFoundError:
        print(f"There is no model in name: {model_name}")
        return

    normalized_data_x = scaler_x.transform(data_x)
    steps_amount = normalized_data_x.shape[0]

    sequences_x = []
    real_offset_y = []

    # Splits the data into data sequences
    for offset in range(steps_amount - sequence_length):
        sequences_x.append(normalized_data_x[offset: offset + sequence_length, :])
        real_offset_y.append(np.add.accumulate(real_y[offset: offset + sequence_length, :])[-1])

    sequences_x = np.stack(sequences_x)
    real_offset_y = np.stack(real_offset_y)

    time_intervals = flight_x_df[MAIN_TIMESTAMP_COLUMN].to_numpy().reshape(-1, 1)[sequence_length:]
    time_offset = np.add.accumulate(time_intervals)

    predicted_offset_y = model.predict(sequences_x)
    # predicted_offset_y = scaler_y.inverse_transform(predicted_offset_y)

    for i in range(2 * sequence_length - 1, len(time_intervals)):
        real_offset_y[i] += real_offset_y[i - sequence_length]
        predicted_offset_y[i] += predicted_offset_y[i - sequence_length]

    time_stamps_to_print = list(range(10))
    time_stamps_to_print.extend([15, 20, 25, 30, 40, 50, 100])
    index = 0

    for predicted, real, time_offset in zip(predicted_offset_y, real_offset_y, time_offset):
        if index < len(time_stamps_to_print) and \
                int(time_offset) == time_stamps_to_print[index]:
            print(
                f"position:\nreal offset:      {real}\npredicted offset: {predicted}\ntime offset(seconds): {time_offset}\n\n")
            index += 1
    #
    return predicted_offset_y, real_offset_y, time_offset
