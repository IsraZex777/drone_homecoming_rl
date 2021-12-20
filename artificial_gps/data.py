import os
import numpy as np
import pandas as pd
from scipy import stats
from pprint import pprint
from json import loads

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import (
    layers,
    initializers
)

from flight_recording import (
    TIMESTAMP_INPUT_COLUMNS
)

from .utils import print_exec_time

from .settings import (
    INPUT_SEQUENCE_LEN,
    INPUT_DATA_COLUMNS,
    OUTPUT_DATA_COLUMNS,
    GLOBAL_DATA_FOLDER_PATH
)


def convert_timestamp_to_interval_seconds(flight_input_df: pd.DataFrame):
    """
    Converts the timestamp fields into the amount of seconds between each two timestamps

    Note: each timestamp represents the amount eof NANO seconds (1,000,000,000 nanoseconds = 1 seconds)
    """
    # Converts the start time to time interval
    next_time_df = flight_input_df[TIMESTAMP_INPUT_COLUMNS].shift(-1)
    time_diff_df = (next_time_df - flight_input_df[TIMESTAMP_INPUT_COLUMNS]) / 1_000_000_000
    flight_input_df.loc[:, TIMESTAMP_INPUT_COLUMNS] = time_diff_df
    return flight_input_df


def convert_location_to_step(flight_output_df: pd.DataFrame):
    next_coordinates_df = flight_output_df.shift(-1)
    coordinate_diff = flight_output_df - next_coordinates_df

    return coordinate_diff


def load_sequences():
    all_csv_files = os.listdir(GLOBAL_DATA_FOLDER_PATH)
    flight_data_x_df = pd.DataFrame(columns=INPUT_DATA_COLUMNS)
    flight_data_y_df = pd.DataFrame(columns=OUTPUT_DATA_COLUMNS)
    for csv_name in all_csv_files:
        if not csv_name.endswith("csv"):
            continue

        csv_path = os.path.join(GLOBAL_DATA_FOLDER_PATH, csv_name)
        flight_df = pd.read_csv(csv_path)

        x_df = flight_df[INPUT_DATA_COLUMNS].copy()
        x_df = convert_timestamp_to_interval_seconds(x_df)

        y_df = flight_df[OUTPUT_DATA_COLUMNS].copy()
        y_df = convert_location_to_step(y_df)

        # Drops the last record because the process is based of difference
        x_df.drop(x_df.tail(1).index, inplace=True)
        y_df.drop(y_df.tail(1).index, inplace=True)

        flight_data_x_df = flight_data_x_df.append(x_df, ignore_index=True)
        flight_data_y_df = flight_data_y_df.append(y_df, ignore_index=True)

    # data_x = np.concatenate(flight_data_x)
    # data_y = np.concatenate(flight_data_y)

    return flight_data_x_df, flight_data_y_df


@print_exec_time
def load_preprocessed_sequences():
    data_x_df, data_y_df = load_sequences()

    # Removes outliers from y values
    valid_indexes = (np.abs(stats.zscore(data_y_df)) < 3).all(axis=1)
    data_y_df = data_y_df[valid_indexes]
    data_x_df = data_x_df[valid_indexes]

    data_x = data_x_df.to_numpy()
    data_y = data_y_df.to_numpy()

    # scaler_x = MinMaxScaler()
    scaler_x = StandardScaler()
    # scaler_y = MinMaxScaler()
    scaler_y = StandardScaler()

    data_x = scaler_x.fit_transform(data_x)
    data_y = scaler_y.fit_transform(data_y)

    return data_x, data_y, scaler_x, scaler_y


def split_data(data: np.array):
    """
    Splits data into train, dev and test
    :return:
    """
    data_len = len(data)

    train, dev, test = np.split(data, [int(.8 * data_len), int(.9 * data_len)])

    return train, dev, test


def shuffle_data_set(x: np.array, y: np.array):
    example_amount = x.shape[0]
    shuffle_indexes = np.random.permutation(example_amount)
    x = x[shuffle_indexes]
    y = y[shuffle_indexes]

    return x, y


def load_preprocessed_dataset():
    flight_data_x, flight_data_y, scaler_x, scaler_y = load_preprocessed_sequences()

    flight_data_x, flight_data_y = shuffle_data_set(flight_data_x, flight_data_y)

    train_x, dev_x, test_x = split_data(flight_data_x)
    train_y, dev_y, test_y = split_data(flight_data_y)

    return train_x, train_y, dev_x, dev_y, test_x, test_y, scaler_x, scaler_y


def load_dataset():
    flight_data_x, flight_data_y = load_sequences()

    flight_data_x, flight_data_y = shuffle_data_set(flight_data_x, flight_data_y)

    train_x, dev_x, test_x = split_data(flight_data_x)
    train_y, dev_y, test_y = split_data(flight_data_y)

    return train_x, train_y, dev_x, dev_y, test_x, test_y