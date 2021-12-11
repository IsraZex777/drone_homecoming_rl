import os
import numpy as np
import pandas as pd
from pprint import pprint
from json import loads

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import (
    layers,
    initializers
)

from flight_recording import (
    INPUT_DATA_COLUMNS,
    TIMESTAMP_INPUT_COLUMNS
)

from .utils import print_exec_time

from .settings import (
    INPUT_SEQUENCE_LEN,
    OUTPUT_DATA_COLUMNS,
    GLOBAL_DATA_FOLDER_PATH
)


def convert_timestamp_to_interval(flight_input_df: pd.DataFrame):
    # Converts the start time to time interval
    next_time_df = flight_input_df[TIMESTAMP_INPUT_COLUMNS].shift(-1)
    time_diff_df = next_time_df - flight_input_df[TIMESTAMP_INPUT_COLUMNS]
    flight_input_df.loc[:, TIMESTAMP_INPUT_COLUMNS] = time_diff_df
    return flight_input_df


def convert_location_to_step(flight_output_df: pd.DataFrame):
    next_coordinates_df = flight_output_df.shift(-1)
    coordinate_diff = flight_output_df - next_coordinates_df

    return coordinate_diff


@print_exec_time
def load_preprocessed_sequences():
    all_csv_files = os.listdir(GLOBAL_DATA_FOLDER_PATH)

    flight_data_x = []
    flight_data_y = []
    for csv_name in all_csv_files:
        if not csv_name.endswith("csv"):
            continue

        csv_path = os.path.join(GLOBAL_DATA_FOLDER_PATH, csv_name)
        flight_df = pd.read_excel(csv_path)

        x_df = flight_df[INPUT_DATA_COLUMNS].copy()
        x_df = convert_timestamp_to_interval(x_df)

        y_df = flight_df[OUTPUT_DATA_COLUMNS].copy()
        y_df = convert_location_to_step(y_df)

        # Drops the last record because the process is based of difference
        x_df.drop(x_df.tail(1).index, inplace=True)
        y_df.drop(y_df.tail(1).index, inplace=True)

        flight_data_x.append(x_df.to_numpy())
        flight_data_y.append(y_df.to_numpy())

    data_x = np.concatenate(flight_data_x)
    data_y = np.concatenate(flight_data_y)
    #
    # standard_scaler = StandardScaler()
    # data_y = standard_scaler.fit_transform(data_y)

    return data_x, data_y


def split_data(data: np.array):

    """
    Splits data into train, dev and test
    :return:
    """
    data_len = len(data)

    train, dev, test = np.split(data, [int(.8 * data_len), int(.9 * data_len)])

    return train, dev, test


def load_preprocessed_dataset():
    flight_data_x, flight_data_y = load_preprocessed_sequences()

    train_x, dev_x, test_x = split_data(flight_data_x)
    train_y, dev_y, test_y = split_data(flight_data_y)

    return train_x, train_y, dev_x, dev_y, test_x, test_y
