import os
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from flight_recording import (
    TIMESTAMP_COLUMNS
)

from .utils import print_exec_time

from .settings import (
    DATA_FOLDER_PATH,
    INPUT_SEQUENCE_LENGTH
)


def _convert_timestamp_to_interval_seconds(flight_input_df: pd.DataFrame, timestamp_columns: list):
    """
    Converts the timestamp fields into the amount of seconds between each two timestamps

    Note: each timestamp represents the amount eof NANO seconds (1,000,000,000 nanoseconds = 1 seconds)
    """
    # Converts the start time to time interval
    next_time_df = flight_input_df[timestamp_columns].shift(-1)
    time_diff_df = (next_time_df - flight_input_df[timestamp_columns]) / 1_000_000_000
    flight_input_df.loc[:, timestamp_columns] = time_diff_df
    return flight_input_df


def _convert_location_to_step(flight_output_df: pd.DataFrame):
    next_coordinates_df = flight_output_df.shift(-1)
    coordinate_diff = flight_output_df - next_coordinates_df

    return coordinate_diff


def load_flight_steps_from_file(csv_name: str, input_columns: list, output_columns: list):
    """

    @param csv_name:
    @param input_columns:
    @param output_columns:
    @return:
    """
    if not csv_name.endswith("csv"):
        return ValueError(f"File with unsupported extension, expected csv (file: {csv_name})")

    csv_path = os.path.join(DATA_FOLDER_PATH, csv_name)
    flight_df = pd.read_csv(csv_path)

    x_df = flight_df[input_columns].copy()
    timestamp_columns = [column for column in input_columns if column in TIMESTAMP_COLUMNS]
    x_df = _convert_timestamp_to_interval_seconds(x_df, timestamp_columns)

    y_df = flight_df[output_columns].copy()
    y_df = _convert_location_to_step(y_df)

    # Drops the last record because the process is based of difference
    x_df.drop(x_df.tail(1).index, inplace=True)
    y_df.drop(y_df.tail(1).index, inplace=True)

    return x_df, y_df


def load_flight_steps(input_columns: list, output_columns: list):
    """
    Loads flight recording data from all files in the "data" folder.
    After loading the data it calculates the steps (time differences (timestamps in x) and location steps (y data))
    @param input_columns:
    @param output_columns:
    @return:
    """
    all_csv_files = os.listdir(DATA_FOLDER_PATH)

    x_dfs = []
    y_dfs = []
    for csv_name in all_csv_files:
        try:
            x_df, y_df = load_flight_steps_from_file(csv_name, input_columns, output_columns)
            x_dfs.append(x_df)
            y_dfs.append(y_df)
        except ValueError as error:
            print(str(error))

    flight_data_x_df = pd.concat(x_dfs, ignore_index=True)
    flight_data_y_df = pd.concat(y_dfs, ignore_index=True)
    return flight_data_x_df, flight_data_y_df


def load_preprocessed_flight_steps(input_columns: list, output_columns: list):
    data_x_df, data_y_df = load_flight_steps(input_columns, output_columns)

    # Removes outliers from y values
    valid_indexes = (np.abs(stats.zscore(data_y_df)) < 3).all(axis=1)
    data_y_df = data_y_df[valid_indexes]
    data_x_df = data_x_df[valid_indexes]

    data_x = data_x_df.to_numpy()
    data_y = data_y_df.to_numpy()

    scaler_x = MinMaxScaler()
    # scaler_x = StandardScaler()
    scaler_y = MinMaxScaler()
    # scaler_y = StandardScaler()

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


def load_preprocessed_dataset(input_columns: list, output_columns: list):
    """
    Loads the whole dataset with preprocessing

    @param input_columns: Input columns names
    @param output_columns: Output columns names
    @return: Loaded, preprocessed, shuffled, splitted data set
    """
    flight_data_x, flight_data_y, scaler_x, scaler_y = load_preprocessed_flight_steps(input_columns, output_columns)

    flight_data_x, flight_data_y = shuffle_data_set(flight_data_x, flight_data_y)

    train_x, dev_x, test_x = split_data(flight_data_x)
    train_y, dev_y, test_y = split_data(flight_data_y)

    return train_x, train_y, dev_x, dev_y, test_x, test_y, scaler_x, scaler_y


def load_dataset(input_columns: list, output_columns: list):
    """
    Loads the whole dataset without preprocessing

    @param input_columns: Input columns names
    @param output_columns: Output columns names
    @return: Loaded, shuffled, splitted data set
    """
    flight_data_x, flight_data_y = load_flight_steps(input_columns, output_columns)

    flight_data_x, flight_data_y = shuffle_data_set(flight_data_x, flight_data_y)

    train_x, dev_x, test_x = split_data(flight_data_x)
    train_y, dev_y, test_y = split_data(flight_data_y)

    return train_x, train_y, dev_x, dev_y, test_x, test_y


def load_preprocessed_flight_sequences(input_columns: list, output_columns: list, sequence_length: int):
    """
    Loads flight steps and orders it to sequences of sequence_length length.
    In order to feed it to rnn/lstm model

    @param input_columns: The input columns
    @param output_columns: The outputs columns
    @param sequence_length: Target sequence length
    @return:
    """
    all_csv_files = os.listdir(DATA_FOLDER_PATH)

    # x, y data from all flight sessions
    x_sessions = []
    y_sessions = []

    # The data feed to the rnn model
    sequences_x = []
    sequences_y = []

    for csv_name in all_csv_files:
        try:
            x_df, y_df = load_flight_steps_from_file(csv_name, input_columns, output_columns)

            x_sessions.append(x_df.to_numpy())
            y_sessions.append(y_df.to_numpy())

        except ValueError as error:
            print(str(error))

    all_x_data = np.concatenate(x_sessions)
    all_y_data = np.concatenate(y_sessions)

    # creating normalizers
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    #
    # scaler_x = StandardScaler()
    # scaler_y = StandardScaler()

    scaler_x.fit(all_x_data)
    scaler_y.fit(all_y_data)

    for session_data_x, session_data_y in zip(x_sessions, y_sessions):
        normalized_data_x = scaler_x.transform(session_data_x)
        normalized_data_y = scaler_y.transform(session_data_y)
        steps_amount = normalized_data_x.shape[0]

        # Splits the data into data sequences
        for offset in range(steps_amount - sequence_length):
            sequences_x.append(normalized_data_x[offset: offset + sequence_length, :])
            sequences_y.append(np.add.accumulate(normalized_data_y[offset: offset + sequence_length, :])[-1])

    sequences_x = np.stack(sequences_x)
    sequences_y = np.stack(sequences_y)
    return sequences_x, sequences_y, scaler_x, scaler_y


def load_preprocessed_rnn_dataset(input_columns: list, output_columns: list, sequence_length: int):
    """
    Loads the whole dataset with preprocessing

    @param input_columns: Input columns names
    @param output_columns: Output columns names
    @param sequence_length: The size of input x sequence
    @return: Loaded, preprocessed, shuffled, splitted data set
    """
    flight_data_x, flight_data_y, scaler_x, scaler_y = load_preprocessed_flight_sequences(input_columns, output_columns,
                                                                                          sequence_length)

    flight_data_x, flight_data_y = shuffle_data_set(flight_data_x, flight_data_y)

    train_x, dev_x, test_x = split_data(flight_data_x)
    train_y, dev_y, test_y = split_data(flight_data_y)

    return train_x, train_y, dev_x, dev_y, test_x, test_y, scaler_x, scaler_y
