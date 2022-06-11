import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from position_prediction.settings import (
    TIMESTAMP_COLUMN,
    ORIENTATION_SENSORS,
    DATA_FOLDER_PATH,
    BAROMETER_SENSORS,
    ORIENTATION_DIFF_COLUMNS,
    INPUT_CLOUMNS,
    INPUT_1_CLOUMNS,
    INPUT_2_CLOUMNS,
    INPUT_3_CLOUMNS,
    OUTPUT_COLUMNS
)

from .utils import (
    split_data,
    columns_names_to_indexes
)


def recording_to_x_data(flight_df, input_columns: list):
    x_df = flight_df[input_columns].copy()

    # Barometer data smoothing
    x_df[BAROMETER_SENSORS] = x_df[BAROMETER_SENSORS].ewm(span=20, adjust=False).mean()

    # Creating orientation diff
    next_orientaion_df = x_df[ORIENTATION_SENSORS].shift(-1)
    orientaion_diff = (next_orientaion_df - x_df[ORIENTATION_SENSORS])
    orientaion_diff.rename(columns={key: f"{key}_diff" for key in orientaion_diff.columns}, inplace=True)

    # Creating timestamp diff
    timestamp_columns = [TIMESTAMP_COLUMN]
    next_time_df = x_df[timestamp_columns].shift(-1)
    time_diff_df = (next_time_df - x_df[timestamp_columns]) / 1_000_000_000
    x_df.loc[:, timestamp_columns] = time_diff_df

    # Appanding orientation diff
    x_df = pd.concat([x_df, orientaion_diff], axis=1)

    # Drops the last record because the process is based of difference
    x_df.drop(x_df.tail(1).index, inplace=True)

    return x_df


def recording_to_data(flight_df, input_columns: list, output_columns: list):
    x_df = recording_to_x_data(flight_df, input_columns, output_columns)

    y_df = flight_df[output_columns].copy()

    # Creating X, Y position diff
    columns = ["position_x", "position_y"]
    next_position_df = y_df[columns].shift(-1)
    position_diff = y_df[columns] - next_position_df
    y_df.loc[:, columns] = position_diff

    # Drops the last record because the process is based of difference
    y_df.drop(y_df.tail(1).index, inplace=True)

    return x_df, y_df


def load_flight_steps_from_file(csv_name: str, input_columns: list, output_columns: list):
    """

    @param csv_name:
    @param input_columns:
    @param output_columns:
    @return:
    """
    if not csv_name.endswith("csv"):
        raise ValueError(f"File with unsupported extension, expected csv (file: {csv_name})")

    csv_path = os.path.join(DATA_FOLDER_PATH, csv_name)
    flight_df = pd.read_csv(csv_path)

    return recording_to_data(flight_df, input_columns, output_columns)


def load_dataset(input_columns: list, output_columns: list):
    """
    Loads flight steps and orders it to sequences of sequence_length length.
    In order to feed it to rnn/lstm s2s model

    @param input_columns: The input columns
    @param output_columns: The outputs columns
    @return:
    """
    all_csv_files = os.listdir(DATA_FOLDER_PATH)
    all_csv_files.remove("bot-train-3_23Apr_17:24_record.csv")

    # x, y data from all flight sessions
    x_sessions = []
    y_sessions = []

    for csv_name in all_csv_files:
        try:
            x_df, y_df = load_flight_steps_from_file(csv_name, input_columns, output_columns)

            x_sessions.append(x_df)
            y_sessions.append(y_df)

        except ValueError as error:
            print(str(error))

    x_df = pd.concat(x_sessions, ignore_index=True)
    y_df = pd.concat(y_sessions, ignore_index=True)

    return x_df, y_df


def preprocess_ann_dataset(x_df: pd.DataFrame, y_df: pd.DataFrame):
    scaler_y = MinMaxScaler()
    scaler_x = MinMaxScaler((-1, 1))
    data_x = scaler_x.fit_transform(x_df)

    data_y = y_df.to_numpy()
    data_y = scaler_y.fit_transform(data_y)

    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.float32)
    return data_x, data_y, scaler_x, scaler_y


def load_preprocessed_dataset():
    """
    Loads the whole dataset with preprocessing

    @return: Loaded, preprocessed, shuffled, splitted data set
    """
    x_data, y_data = load_dataset(INPUT_CLOUMNS, OUTPUT_COLUMNS)

    x_data, y_data, scaler_x, scaler_y = preprocess_ann_dataset(x_data, y_data)

    input_columns_temp = INPUT_CLOUMNS[:]
    input_columns_temp.extend(ORIENTATION_DIFF_COLUMNS)

    pred_speed_input_indexes = columns_names_to_indexes(INPUT_1_CLOUMNS, input_columns_temp)
    pred_x_y_pos_input_indexes = columns_names_to_indexes(INPUT_2_CLOUMNS, input_columns_temp)
    pred_z_pos_input_indexes = columns_names_to_indexes(INPUT_3_CLOUMNS, input_columns_temp)

    train_x_1, dev_x_1, test_x_1 = split_data(x_data[:, pred_speed_input_indexes], train_per=0.88, dev_per=0.12)
    train_x_2, dev_x_2, test_x_2 = split_data(x_data[:, pred_x_y_pos_input_indexes], train_per=0.88, dev_per=0.12)
    train_x_3, dev_x_3, test_x_3 = split_data(x_data[:, pred_z_pos_input_indexes], train_per=0.88, dev_per=0.12)

    train_x = (train_x_1, train_x_2, train_x_3)
    dev_x = (dev_x_1, dev_x_2, dev_x_3)

    train_y, dev_y, test_y = split_data(y_data, train_per=0.88, dev_per=0.12)

    return train_x, train_y, dev_x, dev_y, scaler_x, scaler_y
