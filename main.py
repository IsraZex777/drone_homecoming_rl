import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from artificial_gps.experiment1 import train_static_model
from artificial_gps.utils import save_model_with_scalers_binary
from artificial_gps.nn_tuned_training import train_tuned_model
from artificial_gps.model_tester import test_model_predictions, test_lstm_model_predictions
from artificial_gps.lstm_experiment import train_lstm_model
from artificial_gps.lstm_tuned_training import train_save_tuned_lstm_model, get_best_model

from artificial_gps.settings import (
    INPUT_DATA_COLUMNS,
    OUTPUT_DATA_COLUMNS,
    INPUT_SEQUENCE_COLUMNS,
    OUTPUT_SEQUENCE_COLUMNS,
    INPUT_SEQUENCE_LENGTH
)

if __name__ == "__main__":
    pass
    # model, scaler_x, scaler_y = train_static_model()
    # save_model_with_scalers_binary(model, scaler_x, scaler_y, "wihtout_velocity_min_max")
    # test_model_predictions("wihtout_velocity",
    #                        "flight_2021:12:21_20:05:53_1_record.csv",
    #                        INPUT_DATA_COLUMNS,
    #                        OUTPUT_DATA_COLUMNS)
    # train_the_best_model()

    # model, scaler_x, scaler_y = train_lstm_model()
    # save_model_with_scalers_binary(model, scaler_x, scaler_y, "LSTM_after_hahaaaa")
    #
    train_save_tuned_lstm_model()

    # trained_on = ["flight_2021:12:21_20:52:19_1_record.csv",
    #               "flight_2021:12:21_20:05:53_1_record.csv",
    #               "flight_2021:12:26_21:37:31_1_record.csv"]
    # new_data = ["flight_2021:12:21_19:38:37_1_record.csv"]
    # test_lstm_model_predictions("lstm_26Dec_0",
    #                             trained_on[0],
    #                             INPUT_SEQUENCE_COLUMNS,
    #                             OUTPUT_SEQUENCE_COLUMNS,
    #                             INPUT_SEQUENCE_LENGTH)
    # test_model_predictions("lstm1",
    #                        "flight_2021:12:21_18:25:50_1_record.csv",
    #                        INPUT_SEQUENCE_COLUMNS,
    #                        OUTPUT_SEQUENCE_COLUMNS)
    #
    # test_model_predictions("lstm1",
    #                        "flight_2021:12:21_18:25:50_1_record.csv",
    #                        INPUT_DATA_COLUMNS,
    #                        OUTPUT_DATA_COLUMNS)

    # train_tuned_model()
    # output = load_preprocessed_sequences()
    # print(output[0][1000])
    # print(output[1][1000])
    # output = train_static_model()
    # x_df, y_df = load_preprocessed_sequences()
    # plt.hist(y_df.loc[:, "gps_altitude"], bins=10000)
    # plt.show()
    # model, scaler_x, scaler_y = train_static_model()
    # save_model_with_scalers_binary(model, scaler_x, scaler_y, "3rd")
