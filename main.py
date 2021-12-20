import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from artificial_gps.data import load_preprocessed_sequences
from artificial_gps.experiment1 import train_static_model
from artificial_gps.utils import save_model_with_scalers_binary
from artificial_gps.data import load_preprocessed_sequences

if __name__ == "__main__":
    # output = load_preprocessed_sequences()
    # print(output[0][1000])
    # print(output[1][1000])
    # output = train_static_model()
    # x_df, y_df = load_preprocessed_sequences()
    # plt.hist(y_df.loc[:, "gps_altitude"], bins=10000)
    # plt.show()
    model, scaler_x, scaler_y = train_static_model()
    save_model_with_scalers_binary(model, scaler_x, scaler_y, "1st")
