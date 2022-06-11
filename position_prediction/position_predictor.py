import numpy as np

from position_prediction.settings import (
    INPUT_CLOUMNS,
    INPUT_1_CLOUMNS,
    INPUT_2_CLOUMNS,
    INPUT_3_CLOUMNS,
    ORIENTATION_DIFF_COLUMNS
)

from utils import (
    columns_names_to_indexes
)

from data_preprocessing import recording_to_x_data


class PositionPredictor:
    def __init__(self, model, scaler_x, scaler_y):
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def predict_position_offset(self, flight_recording_df):
        x_df = recording_to_x_data(flight_recording_df, INPUT_CLOUMNS)

        input_columns_temp = INPUT_CLOUMNS[:]
        input_columns_temp.extend(ORIENTATION_DIFF_COLUMNS)

        x_df = x_df[input_columns_temp]
        data_x = self.scaler_x.transform(x_df)

        pred_speed_input_indexes = columns_names_to_indexes(INPUT_1_CLOUMNS, input_columns_temp)
        pred_x_y_pos_input_indexes = columns_names_to_indexes(INPUT_2_CLOUMNS, input_columns_temp)
        pred_z_pos_input_indexes = columns_names_to_indexes(INPUT_3_CLOUMNS, input_columns_temp)

        data_x_1 = data_x[:, pred_speed_input_indexes]
        data_x_2 = data_x[:, pred_x_y_pos_input_indexes]
        data_x_3 = data_x[:, pred_z_pos_input_indexes]

        pred_y = self.model.predict([data_x_1, data_x_2, data_x_3])
        pred_xy_pos_step = self.scaler_y.inverse_transform(pred_y)

        pred_xy_pos = np.add.accumulate(pred_xy_pos_step[:, [0, 1]])

        pred_xy_pos = np.hstack([pred_xy_pos, pred_xy_pos_step[:, 2].reshape(-1, 1)])

        return -pred_xy_pos[-1, 0], -pred_xy_pos[-1, 1], pred_xy_pos[-1, 2] - pred_xy_pos[0, 2]
