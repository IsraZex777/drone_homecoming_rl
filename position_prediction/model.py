import tensorflow as tf
from tensorflow.keras import (
    layers,
    Model
)

from position_prediction.settings import (
    INPUT_1_CLOUMNS,
    INPUT_2_CLOUMNS,
    INPUT_3_CLOUMNS,
)


def create_ann_model() -> Model:
    """
    Creates LSTM model
    @return:
    """
    pred_speed_input = layers.Input(shape=len(INPUT_1_CLOUMNS))
    pred_speed_hidden = layers.Dense(200, activation="relu")(pred_speed_input)
    pred_speed_hidden = layers.Dense(300, activation="relu")(pred_speed_hidden)
    pred_speed_hidden = layers.Dense(200, activation="relu")(pred_speed_hidden)
    pred_speed_ouput = layers.Dense(2, activation="linear")(pred_speed_hidden)

    pred_x_y_pos_input = layers.Input(shape=len(INPUT_2_CLOUMNS))
    pred_x_y_pos_hideen = layers.Dense(16, activation="relu")(pred_x_y_pos_input)

    combined = layers.Concatenate()([pred_speed_ouput, pred_x_y_pos_hideen])

    hidden = layers.Dense(16, activation="sigmoid")(combined)
    pos_x_y_out = layers.Dense(2, activation="linear")(hidden)

    b2z_pos_input = layers.Input(shape=len(INPUT_3_CLOUMNS))
    b2z_pos_hidden = layers.Dense(16, activation="tanh")(b2z_pos_input)
    z_pos_out = layers.Dense(1)(b2z_pos_hidden)

    combined_out = layers.Concatenate()([pos_x_y_out, z_pos_out])

    model = tf.keras.Model(inputs=[pred_speed_input, pred_x_y_pos_input, b2z_pos_input], outputs=combined_out,
                           name="position_predict")
    model.compile(loss='mean_squared_error',
                  optimizer="adam")

    return model
