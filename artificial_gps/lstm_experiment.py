import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import (
    layers,
    Model,
    Sequential,
    optimizers
)

from .data import (
    load_preprocessed_rnn_dataset
)
from .settings import (
    INPUT_SEQUENCE_COLUMNS,
    OUTPUT_SEQUENCE_COLUMNS,
    INPUT_SEQUENCE_LENGTH
)


def create_model(input_columns_amount: int, output_columns_amount: int, sequence_length: int) -> Model:
    """
    Creates LSTM model
    @param input_columns_amount:
    @param output_columns_amount:
    @param sequence_length:
    @return:
    """
    model = Sequential()

    # lstm_params = {}
    # if layer_index == 1:  # Is first Layer
    #     lstm_params["input_shape"] = (hp.get("sentence_max_length"), hp.get("embedding_dim"))
    # if layer_index < lstm_layers:  # Is not last later
    #     lstm_params["return_sequences"] = True

    model.add(layers.LSTM(128, input_shape=(sequence_length, input_columns_amount)))

    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(output_columns_amount))

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(learning_rate=0.01))

    return model


def train_lstm_model():
    train_x, train_y, dev_x, dev_y, test_x, test_y, scaler_x, scaler_y = \
        load_preprocessed_rnn_dataset(INPUT_SEQUENCE_COLUMNS, OUTPUT_SEQUENCE_COLUMNS, INPUT_SEQUENCE_LENGTH)

    model = create_model(len(INPUT_SEQUENCE_COLUMNS), len(OUTPUT_SEQUENCE_COLUMNS), INPUT_SEQUENCE_LENGTH)

    # 🥰
    model.fit(train_x,
              train_y,
              epochs=40,
              batch_size=128,
              validation_data=(dev_x, dev_y))

    return model, scaler_x, scaler_y
