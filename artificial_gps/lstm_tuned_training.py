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
    INPUT_SEQUENCE_LENGTH,
    TUNERS_FOLDER_PATH
)
from .utils import save_model_with_scalers_binary


def create_model(hp: kt.HyperParameters) -> Model:
    """
    Creates dl model
    :param hp: Hyper parameters (Initialized by a tuning algorithm)
    :return:
    """
    model = Sequential()

    lstm_layers = hp.Int("lstm_layers_amount", min_value=1, max_value=3)
    dense_layers = hp.Int("dense_layers_amount", min_value=0, max_value=2)
    dense_activation = hp.Choice("dense_activation", ["sigmoid", "relu", "sigmoid", "relu", "tanh"])
    learning_rate = hp.Choice("adam_learning_rate", [0.1, 0.2, 0.05, 1e-2, 1e-3, 5e-3, 5e-4])

    for layer_index in range(1, lstm_layers + 1):
        units = hp.Int(f"lstm_{layer_index}_units", min_value=16, max_value=256, step=16)

        lstm_params = {}
        if layer_index == 1:  # Is first Layer
            lstm_params["input_shape"] = (hp.get("sequence_length"), hp.get("input_columns_amount"))
        if layer_index < lstm_layers:  # Is not last later
            lstm_params["return_sequences"] = True

        model.add(layers.LSTM(units, **lstm_params))

        if hp.Boolean(f"lstm_{layer_index}_dropout"):
            dropout_rate = hp.Float(f"lstm_{layer_index}_dropout_rate", min_value=0, max_value=0.4, step=0.05)
            model.add(layers.Dropout(dropout_rate))

    for layer_index in range(1, dense_layers + 1):
        units = hp.Int(f"dense_{layer_index}_units", min_value=16, max_value=256, step=16)
        model.add(layers.Dense(units, activation=dense_activation))

        if hp.Boolean(f"dense_{layer_index}_dropout"):
            dropout_rate = hp.Float(f"dense_{layer_index}_dropout_rate", min_value=0, max_value=0.4, step=0.05)
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(hp.get("output_columns_amount")))

    if hp.Boolean(f"learning_rate_decay"):
        learning_rate = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.9)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(learning_rate=learning_rate))

    return model


def create_model_tuner():
    hp = kt.HyperParameters()
    hp.Fixed("input_columns_amount", len(INPUT_SEQUENCE_COLUMNS))
    hp.Fixed("output_columns_amount", len(OUTPUT_SEQUENCE_COLUMNS))
    hp.Fixed("sequence_length", INPUT_SEQUENCE_LENGTH)

    tuner = kt.RandomSearch(
        create_model,
        hyperparameters=hp,
        tune_new_entries=True,
        objective="val_loss",
        max_trials=200,
        overwrite=False,
        directory=TUNERS_FOLDER_PATH,
        project_name="artificial_gps_hp_tuner_lstm_27Dec_night",
        executions_per_trial=1,
    )

    return tuner


def train_save_tuned_lstm_model():
    train_x, train_y, dev_x, dev_y, test_x, test_y, scaler_x, scaler_y = \
        load_preprocessed_rnn_dataset(INPUT_SEQUENCE_COLUMNS, OUTPUT_SEQUENCE_COLUMNS, INPUT_SEQUENCE_LENGTH)

    tuner = create_model_tuner()

    #
    tuner.search(train_x,
                 train_y,
                 epochs=20,
                 batch_size=256,
                 validation_data=(dev_x, dev_y))

    # best_models_amount = 10
    # for index, model in enumerate(tuner.get_best_models(num_models=best_models_amount)):
    #     save_model_with_scalers_binary(model, scaler_x, scaler_y, f"lstm_26Dec_{index}")


def get_best_model():
    tuner = create_model_tuner()
    model = tuner.get_best_models()[0]
    return model
