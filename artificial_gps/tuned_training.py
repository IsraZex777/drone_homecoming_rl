import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import (
    layers,
    Model,
    Sequential,
    optimizers
)

from flight_recording import (
    INPUT_DATA_COLUMNS
)

from .data import (
    load_preprocessed_dataset,
)
from .utils import (
    print_exec_time,
    save_model,
)

from .settings import (
    INPUT_SEQUENCE_LEN,
    OUTPUT_DATA_COLUMNS
)


@print_exec_time
def create_model_dynamic(hp: kt.HyperParameters) -> Model:
    """
    Creates dl model
    :param hp: Hyper parameters (Initialized by a tuning algorithm)
    :return:
    """
    model = Sequential()
    dense_units_1 = hp.Int("dense_units_1", min_value=64, max_value=256, step=16)
    dense_dropout_1 = hp.Float("dense_dropout_1", min_value=0, max_value=0.4, step=0.05)
    dense_units_2 = hp.Int("dense_units_2", min_value=64, max_value=256, step=16)
    dense_dropout_2 = hp.Float("dense_dropout_2", min_value=0, max_value=0.4, step=0.05)
    dense_units_3 = hp.Int("dense_units_3", min_value=64, max_value=256, step=16)
    dense_dropout_3 = hp.Float("dense_dropout_3", min_value=0, max_value=0.4, step=0.05)
    dense_units_4 = hp.Int("dense_units_3", min_value=64, max_value=256, step=16)
    dense_dropout_4 = hp.Float("dense_dropout_3", min_value=0, max_value=0.4, step=0.05)
    dense_units_5 = hp.Int("dense_units_3", min_value=64, max_value=256, step=16)
    dense_dropout_5 = hp.Float("dense_dropout_3", min_value=0, max_value=0.4, step=0.05)
    dense_activation = hp.Choice("dense_activation", ["sigmoid", "relu", "tanh"])

    model.add(layers.Input(len(INPUT_DATA_COLUMNS)))
    model.add(layers.Dense(dense_units_1, activation=dense_activation))
    model.add(layers.Dropout(dense_dropout_1))
    model.add(layers.Dense(dense_units_2, activation=dense_activation))
    model.add(layers.Dropout(dense_dropout_2))
    model.add(layers.Dense(dense_units_3, activation=dense_activation))
    model.add(layers.Dropout(dense_dropout_3))
    model.add(layers.Dense(dense_units_4, activation=dense_activation))
    model.add(layers.Dropout(dense_dropout_4))
    model.add(layers.Dense(dense_units_5, activation=dense_activation))
    model.add(layers.Dropout(dense_dropout_5))
    model.add(layers.Dense(len(OUTPUT_DATA_COLUMNS)))

    adam_learning_rate = hp.Choice("adam_learning_rate", [0.01, 0.001])
    optimizer = optimizers.Adam(learning_rate=adam_learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model_tuner():
    hp = kt.HyperParameters()

    tuner = kt.RandomSearch(
        create_model_dynamic,
        hyperparameters=hp,
        tune_new_entries=True,
        objective="val_loss",
        max_trials=10,
        overwrite=True,
        directory="./",
        project_name="artificial_gps_hp_tuner"
    )

    return tuner


@print_exec_time
def train_tuned_model():
    train_x, train_y, dev_x, dev_y, test_x, test_y = load_preprocessed_dataset()

    tuner = create_model_tuner()

    tuner.search(train_x,
                 train_y,
                 epochs=30,
                 batch_size=64,
                 validation_data=(dev_x, dev_y))


if __name__ == "__main__":
    # main()
    pass
