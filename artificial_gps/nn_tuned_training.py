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
    load_preprocessed_dataset,
)
from .utils import (
    print_exec_time,
    save_model,
)

from .settings import (
    INPUT_SEQUENCE_LEN,
    OUTPUT_DATA_COLUMNS,
    INPUT_DATA_COLUMNS
)


@print_exec_time
def create_model_dynamic(hp: kt.HyperParameters) -> Model:
    """
    Creates dl model
    :param hp: Hyper parameters (Initialized by a tuning algorithm)
    :return:
    """
    model = Sequential()
    dense_layer_amount = hp.Int("dense_layer_amount", min_value=1, max_value=4, step=1)
    # dense_activation = hp.Choice("dense_activation", ["sigmoid", "relu", "tanh"])

    model.add(layers.Input(len(INPUT_DATA_COLUMNS)))
    for layer_index in range(1, dense_layer_amount + 1):
        units = hp.Int(f"dense_{layer_index}_units", min_value=64, max_value=256, step=32)

        model.add(layers.Dense(units, activation="sigmoid"))

        # if hp.Boolean(f"dense_{layer_index}_dropout"):
        #     dropout_rate = hp.Float(f"dense_{layer_index}_dropout_rate", min_value=0, max_value=0.4, step=0.05)
        #     model.add(layers.Dropout(dropout_rate))

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
        max_trials=1000,
        overwrite=True,
        directory="./",
        project_name="artificial_gps_hp_tuner"
    )

    return tuner


@print_exec_time
def train_tuned_model():
    train_x, train_y, dev_x, dev_y, test_x, test_y, scaler_x, scaler_y = load_preprocessed_dataset()

    tuner = create_model_tuner()

    tuner.search(train_x,
                 train_y,
                 epochs=30,
                 batch_size=64,
                 validation_data=(dev_x, dev_y))


if __name__ == "__main__":
    # main()
    pass
