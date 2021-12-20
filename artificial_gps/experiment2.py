import keras
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



class MovementLayer(layers.Layer):
    pass


@print_exec_time
def create_model() -> Model:
    """
    Creates dl model
    :param hp: Hyper parameters (Initialized by a tuning algorithm)
    :return:
    """
    model = Sequential()

    model.add(layers.Input(len(INPUT_DATA_COLUMNS)))
    model.add(layers.Dense(1, activation="relu"))
    model.add(layers.Dense(len(OUTPUT_DATA_COLUMNS)))

    optimizer = optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


@print_exec_time
def train_static_model():
    train_x, train_y, dev_x, dev_y, test_x, test_y = load_preprocessed_dataset()

    model = create_model()

    model.fit(train_x,
              train_y,
              epochs=30,
              batch_size=64,
              validation_data=(dev_x, dev_y))


if __name__ == "__main__":
    # main()
    pass


"""


Hyperparameter     |Best Value So Far 
dense_layer_amount |3                 
dense_activation   |sigmoid           
dense_1_units      |96                        
adam_learning_rate |0.001                         
dense_2_units      |224                         
dense_3_units      |256          

5.7440 * e^-5                           
"""