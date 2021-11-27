from functools import wraps
from keras.models import model_from_json
from time import time


def print_exec_time(func):
    """
    Decorator that prints the execution time of a function
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        output = func(*args, **kwargs)
        end_time = time()
        print(f"Called {func.__name__}. Ran {'%.2f' % (end_time - start_time)} seconds")
        return output

    return wrapper


def save_model(model, model_name):
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"{model_name}.h5")


def load_model(model_name):
    with open(f'{model_name}.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{model_name}.h5")

    return loaded_model
