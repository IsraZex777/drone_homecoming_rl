import os

import pandas as pd

from flight_recording import record_flight_for_seconds
from project_logging import create_general_logger
from rl_ddpg.ddpg_training import (
    start_ddpg_training,
    train_ddpg_offline
)
from rl_dqn import (
    start_dqn_training,
    train_dqn_offline
)

from rl_global.constants import RL_FORWARD_PATHS

from position_prediction import (
    train_model
)

from position_prediction import (
    PositionPredictor,
    load_model_with_scalers_binary
)

from position_prediction.settings import (
    DATA_FOLDER_PATH,
    INPUT_CLOUMNS
)


def record_data():
    record_flight_for_seconds(30, f"forward_path-simple_3")
    print("finished")


def force_dl_run_on_cpu():
    """
    This function forces the algorithm run on cpu instead on gpu because the airsim loads the gpu very much
    @return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main_train_offline():
    logger = create_general_logger("ddpg_rl_train_offline")
    force_dl_run_on_cpu()
    replay_memory_name = "2022_06_07_memory_0034_record"
    training_name = "2022_06_11_1336"
    train_dqn_offline(replay_memory_file_name=replay_memory_name,
                      training_name=training_name,
                      logger=logger)


def main_train_online():
    logger = create_general_logger("ddpg_rl_train")

    force_dl_run_on_cpu()
    files = [os.path.join(RL_FORWARD_PATHS, csv) for csv in os.listdir(RL_FORWARD_PATHS)]

    replay_memory_name = "2022_06_11_memory_2237_record"
    # training_name = "2022_06_05_1617"
    training_name = "2022_06_11_1336"
    # pos_prediction_model_name = "ann_pos_11Jun_1510"
    pos_prediction_model_name = ""
    start_dqn_training(drone_name="drone1",
                       load_replay_memory=False,
                       forward_path_csv_files=files,
                       update_replay_memory=False,
                       replay_memory_file_name=replay_memory_name,
                       load_last_model=False,
                       training_name=training_name,
                       is_training=False,
                       pos_prediction_model_name=pos_prediction_model_name,
                       logger=logger)


def main_train_position_prediction_model():
    model, history = train_model()


def main_predict_position():
    model_name = "ann_pos_11Jun_1510"
    model, scaler_x, scaler_y = load_model_with_scalers_binary(model_name)
    predictor = PositionPredictor(model, scaler_x, scaler_y)

    file_name = "bot-train-3_23Apr_17:24_record.csv"
    file_path = os.path.join(DATA_FOLDER_PATH, file_name)
    data_x = pd.read_csv(file_path)
    print(predictor.predict_position_offset(data_x))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # main_predict_position()
    # main_train_position_prediction_model()
    # record_data()
    # main_train_offline()
    main_train_online()
    # record_data()
    # bots_main()
