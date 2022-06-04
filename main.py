import os
from flight_recording import record_flight_for_seconds
from project_logging import create_general_logger
from rl_ddpg.ddpg_training import (
    start_training,
    train_offline
)

from flight_recording.bots import bots_main
from flight_recording.settings import RECORDS_FOLDER


def record_data():
    record_flight_for_seconds(30, f"forward_path-forward_and_turn_right")
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
    replay_memory_name = "2022_06_03_memory_5"
    training_name = "2022_05_04_1458"
    train_offline(replay_memory_file_name=replay_memory_name,
                  training_name=training_name,
                  logger=logger)


def main_train_online():
    logger = create_general_logger("ddpg_rl_train")

    force_dl_run_on_cpu()
    forward_path_csv_name = "forward_path-simple_forward_record.csv"
    forward_path_csv_path = os.path.join(RECORDS_FOLDER, forward_path_csv_name)

    replay_memory_name = "2022_06_03_memory_6"
    training_name = "2022_05_04_1458"
    start_training(drone_name="drone1",
                   load_replay_memory=True,
                   update_replay_memory=True,
                   replay_memory_file_name=replay_memory_name,
                   forward_path_csv_path=forward_path_csv_path,
                   load_last_model=True,
                   training_name=training_name,
                   logger=logger)


if __name__ == "__main__":
    # record_data()
    # main_train_offline()
    main_train_online()
    # bots_main()