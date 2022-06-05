import os
from flight_recording import record_flight_for_seconds
from project_logging import create_general_logger
from rl_ddpg.ddpg_training import (
    start_ddpg_training,
    train_ddpg_offline
)
from rl_dqn import start_dqn_training
from flight_recording.bots import bots_main

from rl_global.constants import RL_FORWARD_PATHS


def record_data():
    record_flight_for_seconds(30, f"forward_path-turn_left_forward")
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
    replay_memory_name = "2022_06_03_memory_7"
    training_name = "2022_05_04_2204"
    train_ddpg_offline(replay_memory_file_name=replay_memory_name,
                       training_name=training_name,
                       logger=logger)


def main_train_online():
    logger = create_general_logger("ddpg_rl_train")

    force_dl_run_on_cpu()
    files = [os.path.join(RL_FORWARD_PATHS, csv) for csv in os.listdir(RL_FORWARD_PATHS)]

    replay_memory_name = "2022_06_05_memory_1617"
    training_name = "2022_06_05_1617"
    start_dqn_training(drone_name="drone1",
                       load_replay_memory=True,
                       forward_path_csv_files=files,
                       update_replay_memory=True,
                       replay_memory_file_name=replay_memory_name,
                       load_last_model=False,
                       training_name=training_name,
                       logger=logger)


if __name__ == "__main__":
    # record_data()
    # main_train_offline()
    main_train_online()
    # record_data()
    # bots_main()
