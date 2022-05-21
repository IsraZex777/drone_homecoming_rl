import os
from flight_recording import record_flight_for_seconds
from project_logging import create_general_logger
from ddpg_training import start_training

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


def main():
    logger = create_general_logger("ddpg_rl_train")

    force_dl_run_on_cpu()
    forward_path_csv_name = "forward_path-simple_forward_record.csv"
    forward_path_csv_path = os.path.join(RECORDS_FOLDER, forward_path_csv_name)
    start_training(drone_name="drone1",
                   forward_path_csv_path=forward_path_csv_path,
                   logger=logger)



if __name__ == "__main__":
    main()
