from return_home_simulation import run_drone_return_to_home_simulation


if __name__ == "__main__":
    model_name = "ann_pos_11Jun_next_generation"
    # prediction_model_path = os.path.join(MODELS_FOLDER_PATH, model_name)
    # print(prediction_model_path)
    # model, scaler_x, scaler_y = load_model_with_scalers_binary(prediction_model_path)
    # pos_predictor = PositionPredictor(model, scaler_x, scaler_y)

    # forward_path_name = "/home/israzex/Desktop/drone_homecoming_rl/rl_global/rl_forward_paths/forward_path-turn_left_forward_record.csv"
    # return_home_actor = ReturnHomeActor(forward_path_name, model_name)

    q_model_name = "rl_2022_06_11_1336_q"
    # q_model = load_model(os.path.join(MODELS_FOLDER_PATH, q_model_name))
    # run_drone_actor(pos_predictor, return_home_actor, q_model)

    run_drone_return_to_home_simulation(model_name, q_model_name)
