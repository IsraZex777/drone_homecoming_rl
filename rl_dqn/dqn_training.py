import logging
import random
import os.path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import List

from .dqn_algorithm import DQNAlgorithm
from .actor import make_actor_action
from rl_global import ReturnHomeActor

from rl_global.airsim_gym import AirSimDroneEnvironment
from rl_global.constants import (
    total_episodes,
    total_epochs,
    batch_size
)
from .dqn_replay_memory import DQNReplayMemory
from position_prediction.settings import MODELS_FOLDER_PATH

from rl_global.utils import (
    load_replay_memory_from_file,
    save_replay_memory_to_file,
    is_replay_memory_file_exist,
    save_model,
    load_model
)

from rl_global.constants import (
    epsilon_interval,
    epsilon_greedy_frames,
    epsilon_min,
    max_episode_steps
)

from drone_interface import DroneActions


def start_dqn_training(drone_name: str,
                       forward_path_csv_files: List[str],
                       replay_memory_file_name: str = None,
                       load_replay_memory: bool = False,
                       update_replay_memory: bool = False,
                       load_last_model: bool = False,
                       training_name: str = "online_train",
                       is_training=True,
                       logger: logging.Logger = logging.getLogger("dummy")) -> None:
    forward_paths_amount = len(forward_path_csv_files)
    forward_path_index = random.randint(0, forward_paths_amount - 1)

    if load_replay_memory and is_replay_memory_file_exist(replay_memory_file_name):
        replay_memory = load_replay_memory_from_file(replay_memory_file_name)
    else:
        replay_memory = DQNReplayMemory()

    env = AirSimDroneEnvironment(drone_name=drone_name,
                                 forward_path_csv_path=forward_path_csv_files[forward_path_index],
                                 logger=logger)
    dqn_algo = DQNAlgorithm()

    if load_last_model:
        q_model_path = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_q")

        dqn_algo.q_model = load_model(q_model_path)
        dqn_algo.target_q_model.set_weights(dqn_algo.q_model.get_weights())

    return_home_agent = ReturnHomeActor(forward_path_csv_files[forward_path_index])

    avg_reward_list = []

    if is_training:
        epsilon = 1
    else:
        epsilon = 0

    for ep in range(total_episodes):
        forward_path_index = random.randint(0, forward_paths_amount - 1)
        logger.info(f"Loads following forward path: {forward_path_csv_files[forward_path_index]}")
        return_home_agent.reset_forwarding_info(forward_path_csv_files[forward_path_index])
        prev_observation = env.reset(forward_path_csv_files[forward_path_index])
        prev_state = return_home_agent.observation_to_normalized_state(prev_observation)

        episodic_reward = 0
        is_done = False

        step = 0
        while not is_done and step < max_episode_steps:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            if is_training:
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

            action = make_actor_action(dqn_algo.q_model, tf_prev_state, epsilon=epsilon, logger=logger)

            observation, reward, is_done, info = env.step(action)

            state = return_home_agent.observation_to_normalized_state(observation)

            samples_amount = len(replay_memory.memory)
            log = (f"Train episode: {ep} ({samples_amount} samples), "
                   f"epsilon: {epsilon}, "
                   f"a_type: {action[0].name}, "
                   f"a_duration: {action[1] :.1f}), "
                   f"reward: {reward: .3f}, "
                   f"is_done: {is_done} \n "
                   f"q_values: {[f'{value:.4f}' for value in dqn_algo.q_model(tf_prev_state).numpy()[0]]}"
                   f"({DroneActions(tf.argmax(dqn_algo.q_model(tf_prev_state), axis=1))})\n "
                   f"prev_state: {[f'{value:.4f}' for value in prev_state.numpy()]}\n "
                   f"state     : {[f'{value:.4f}' for value in state.numpy()]} ")
            logger.info(log)
            print(log)

            replay_memory.push(prev_state, action[0].value, reward, state, is_done)

            episodic_reward += reward
            if is_done:
                logger.info(f"Epoch learn terminated because the following reason: {info['reason']}")
            else:
                if len(replay_memory) > batch_size * 4:
                    logger.debug(f"Updates actor and critic policies based on DQN Algorithm "
                                 f"(data amount: {len(replay_memory)})")
                    prev_states, action_types, rewards, next_states, is_done_tensor = replay_memory.sample(batch_size)
                    dqn_algo.update_q_wights((prev_states, action_types, rewards, next_states, is_done_tensor))
                prev_state = state

            if len(replay_memory) % 50 == 0:
                dqn_algo.update_target()

            step += 1

        avg_reward_list.append(episodic_reward / step)

        if update_replay_memory:
            save_replay_memory_to_file(replay_memory_file_name, replay_memory)

        logger.info(f"Episode {ep} has finished. Avg reward: {episodic_reward / step}")
        logger.info(f"Total Avg reward list: {avg_reward_list}")

        # saves models
        q_model_path = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_q")
        save_model(dqn_algo.q_model, q_model_path)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()


def train_dqn_offline(replay_memory_file_name: str,
                      training_name: str,
                      logger: logging.Logger = logging.getLogger("dummy")) -> None:
    """
    Trains actor and critic models based on previously collected replay memory.
    Without online interactive with the environment
    @param replay_memory_file_name:
    @param training_name:
    @param logger:
    @return:
    """
    replay_memory = load_replay_memory_from_file(replay_memory_file_name)

    dqn_algo = DQNAlgorithm()

    for ep in range(total_epochs):
        print(f"epoch: {ep} out of {total_epochs}")
        epoch_loss = 0
        for index, batch_data in enumerate(replay_memory.get_batches(batch_size, shuffle=True)):
            prev_states, action_types, action_durations, rewards, next_states = batch_data
            loss = dqn_algo.update_q_wights((prev_states, action_types, action_durations, rewards, next_states))

            epoch_loss += loss

            if index % 10 == 0:
                dqn_algo.update_target()

        print(f"epoch {ep} ag loss: {epoch_loss / (len(replay_memory) / batch_size)}")
    # saves models
    model_path = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_q")

    save_model(dqn_algo.q_model, model_path)
