import logging
import os.path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
from artificial_gps.settings import MODELS_FOLDER_PATH

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
                       forward_path_csv_path: str,
                       replay_memory_file_name: str = None,
                       load_replay_memory: bool = False,
                       update_replay_memory: bool = False,
                       load_last_model: bool = False,
                       training_name: str = "online_train",
                       logger: logging.Logger = logging.getLogger("dummy")) -> None:
    if load_replay_memory and is_replay_memory_file_exist(replay_memory_file_name):
        replay_memory = load_replay_memory_from_file(replay_memory_file_name)
    else:
        replay_memory = DQNReplayMemory()

    env = AirSimDroneEnvironment(drone_name=drone_name,
                                 forward_path_csv_path=forward_path_csv_path,
                                 logger=logger)
    dqn_algo = DQNAlgorithm()

    if load_last_model:
        q_model_path = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_q")

        dqn_algo.q_model = load_model(q_model_path)
        dqn_algo.target_q_model.set_weights(dqn_algo.q_model.get_weights())

    return_home_agent = ReturnHomeActor(forward_path_csv_path)

    ep_reward_list = []
    avg_reward_list = []

    epsilon = 1
    for ep in range(total_episodes):
        return_home_agent.reset_forwarding_info()
        prev_observation = env.reset()
        prev_state = return_home_agent.observation_to_normalized_state(prev_observation)

        episodic_reward = 0
        is_done = False

        step = 0
        while not is_done and step < max_episode_steps:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

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
                if len(replay_memory) > batch_size * 2:
                    logger.debug(f"Updates actor and critic policies based on DQN Algorithm "
                                 f"(data amount: {len(replay_memory)})")
                    prev_states, action_types, rewards, next_states, is_done_tensor = replay_memory.sample(batch_size)
                    dqn_algo.update_q_wights((prev_states, action_types, rewards, next_states, is_done_tensor))
                prev_state = state

            if len(replay_memory) % 50 == 0:
                dqn_algo.update_target()

            step += 1
        ep_reward_list.append(episodic_reward)

        if update_replay_memory:
            save_replay_memory_to_file(replay_memory_file_name, replay_memory)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        # saves models
        q_model_path = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_q")
        save_model(dqn_algo.q_model, q_model_path)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()


def train_ddpg_offline(replay_memory_file_name: str,
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

    ddpg_algo = DDPGAlgorithm()

    for ep in range(total_epochs):
        print(f"epoch: {ep} out of {total_epochs}")
        for batch_data in replay_memory.get_batches(batch_size, shuffle=True):
            prev_states, action_types, action_durations, rewards, next_states = batch_data
            ddpg_algo.update_actor_critic_weights((prev_states, action_types,
                                                   action_durations, rewards, next_states))

    # saves models
    actor_model_folder = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_actor")
    critic_model_folder = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_critic")

    save_model(ddpg_algo.actor_model, actor_model_folder)
    save_model(ddpg_algo.critic_model, critic_model_folder)