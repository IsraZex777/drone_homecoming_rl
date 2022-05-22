import logging
import os.path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ddpg_algorithm import DDPGAlgorithm
from ou_action_noice import OUActionNoise
from actor_model import (
    make_actor_action
)
from return_home_actor import ReturnHomeActor

from airsim_gym import AirSimDroneEnvironment
from constants import (
    total_episodes,
    batch_size
)
from drone_controller import DroneActions
from replay_memory import ReplayMemory
from artificial_gps.settings import MODELS_FOLDER_PATH


def save_model(model, model_name):
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"{model_name}.h5")


def load_model(model_name):
    with open(f'{model_name}.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{model_name}.h5")

    return loaded_model


def start_training(drone_name: str,
                   forward_path_csv_path: str,
                   load_last_model: bool = False,
                   logger: logging.Logger = logging.getLogger("dummy")) -> None:
    ou_noise = OUActionNoise(mean=np.array([2]), std_deviation=float(.5) * np.ones(1))
    replay_memory = ReplayMemory()
    env = AirSimDroneEnvironment(drone_name=drone_name,
                                 forward_path_csv_path=forward_path_csv_path,
                                 logger=logger)
    ddpg_algo = DDPGAlgorithm()

    if load_last_model:
        actor_model_folder = os.path.join(MODELS_FOLDER_PATH, "rl_actor_model")
        critic_model_folder = os.path.join(MODELS_FOLDER_PATH, "rl_critic_model")

        ddpg_algo.actor_model = load_model(actor_model_folder)
        ddpg_algo.critic_model = load_model(critic_model_folder)

        ddpg_algo.target_actor.set_weights(ddpg_algo.actor_model.get_weights())
        ddpg_algo.target_critic.set_weights(ddpg_algo.critic_model.get_weights())

    return_home_agent = ReturnHomeActor(forward_path_csv_path)

    ep_reward_list = []
    avg_reward_list = []

    for ep in range(total_episodes):
        return_home_agent.reset_forwarding_info()
        prev_observation = env.reset()
        prev_state = return_home_agent.observation_to_state(prev_observation)

        episodic_reward = 0
        is_done = False

        while not is_done:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action_type_vector, action_duration = make_actor_action(ddpg_algo.actor_model,
                                                                    tf_prev_state,
                                                                    ou_noise,
                                                                    logger=logger)
            action = action_type_vector, action_duration

            observation, reward, is_done, info = env.step(action)

            if is_done:
                logger.info(f"Epoch learn terminated because the following reason: {info['reason']}")
            else:
                state = return_home_agent.observation_to_state(observation)

                log = (f"Train episode: {ep}, "
                       f"action_type: {DroneActions(tf.math.argmax(action_type_vector[0]).numpy()).name}, "
                       f"action_duration: {action_duration}, "
                       f"reward: {reward: .3f}, "
                       f"is_done: {is_done}, "
                       f"prev_state: {prev_state.numpy()}, "
                       f"state: {state.numpy()}, "                       )
                logger.info(log)
                print(log)

                replay_memory.push(prev_state, action_type_vector, action_duration, reward, state)

                episodic_reward += reward

                if len(replay_memory) > batch_size * 2:
                    logger.debug(f"Updates actor and critic policies based on DDPG Algorithm "
                                 f"(data amount: {len(replay_memory)})")
                    prev_states, action_types, action_durations, rewards, next_states = replay_memory.sample(batch_size)
                    ddpg_algo.update_actor_critic_weights((prev_states, action_types,
                                                           action_durations, rewards, next_states))

                prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        # saves models
        actor_model_folder = os.path.join(MODELS_FOLDER_PATH, "rl_actor_model")
        critic_model_folder = os.path.join(MODELS_FOLDER_PATH, "rl_critic_model")

        save_model(ddpg_algo.actor_model, actor_model_folder)
        save_model(ddpg_algo.critic_model, critic_model_folder)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
