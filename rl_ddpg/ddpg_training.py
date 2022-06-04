import logging
import os.path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .ddpg_algorithm import DDPGAlgorithm
from .ou_action_noice import OUActionNoise
from .actor_model import (
    make_actor_action,
    action_duration_to_real,
    action_type_to_real
)
from rl_global.return_home_actor import ReturnHomeActor

from .airsim_gym import AirSimDroneEnvironment
from rl_global.constants import (
    total_episodes,
    total_epochs,
    batch_size
)
from rl_global.replay_memory import ReplayMemory
from artificial_gps.settings import MODELS_FOLDER_PATH

from rl_global.utils import (
    load_replay_memory_from_file,
    save_replay_memory_to_file,
    is_replay_memory_file_exist,
)


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
                   replay_memory_file_name: str = None,
                   load_replay_memory: bool = False,
                   update_replay_memory: bool = False,
                   load_last_model: bool = False,
                   training_name: str = "online_train",
                   logger: logging.Logger = logging.getLogger("dummy")) -> None:
    ou_noise = OUActionNoise(mean=np.array([2]), std_deviation=float(.1) * np.ones(1))

    if load_replay_memory and is_replay_memory_file_exist(replay_memory_file_name):
        replay_memory = load_replay_memory_from_file(replay_memory_file_name)
    else:
        replay_memory = ReplayMemory()

    env = AirSimDroneEnvironment(drone_name=drone_name,
                                 forward_path_csv_path=forward_path_csv_path,
                                 logger=logger)
    ddpg_algo = DDPGAlgorithm()

    if load_last_model:
        actor_model_folder = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_actor")
        critic_model_folder = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_critic")

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
        prev_state = return_home_agent.observation_to_normalized_state(prev_observation)

        episodic_reward = 0
        is_done = False

        while not is_done:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action_type_vector, action_duration = make_actor_action(ddpg_algo.actor_model,
                                                                    tf_prev_state,
                                                                    ou_noise,
                                                                    logger=logger)

            real_action_type = action_type_to_real(action_type_vector)
            real_action_duration = action_duration_to_real(action_duration)
            action = real_action_type, real_action_duration

            observation, reward, is_done, info = env.step(action)

            state = return_home_agent.observation_to_normalized_state(observation)

            samples_amount = len(replay_memory.memory)
            log = (f"Train episode: {ep} ({samples_amount} samples), "
                   f"a_type: {real_action_type.name}, "
                   f"a_duration: {real_action_duration: .2f}({action_duration[0]: .4f}), "
                   f"reward: {reward: .3f}, "
                   f"is_done: {is_done} \n "
                   f"prev_state: {[f'{value:.4f}' for value in prev_state.numpy()]}\n "
                   f"state     : {[f'{value:.4f}' for value in state.numpy()]} ")
            logger.info(log)
            print(log)

            replay_memory.push(prev_state, action_type_vector, action_duration, reward, state)

            episodic_reward += reward
            if is_done:
                logger.info(f"Epoch learn terminated because the following reason: {info['reason']}")
            else:
                if len(replay_memory) > batch_size * 2:
                    logger.debug(f"Updates actor and critic policies based on DDPG Algorithm "
                                 f"(data amount: {len(replay_memory)})")
                    prev_states, action_types, action_durations, rewards, next_states = replay_memory.sample(batch_size)
                    ddpg_algo.update_actor_critic_weights((prev_states, action_types,
                                                           action_durations, rewards, next_states))

                prev_state = state

        ep_reward_list.append(episodic_reward)

        if update_replay_memory:
            save_replay_memory_to_file(replay_memory_file_name, replay_memory)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        # saves models
        actor_model_folder = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_actor")
        critic_model_folder = os.path.join(MODELS_FOLDER_PATH, f"rl_{training_name}_critic")

        save_model(ddpg_algo.actor_model, actor_model_folder)
        save_model(ddpg_algo.critic_model, critic_model_folder)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()


def train_offline(replay_memory_file_name: str,
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
    print(replay_memory)
    print(len(replay_memory))
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
