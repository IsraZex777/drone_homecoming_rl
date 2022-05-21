import logging
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


def start_training(drone_name: str,
                   forward_path_csv_path: str,
                   logger: logging.Logger = logging.getLogger("dummy")) -> None:
    ou_noise = OUActionNoise(mean=np.array([2]), std_deviation=float(.5) * np.ones(1))
    replay_memory = ReplayMemory()
    env = AirSimDroneEnvironment(drone_name=drone_name,
                                 forward_path_csv_path=forward_path_csv_path,
                                 logger=logger)
    ddpg_algo = DDPGAlgorithm()
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

                logger.info(f"Train report(epoch: {ep}) \n"
                            f"prev_state: {prev_state.numpy()} \n"
                            f"action_type: {DroneActions(tf.math.argmax(action_type_vector[0]).numpy()).name} \n"
                            f"action_duration: {action_duration} \n"
                            f"reward: {reward: .3f} \n"
                            f"state: {state.numpy()} \n"
                            f"is_done: {is_done} \n")

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

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
