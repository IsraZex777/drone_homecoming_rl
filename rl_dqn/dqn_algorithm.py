import tensorflow as tf

from .q_model import create_q_model
from rl_global.constants import action_type_amount


class DQNAlgorithm:
    def __init__(self, gamma: float = 0.95,
                 q_model_lr: float = 0.002):
        self.gamma = gamma

        self.q_model = create_q_model()
        self.target_q_model = create_q_model()

        self.q_optimizer = tf.keras.optimizers.Adam(q_model_lr)
        self.loss_function = tf.keras.losses.Huber()

    def update_q_wights(self, transition_batch: tuple) -> None:
        """
        Updates Critic weights
        @param transition_batch:
        @return:
        """
        state_batch, action_types, reward_batch, next_state_batch, done_batch = transition_batch

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = self.target_q_model.predict(next_state_batch)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = reward_batch + self.gamma * tf.reshape(tf.reduce_max(future_rewards, axis=1), (-1, 1))

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_batch) - done_batch
        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_types, action_type_amount)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.q_model(state_batch)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))

    def update_target(self):
        self.target_q_model.set_weights(self.q_model.get_weights())
