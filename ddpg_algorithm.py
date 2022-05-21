import tensorflow as tf
from typing import Tuple

from actor import create_actor_model
from critic import create_critic_model


class DDPGAlgorithm:
    def __init__(self, gamma: float = 0.95,
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.002):
        self.gamma = gamma

        self.actor_model = create_actor_model()
        self.target_actor = create_actor_model()
        self.critic_model = create_critic_model()
        self.target_critic = create_critic_model()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(actor_lr)

    def _update_critic_weights(self, transition_batch: tuple) -> None:
        """
        Updates Critic weights
        @param transition_batch:
        @return:
        """
        state_batch, action_batch, reward_batch, next_state_batch = transition_batch

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

    def _update_actor_weights(self, transition_batch: tuple) -> None:
        """

        @param transition_batch:
        @return:
        """
        state_batch, action_batch, reward_batch, next_state_batch = transition_batch

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def update_actor_critic_weights(self, transition_batch: tuple) -> None:
        """
        Updates actor and critic weights as defined by the DDPG Algorithm

        @param transition_batch: Input transition memory to train upon
        @return: None
        """
        self._update_critic_weights(transition_batch)
        self._update_actor_weights(transition_batch)
