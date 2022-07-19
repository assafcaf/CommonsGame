import os
import random
import numpy as np
import tensorflow as tf
from HLC.utils.memory import ReplayMemory, ReplayBuffer, ReplayBuffer2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from HLC.networks.dqn_network import DQNNetwork, build_dense_model


class Agent:
    """
    Class for DQN model architecture.
    """
    def __init__(self, input_shape, num_actions, minibatch_size=128, capacity=10000, lr=1e-4,
                 replay_start_size=10000, agent_directory="", discount_factor=.99, tau=0.0001):

        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.num_actions = num_actions
        self.agent_directory = agent_directory
        self.replay_start_size = replay_start_size
        self.tau = tau

        # memory
        # self.memory = ReplayBuffer(input_shape=input_shape, capacity=capacity, minibatch_size=minibatch_size)
        self.memory = ReplayBuffer2(buffer_size=capacity, batch_size=minibatch_size)

        # agent networks
        self.main_network = DQNNetwork(input_shape, num_actions)
        self.target_network = DQNNetwork(input_shape, num_actions)
        self.update_target_network()
        self.optimizer = Adam(learning_rate=lr, epsilon=1e-6)
        self.loss = tf.keras.losses.Huber()

        # create directory to save agent weights
        if not os.path.isdir(self.agent_directory):
            os.mkdir(self.agent_directory)

    def get_action(self, state, exploration_rate):
        """Get action by ε-greedy method.

        Args:
            state (np.uint8): recent self.agent_history_length frames. (Default: (84, 84, 4))
            exploration_rate (int): Exploration rate for deciding random or optimal action.

        Returns:
            action (tf.int32): Action index
        """
        if random.random() < exploration_rate:
            action = np.random.choice(self.num_actions)
        else:
            recent_state = tf.expand_dims(state, axis=0)
            q_value = self.main_network(tf.cast(recent_state, tf.float32)).numpy()
            action = q_value.argmax()
        return action

    def get_action_sm(self, state):
        """Get action by ε-greedy method.

        Args:
            state (np.uint8): recent self.agent_history_length frames. (Default: (84, 84, 4))
        Returns:
            action (tf.int32): Action index
        """

        recent_state = tf.expand_dims(state, axis=0)
        q_value = self.main_network.call(tf.cast(recent_state, tf.float32))
        probs = tf.nn.softmax(q_value).numpy().ravel()
        action = np.random.choice(self.num_actions, p=probs)
        return action

    def learn(self):
        """Update main q network by experience replay method.
        Returns:
            loss (tf.float32): MSE loss of temporal difference.
        """
        # indices = self.memory.get_minibatch_indices()
        states, actions, rewards, next_states, terminal = self.memory.generate_minibatch_samples()

        with tf.GradientTape() as tape:
            next_state_max_q = tf.math.reduce_max(self.target_network(next_states), axis=1)
            expected_q = rewards + self.discount_factor * next_state_max_q * (1 - terminal)
            main_q = self.main_network(states)
            actual_q = tf.reduce_sum(main_q * tf.one_hot(actions, self.num_actions, 1.0, 0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), actual_q)

        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))

        certainty = tf.reduce_mean(tf.reduce_max(tf.nn.softmax(main_q), axis=1)).numpy()
        self.soft_update_target_network()
        return loss, tf.math.reduce_mean(main_q), tf.reduce_mean(certainty)

    def soft_update_target_network(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
              target_model (PyTorch model): weights will be copied to
              tau (float): interpolation parameter
        """
        main_w = self.main_network.get_weights()
        target_w = self.target_network.get_weights()
        for i in range(len(main_w)):
            target_w[i] = self.tau * main_w[i] + (1 - self.tau) * target_w[i]
        self.target_network.set_weights(target_w)

    def update_target_network(self):
        """Synchronize weights of target network by those of main network."""
        self.target_network.set_weights(self.main_network.trainable_variables)

    def remember(self, observation, action, reward, observation_next, done):
        self.memory.push(observation, action, reward, observation_next, done)

    def save_weights(self, ep):
        self.main_network.save_weights(os.path.join(self.agent_directory, f"episode_{ep}", ""))

    def load_weights(self, path):
        self.main_network.load_weights(path)
        self.target_network = tf.keras.models.clone_model(self.main_network)

