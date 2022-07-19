from collections import namedtuple
import random
import tensorflow as tf
import numpy as np
from collections import deque, namedtuple
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminal"))


class ReplayMemoryTfCast:
    """
    This class manages memory of agent.
    """

    def __init__(self, capacity, state_shape=(84, 84), history_len=4, minibatch_size=32, verbose=True):
        self.capacity = int(capacity)
        self.history_len = int(history_len)
        self.minibatch_size = int(minibatch_size)
        self._memory = []
        self._index = 0
        self._full = False
        self.verbose = verbose

        if verbose:
            # state and next_state will use uint8 (8 bit = 1 Byte)
            # action uses int32 (32 bit = 4 Byte)
            # reward uses float32 (32 bit = 4 Byte)
            # terminal uses boolean (8 bit = 1 Byte (numpy))
            total_est_mem = np.float64(self.capacity * (np.prod(state_shape) * 4 * 2 + 4 + 4 + 1)) / 1024.0 ** 3
            print("Estimated memory usage ONLY for storing replays: {:.4f} GB".format(total_est_mem))

    def __len__(self):
        return len(self._memory)

    def __getitem__(self, key):
        return self._memory[key]

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.capacity)

    @property
    def cur_index(self):
        return self._index

    def is_full(self):
        return self._full

    def push(self, state, action, reward, next_state, terminal):

        trsn = Transition(state, action, reward, next_state, terminal)
        if len(self._memory) < self.capacity:
            self._memory.append(None)
        if self._index + 1 == self.capacity:
            self._full = True
        self._memory[self._index] = trsn
        self._index = (self._index + 1) % self.capacity

    def get_minibatch_indices(self):

        indices = []
        while len(indices) < self.minibatch_size:
            while True:
                if self.is_full():
                    index = np.random.randint(low=self.history_len, high=self.capacity, dtype=np.int32)
                else:
                    index = np.random.randint(low=self.history_len, high=self.cur_index, dtype=np.int32)

                if np.any([sample.terminal for sample in self._memory[index - self.history_len:index]]):
                    continue
                indices.append(index)
                break
        return indices

    def generate_minibatch_samples(self, indices):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = [], [], [], [], []

        for index in indices:
            selected_mem = self._memory[index]
            state_batch.append(tf.constant(selected_mem.state, tf.float32))
            action_batch.append(tf.constant(selected_mem.action, tf.int32))
            reward_batch.append(tf.constant(selected_mem.reward, tf.float32))
            next_state_batch.append(tf.constant(selected_mem.next_state, tf.float32))
            terminal_batch.append(tf.constant(selected_mem.terminal, tf.float32))

        return tf.stack(state_batch, axis=0), tf.stack(action_batch, axis=0), tf.stack(reward_batch, axis=0), tf.stack(
            next_state_batch, axis=0), tf.stack(terminal_batch, axis=0)


class ReplayMemory:
    """
    This class manages memory of agent.
    """

    def __init__(self, capacity, state_shape=(84, 84), history_len=4, minibatch_size=32, verbose=True):
        self.capacity = int(capacity)
        self.history_len = int(history_len)
        self.minibatch_size = int(minibatch_size)
        self._memory = []
        self._index = 0
        self._full = False
        self.verbose = verbose

        if verbose:
            # state and next_state will use uint8 (8 bit = 1 Byte)
            # action uses int32 (32 bit = 4 Byte)
            # reward uses float32 (32 bit = 4 Byte)
            # terminal uses boolean (8 bit = 1 Byte (numpy))
            total_est_mem = np.float64(self.capacity * (np.prod(state_shape) * 4 * 2 + 4 + 4 + 1)) / 1024.0 ** 3
            print("Estimated memory usage ONLY for storing replays: {:.4f} GB".format(total_est_mem))

    def __len__(self):
        return len(self._memory)

    def __getitem__(self, key):
        return self._memory[key]

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.capacity)

    @property
    def cur_index(self):
        return self._index

    def is_full(self):
        return self._full

    def push(self, state, action, reward, next_state, terminal):

        trsn = Transition(state, action, reward, next_state, terminal)
        if len(self._memory) < self.capacity:
            self._memory.append(None)
        if self._index + 1 == self.capacity:
            self._full = True
        self._memory[self._index] = trsn
        self._index = (self._index + 1) % self.capacity

    def get_minibatch_indices(self):

        indices = []
        while len(indices) < self.minibatch_size:
            while True:
                if self.is_full():
                    index = np.random.randint(low=self.history_len, high=self.capacity, dtype=np.int32)
                else:
                    index = np.random.randint(low=self.history_len, high=self.cur_index, dtype=np.int32)

                if np.any([sample.terminal for sample in self._memory[index - self.history_len:index]]):
                    continue
                indices.append(index)
                break
        return indices

    def generate_minibatch_samples(self, indices):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = [], [], [], [], []

        for index in indices:
            selected_mem = self._memory[index]
            state_batch.append(tf.constant(selected_mem.state, tf.float32))
            action_batch.append(tf.constant(selected_mem.action, tf.int32))
            reward_batch.append(tf.constant(selected_mem.reward, tf.float32))
            next_state_batch.append(tf.constant(selected_mem.next_state, tf.float32))
            terminal_batch.append(tf.constant(selected_mem.terminal, tf.float32))

        return tf.stack(state_batch, axis=0), tf.stack(action_batch, axis=0), tf.stack(reward_batch, axis=0), \
            tf.stack(next_state_batch, axis=0), tf.stack(terminal_batch, axis=0)


class ReplayBuffer:
    def __init__(self, input_shape, capacity, minibatch_size=32):
        self.index = 0
        self.minibatch_size = minibatch_size
        self.is_full = False
        self.capacity = capacity
        self.observations = np.zeros((capacity, *input_shape), dtype=np.int)
        self.next_observations = np.zeros((capacity, *input_shape), dtype=np.int)
        self.rewards = np.zeros(capacity, dtype=np.float)
        self.actions = np.zeros(capacity, dtype=np.int)
        self.terminal = np.zeros(capacity, dtype=np.bool)

    def push(self, observation, action, reward, next_observation, terminal):
        self.observations[self.index, :] = observation
        self.next_observations[self.index, :] = next_observation
        self.rewards[self.index] = reward
        self.actions[self.index] = action
        self.terminal[self.index] = terminal
        self.index = (1 + self.index) % self.capacity
        if self.index == 0 and not self.is_full:
            self.is_full = self.capacity
            print(f"Memory in size of<{self.capacity}> is full...")

    def get_minibatch_indices(self):
        return random.sample(range(self.capacity if self.is_full else self.index), self.minibatch_size)

    def generate_minibatch_samples(self, indices):
        # indices = random.sample(range(self.capacity if self.is_full else self.index), batch_size)
        batch_sample = (tf.cast(tf.stack(self.observations[indices, :], axis=0), dtype=tf.float32),
                        tf.cast(tf.stack(self.actions[indices], axis=0), dtype=tf.int32),
                        tf.cast(tf.stack(self.rewards[indices], axis=0), dtype=tf.float32),
                        tf.cast(tf.stack(self.next_observations[indices, :], axis=0), dtype=tf.float32),
                        tf.cast(tf.stack(self.terminal[indices], axis=0), dtype=tf.float32))
        return batch_sample


class ReplayBuffer2:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def generate_minibatch_samples(self):
        """Raclass ReplayBuffer:ndomly sample a batch of experiences from memory."""
        # sample batch from experience
        experiences = random.sample(self.memory, k=self.batch_size)

        # convert batch to tf objects
        states = tf.cast(tf.stack(np.array([e.state for e in experiences if e is not None]), axis=0), dtype=tf.float32)
        actions = tf.cast(tf.stack([e.action for e in experiences if e is not None], axis=0), dtype=tf.int32)
        rewards = tf.cast(tf.stack([e.reward for e in experiences if e is not None], axis=0), dtype=tf.float32)
        next_states = tf.cast(tf.stack([e.next_state for e in experiences if e is not None], axis=0), dtype=tf.float32)
        dones = tf.cast(tf.stack([e.done for e in experiences if e is not None], axis=0), dtype=tf.float32)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
