
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Input
from tensorflow.keras.initializers import VarianceScaling as initializer


class DQNNetwork2(Model):
    """
    Class for DQN model architecture.
    """
    
    def __init__(self, num_actions: int, agent_history_length: int):
        super(DQNNetwork, self).__init__()
        self.normalize = Lambda(lambda x: x / 255.0)
        self.conv1 = Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu", input_shape=(None, 84, 84, agent_history_length))
        self.conv2 = Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')
        self.dense2 = Dense(num_actions, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="linear")

    @tf.function
    def call(self, x):
        normalized = self.normalize(x)
        h1 = self.conv1(normalized)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.flatten(h3)
        h5 = self.dense1(h4)
        out = self.dense2(h5)
        return out


class DQNNetwork(Model):
    """
    Class for DQN model architecture.
    """

    def __init__(self, input_shape, num_actions: int):
        super(DQNNetwork, self).__init__()
        self.layers_ = []
        # self.layers_.append(Input(shape=(None, *input_shape)))
        self.layers_.append(Flatten(input_dim=input_shape))
        self.layers_.append(Dense(32, kernel_initializer=initializer(scale=.2), activation='relu'))
        self.layers_.append(Dense(32, kernel_initializer=initializer(scale=.2), activation='relu'))
        self.layers_.append(Dense(num_actions, kernel_initializer=initializer(scale=.2), activation="linear"))

    @tf.function
    def call(self, x):
        for layer in self.layers_:
            x = layer(x)
        return x


def build_dense_model(input_shape, lr, n_actions):
    """
    build keras.Sequential conv model
    @return: predict model, target model, optimizer
    """
    # inputs
    inputs = Input(shape=input_shape)
    # conv1 = Conv2D(32, 3, strides=1, padding='same', kernel_initializer="glorot_uniform", name="conv1")(inputs)
    # conv2 = Conv2D(16, 3, strides=2, padding='same', kernel_initializer="glorot_uniform", name="conv2")(conv1)

    # flatten
    flatten = Flatten(name="flatten")(inputs)

    # dense
    dense1 = Dense(32, kernel_initializer=initializer(scale=.2), activation='relu')(flatten)
    dense2 = Dense(32, kernel_initializer=initializer(scale=.2), activation='relu')(dense1)

    # outputs
    q_value = Dense(n_actions, kernel_initializer=initializer(scale=.2), activation='relu')(dense2)

    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # build predict model
    q_predict = Model(inputs=inputs, outputs=q_value)
    q_predict.compile(optimizer=opt, loss='mse')

    # build target model
    q_target = tf.keras.models.clone_model(q_predict)
    q_target.compile(optimizer=opt, loss='mse')
    return q_predict, q_target

