#!/usr/bin/env python3
import os
import time
import tensorflow as tf
from datetime import date
from env.commons_env import MapEnv  # my env
from env.constants import SMALL_MAP, MEDIUM_MAP, DEFAULT_COLOURS, ORIGINAL_MAP
from HLC.agent.multiAgent import MultiAgent

# handling errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# limit gpu memory usage
def limit_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(5 * 1024))])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# tensorflow gpu-memory usage
gpu_memory_growth()

if __name__ == '__main__':
    # init training params
    ep_length = 250
    n_agents = 1
    lr = 1e-3
    target_network_update_freq = 15
    agents_vision = 15
    batch_size = 256
    capacity = int(1e5)
    input_shape = (agents_vision, agents_vision, 3)
    final_explr_frame = ep_length*600
    replay_start_size = ep_length*10
    save_every = 100
    model_type = "dense"
    # set up directories

    out_put_directory = os.path.join(os.getcwd(),os.path.pardir, "logs", "HLCDDQN_single")
    model_name = f"{date.today().strftime('%Y_%m_%d')}__{int(time.monotonic())}_{model_type}_lr_{str(lr)}"

    # init env
    env = MapEnv(bass_map=SMALL_MAP, num_agents=n_agents, color_map=DEFAULT_COLOURS,
                 agents_vision=agents_vision, normalize=True, ep_length=ep_length)
    num_actions = env.action_space_n

    # build model
    trainer = MultiAgent(input_shape=input_shape,
                         num_actions=num_actions,
                         ep_steps=ep_length,
                         model_name=model_name,
                         num_agents=n_agents,
                         lr=lr, save_weight_interval=save_every,
                         runnig_from=out_put_directory,
                         update_frequency=ep_length//1,
                         final_explr_frame=final_explr_frame,
                         replay_start_size=replay_start_size,
                         capacity=capacity,
                         minibatch_size=batch_size,
                         target_network_update_freq=target_network_update_freq)

    # start training
    trainer.train(env)
