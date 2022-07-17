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


# tensorflow gpu-memory usage
limit_gpu()

if __name__ == '__main__':
    # init training params
    ep_length = 750
    n_agents = 12
    lr = 1e-5
    agents_vision = 21
    gray_scale = False
    channels = 1 if gray_scale else 3
    input_shape = (agents_vision, agents_vision, channels)
    replay_start_size = int(1e6)
    save_every = 100
    model_type = "dense"
    # set up directories

    out_put_directory = os.path.join(os.getcwd(),os.path.pardir, "logs", "HLCDDQN")
    model_name = f"{date.today().strftime('%Y_%m_%d')}__{int(time.monotonic())}_{model_type}_lr_{str(lr)}"

    # init env
    env = MapEnv(bass_map=ORIGINAL_MAP, num_agents=n_agents, color_map=DEFAULT_COLOURS,
                 agents_vision=agents_vision, gray_scale=gray_scale, normalize=True)
    num_actions = env.action_space_n

    # build model
    trainer = MultiAgent(input_shape=input_shape,
                         num_actions=num_actions,
                         ep_steps=ep_length,
                         agent_history_length=channels,
                         model_name=model_name,
                         num_agents=n_agents,
                         lr=lr, save_weight_interval=save_every,
                         runnig_from=out_put_directory)

    # start training
    trainer.train_no_history(env)
