from env.commons_env import *
from env.utils import obs2gray_scale
from env.constants import DEFAULT_COLOURS, ORIGINAL_MAP
import numpy as np
n_agents = 1
agents_vision = 21
gray_scale = False
episode = 0
ep_length = 750

env = MapEnv(bass_map=ORIGINAL_MAP, num_agents=n_agents, color_map=DEFAULT_COLOURS, ep_length=ep_length,
             agents_vision=agents_vision, gray_scale=gray_scale, normalize=True)
done = False
n_observation = env.reset()
t = 0
episode_score = np.zeros(n_agents)

while not done:
    env.render()
    # action and transition
    print(f"iter: {t}")
    actions = {f"agent-{i}": int(input(f"enter action for agent: {i}")) for i in range(n_agents)}
    n_observation_next, rewards, done, info = env.step(actions)
    episode_score += np.fromiter(rewards.values(), dtype=np.float)
    n_observation = n_observation_next
