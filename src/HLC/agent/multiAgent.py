import os
import PIL
import imageio
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import deque
from HLC.agent.dqn_agent import Agent
from HLC.utils.metrics import get_metrics
from timeit import default_timer as timer


class MultiAgent:
    """
    Class for DQN model architecture.
    """
    def __init__(self, input_shape, num_actions, num_agents, ep_steps, minibatch_size=64, update_frequency=50,
                 capacity=170000, lr=1e-4, replay_start_size=5000, model_name="",
                 save_weight_interval=100, target_network_update_freq=5, runnig_from="", final_explr_frame=500000):

        # Multi agent DRL params
        self.input_shape = input_shape
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.ep_steps = ep_steps
        self.capacity = capacity
        self.discount_factor = 0.99
        self.minibatch_size = minibatch_size
        self.update_frequency = update_frequency
        self.target_network_update_freq = target_network_update_freq
        self.save_weight_interval = save_weight_interval
        self.log_path = os.path.join(runnig_from, model_name)
        self.init_explr = 1.0
        self.final_explr = 0.1
        self.final_explr_frame = final_explr_frame
        self.replay_start_size = replay_start_size
        self.training_frames = int(1e7)
        self.print_log_interval = 1

        # handling outputs directories
        tensorboard_dir_name = os.path.join(self.log_path, "summary")
        self.handle_directories(tensorboard_dir_name)
        self.summary_writer = tf.summary.create_file_writer(tensorboard_dir_name)

        # build agents
        self.agents_models_directory = os.path.join(self.log_path, "agents", "agent_{}")
        self.agents = {f"agent-{i}": Agent(num_actions=num_actions, input_shape=input_shape, lr=lr,
                                           capacity=capacity, agent_directory=self.agents_models_directory.format(i),
                                           replay_start_size=replay_start_size, minibatch_size=minibatch_size)
                       for i in range(num_agents)}

        # log memory usages
        self.display_memory_usage()

    def handle_directories(self, tensorboard_dir_name):
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)

        if os.path.isdir(tensorboard_dir_name):
            for f in os.listdir(tensorboard_dir_name):
                os.remove(os.path.join(tensorboard_dir_name, f))

        # create directory to save agents
        if not os.path.isdir(os.path.join(self.log_path, "agents")):
            os.mkdir(os.path.join(self.log_path, "agents"))

        # create directory to save gifs
        if not os.path.isdir(os.path.join(self.log_path, "gifs")):
            os.mkdir(os.path.join(self.log_path, "gifs"))

    def display_memory_usage(self):
        memory_usage = self.num_agents * np.float64(self.capacity) * (np.prod(self.input_shape) * 4 * 2 + 4 + 4 + 1)\
                       / 1024.0**3
        batch_size = (self.minibatch_size,) + self.input_shape
        print("Estimated memory usage ONLY for storing replays: {:.4f} GB, with Batch size of {}".format(memory_usage,
                                                                                                         batch_size))

    def get_actions(self, states: dict, total_step):
        """Get action by ε-greedy method.

        Args:
            total_step (np.unique)
            states (np.uint8): recent self.agent_history_length frames. (Default: (84, 84, 4))
        Returns:
            action (tf.int32): Action index
        """
        eps = self.get_eps(tf.constant(total_step, tf.float32))
        actions = {agent_id: agent.get_action(states[agent_id], eps)
                   for agent_id, agent in self.agents.items()}
        return actions

    def get_eps(self, current_step, terminal_eps=0.01, terminal_frame_factor=25):
        """Use annealing schedule similar like: https://openai.com/blog/openai-baselines-dqn/ .

        Args:
            current_step (int): Number of entire steps agent experienced.
            terminal_eps (float): Final exploration rate arrived at terminal_frame_factor * self.final_explr_frame.
            terminal_frame_factor (int): Final exploration frame, which is terminal_frame_factor * self.final_explr_frame.

        Returns:
            eps (float): Calculated epsilon for ε-greedy at current_step.
        """
        terminal_eps_frame = self.final_explr_frame * terminal_frame_factor

        if current_step < self.replay_start_size:
            eps = self.init_explr
        elif self.replay_start_size <= current_step < self.final_explr_frame:
            eps = (self.final_explr - self.init_explr) / (self.final_explr_frame - self.replay_start_size) * (current_step - self.replay_start_size) + self.init_explr
        elif self.final_explr_frame <= current_step < terminal_eps_frame:
            eps = (terminal_eps - self.final_explr) / (terminal_eps_frame - self.final_explr_frame) * (current_step - self.final_explr_frame) + self.final_explr
        else:
            eps = terminal_eps
        return eps

    def update_main_q_network(self):
        """Update main q network by experience replay method.
        Returns:
            loss (tf.float32): Huber loss of temporal difference.
        """
        loss, max_predicted_reward, certainty = list(zip(*[agent.update_main_q_network()
                                                           for _, agent in self.agents.items()]))
        return {"loss": np.mean(loss),
                "max_predicted_reward": np.mean(max_predicted_reward),
                "certainty": np.mean(certainty)}

    def update_target_network(self):
        """Update main q network by experience replay method.
        Returns:
            loss (tf.float32): Huber loss of temporal difference.
        """
        print("Updating target network...")
        loss = np.zeros(self.num_agents)
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            loss[i] = agent.update_target_network()

    def remember(self, transition):
        n_observation, actions, rewards, n_observation_next, done = transition
        for agent_id, agent in self.agents.items():
            if n_observation[agent_id].sum() != 0:
                agent.remember(n_observation[agent_id],
                               actions[agent_id],
                               rewards[agent_id],
                               n_observation_next[agent_id],
                               done[agent_id])

    def train(self, env):
        total_step = 0
        episode = 0
        latest_100_score = deque(maxlen=100)
        fit_results = None
        learned = False

        while total_step < self.training_frames:
            n_observation = env.reset()
            episode_step = 0
            done = False
            episode_score = np.zeros(self.num_agents)
            start = timer()
            t = 0
            # allocate memory fot high level metrics
            time_reward_collected, sleepers_time, fire_tracking, frames = [], [], [], []

            # one episode loop
            while not done:

                # action and transition
                actions = self.get_actions(n_observation, total_step)
                n_observation_next, rewards, done, info = env.step(actions)
                n_done = {agent: done for agent, _ in self.agents.items()}
                episode_score += np.fromiter(rewards.values(), dtype=np.float)

                # store transition in memory
                transition = (n_observation, actions, rewards, n_observation_next, n_done)
                self.remember(transition)
                n_observation = n_observation_next

                # update q network
                if (total_step % self.update_frequency == 0) and (total_step > self.replay_start_size):
                    learned = True
                    fit_results = self.learn()


                # collect behavior data for analysis
                time_reward_collected.extend([t for a, r in rewards.items() if r != 0])
                sleepers_time.extend([1 for a, obs in n_observation.items() if obs.sum() == 0])

                # extend tuple of (|agent hits in time t|,|shots in time t|)
                if env.is_apples_available():
                    current_hits = env.get_current_hits()
                    current_shots = (np.fromiter(actions.values(), dtype=float) == env.action_space["SHOOT"]).sum()  # count amount of shot actions
                    fire_tracking.extend([(current_hits, current_shots)])

                t += 1
                total_step += 1
                episode_step += 1

            # end of episode
            # if (episode % self.target_network_update_freq == 0) and learned:
            #     self.update_target_network()
            total_score = episode_score.sum()
            latest_100_score.append(total_score)
            eps = self.get_eps(tf.constant(total_step, tf.float32))
            fire_tracking = np.array(list(zip(*fire_tracking)), dtype=int).T.sum(axis=0)

            # write to summary after training has started
            if learned:
                self.write_summary(total_reward=episode_score, episode=episode, results=fit_results,
                                   eps=eps,  fire_efficiency=fire_tracking[0] / fire_tracking[1],
                                   time_reward_collected=time_reward_collected, sleepers_time=sleepers_time)
            episode += 1

            if episode % self.print_log_interval == 0:
                print(f"# {episode:<3d}, frames: {total_step}, total_rewards: {total_score},"
                      f" avg: {np.mean(latest_100_score):.2f}"
                      f" time: {timer() - start:.2f}, epsilon: {eps:.2f}")

            if episode % self.save_weight_interval == 0:
                print("Saving weights...")
                self.save_agents_weights(episode)

                print("Create gif...")
                self.play(env, self.ep_steps, total_step)

    def play(self,  env, ep_steps, total_steps=None):
        file_name = os.path.join(self.log_path, "gifs")
        n_observation = env.reset()
        frames = []
        test_step = 0
        test_reward = np.zeros(self.num_agents)
        # test_memory = ReplayMemory(10000, verbose=False)

        for t in range(ep_steps):
            frames.append(Image.fromarray(
                np.uint8(env.get_full_state())).resize(
                size=(720, 480), resample=PIL.Image.BOX).convert("RGB"))
            actions = self.get_actions(n_observation, total_steps)
            n_observation_next, reward, done, info = env.step(actions)
            test_reward += np.fromiter(reward.values(), dtype=int)

            # test_memory.push(state, action, reward, next_state, done)
            n_observation = n_observation_next

        score = test_reward.sum().astype(int)
        imageio.mimsave(os.path.join(file_name, f"ep_{total_steps//self.ep_steps}_score{score}.gif"), frames, fps=15)

    def write_summary(self, episode, total_reward, results, eps, fire_efficiency, time_reward_collected, sleepers_time):
        eq, sus, p, ef = get_metrics(total_reward, self.num_agents, self.ep_steps, sleepers_time, time_reward_collected)
        rewards = total_reward.sum()

        with self.summary_writer.as_default():
            # display fit results
            for k, v in results.items():
                tf.summary.scalar(name=k, data=v, step=episode)

            tf.summary.scalar(name="total reward", data=rewards, step=episode)
            tf.summary.scalar(name="epsilon", data=eps, step=episode)
            tf.summary.scalar(name="efficiency", data=ef, step=episode)
            tf.summary.scalar(name="sustainability", data=sus, step=episode)
            tf.summary.scalar(name="equality", data=eq, step=episode)
            tf.summary.scalar(name="peace", data=p, step=episode)
            tf.summary.scalar(name="fire efficiency", data=fire_efficiency, step=episode),

        # write data to log file
        self.summary_writer.flush()

    def save_agents_weights(self, episode):
        for agent_id, agent in self.agents.items():
            agent.save_weights(episode)

    def load_weights(self, run_path, ep):
        agents_dir = os.path.join(run_path, "agents")

        for i, agent_dir in enumerate(os.listdir(agents_dir)):
            weights_path = os.path.join(run_path, agent_dir, f"episode_{ep}")
            self.agents[f"agent-{i}"].load_weights(weights_path)

