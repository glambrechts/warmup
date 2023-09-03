import os
import numpy as np

from random import random
from memory import Trajectory
from models_pytorch.models import Model


class DRQN:
    """
    Deep Recurrent Q-Network RL agent.

    Arguments
    ---------
     - num_actions: int
        Number of discrete action for the environment
     - observation_size: int
        Dimension of the observation space of the environment
     - kwargs: dict
        Additional arguments for the underlying models (neural networks)
    """

    def __init__(self, num_actions, observation_size, **kwargs):
        """
        See class documentation.
        """
        self.num_actions = num_actions
        self.observation_size = observation_size
        self.Q = Model(**kwargs)
        self.Q_tar = Model(**kwargs)

    def update_target(self):
        """
        Updates the target network's weights to the current network's weights.
        """
        self.Q_tar.set_weights(self.Q.get_weights())

    def play(self, env, epsilon=0.2, buffer=None, force=None):
        """
        Sample a trajectory in the environment using an epsilon-greedy policy
        derived from the current estimation of the Q-function..

        Returns
        -------
         - trajectory: Trajectory
            Trajectory sampled until a terminal state or the time horizon.
        """
        trajectory = Trajectory(self.num_actions, self.observation_size)

        if force is None:
            observation = env.reset()
        elif force == 'up':
            observation = env.reset(p_up=1.0)
        elif force == 'down':
            observation = env.reset(p_up=0.0)

        trajectory.add(None, None, observation)
        hidden_states = None

        for t in range(env.get_recommanded_horizon()):

            values, hidden_states = self.Q.predict(
                trajectory.get_last_observed().reshape(1, -1),
                hidden_states)

            if random() < epsilon:
                action = env.exploration_policy()
            else:
                action = np.argmax(values.flatten())

            observation, reward, terminal, _, = env.step(action)
            trajectory.add(action, reward, observation, terminal)

            if terminal:
                break

        if buffer:
            buffer.add(trajectory)

        return trajectory

    def eval_tmaze_optimal(self, env):
        """
        Determines if the policy is optimal for the T-Maze environment.
        """
        trajectory_up = self.play(env, epsilon=0.0, force='up')
        trajectory_down = self.play(env, epsilon=0.0, force='down')

        if (trajectory_up.get_cumulative_reward() == 4.0 and
                trajectory_down.get_cumulative_reward() == 4.0):
            if (trajectory_up.num_transitions == env.length + 1 and
                    trajectory_down.num_transitions == env.length + 1):
                return True

        return False

    def eval(self, env, num_rollouts):
        """
        Evaluates the cumulative return of the current policy on the
        environment with `num_rollouts` rollouts.
        """
        rewards, disc_rewards = [], []
        for _ in range(num_rollouts):
            trajectory = self.play(env, epsilon=0.0, buffer=None)
            rewards.append(trajectory.get_cumulative_reward())
            disc_rewards.append(trajectory.get_cumulative_reward(env.gamma))

        mean_rewards = sum(rewards) / len(rewards)
        mean_disc_rewards = sum(disc_rewards) / len(disc_rewards)
        probas = [0, 0.25, 0.5, 0.75, 1]
        quantiles_rewards = np.quantile(rewards, probas)
        quantiles_disc_rewards = np.quantile(disc_rewards, probas)

        return {
            'train/mean': mean_rewards,
            'train/min': quantiles_rewards[0],
            'train/q1': quantiles_rewards[1],
            'train/median': quantiles_rewards[2],
            'train/q3': quantiles_rewards[3],
            'train/max': quantiles_rewards[4],
            'train/disc_mean': mean_disc_rewards,
            'train/disc_min': quantiles_disc_rewards[0],
            'train/disc_q1': quantiles_disc_rewards[1],
            'train/disc_median': quantiles_disc_rewards[2],
            'train/disc_q3': quantiles_disc_rewards[3],
            'train/disc_max': quantiles_disc_rewards[4]
        }

    def optimize(self, transitions, gamma, learning_rate):
        """
        Improves the current approximation of the Q-function using the set of
        transitions.
        """
        inputs, targets, masks = [], [], []
        for transition in transitions:

            seq_bef, action, reward, _, terminal, seq_aft = transition

            target = reward
            if not terminal:
                Q_next, _ = self.Q_tar.predict(seq_aft)
                target += gamma * Q_next[-1, :].max()

            target = np.array([[target]], dtype=np.float32)
            mask = np.zeros((seq_bef.shape[0], self.num_actions),
                            dtype=bool)
            mask[-1, action] = True

            inputs.append(seq_bef)
            targets.append(target)
            masks.append(mask)

        return self.Q.training_step(inputs, targets, masks, learning_rate)

    def save(self, run_name, episode=None):
        """
        Save the weights of the network and target network.
        """
        os.makedirs('weights', exist_ok=True)

        self.Q.save(f'weights/{run_name}-{episode}-Q.pth')
        self.Q_tar.save(f'weights/{run_name}-{episode}-Q_tar.pth')

    def load(self, run_name, episode=None):
        """
        Load the weights of the network and target network
        """
        self.Q.load(f'weights/{run_name}-{episode}-Q.pth')
        self.Q_tar.load(f'weights/{run_name}-{episode}-Q_tar.pth')
