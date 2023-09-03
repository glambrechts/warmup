import numpy as np
import random


class Trajectory:
    """
    Store a trajectory sampled in an environment.

     - num_actions: int
        The number of actions that can be taken in the environment
     - observation_size: int
        The size of the observation vectors
    """

    def __init__(self, num_actions, observation_size):
        """
        See class documentation.
        """
        self.num_actions = num_actions
        self.observation_size = observation_size
        self.observed = []
        self.terminal = False

    def add(self, action, reward, observation, terminal=False):
        """
        Adds the action, reward, observation and terminal indicator in the
        trajectory.
        """
        assert not self.terminal

        one_hot = np.zeros(self.num_actions, dtype=np.float32)
        if action is not None:
            one_hot[action] = 1.
        action = one_hot

        if reward is not None:
            reward = np.array([reward], dtype=np.float32)
        else:
            reward = np.array([0.], dtype=np.float32)

        self.observed.append(np.concatenate((action, reward, observation)))

        if terminal:
            self.terminal = True

    @property
    def num_transitions(self):
        """
        Returns the number of transitions of the trajectory.
        """
        return len(self.observed) - 1

    def get_cumulative_reward(self, gamma=1.0):
        """
        Returns the (discounted) cumulative reward of the trajectory.
        """
        return sum(obs[self.num_actions] * gamma ** t for t, obs in
                   enumerate(self.observed[1:]))

    def get_last_observed(self, number=None):
        """
        Returns the lats observation of the trajectory.
        """
        if number is None:
            return self.observed[-1]
        else:
            print("This should not happened")
            truncated = self.observed[- number:]
            if len(truncated) < number:
                padding = []
                for i in range(number - len(truncated)):
                    padding.append(np.zeros(self.observed[-1].shape[0]))
                truncated = padding + truncated
            return np.stack(truncated)

    def get_transitions(self):
        """
        Returns the trajectory as a list of transitions.
        """
        sequence = np.stack(self.observed)

        transitions = []
        for t in range(sequence.shape[0] - 1):
            seq_bef = sequence[:t + 1, :]
            seq_aft = sequence[:t + 2, :]
            a = sequence[t + 1, :self.num_actions]
            r = sequence[t + 1, self.num_actions]
            o = sequence[t + 1, self.num_actions + 1:]
            if a.sum() == 0:
                a = None
                r = None
            else:
                a = a.argmax()
                r = r.item()
            d = self.terminal and t == sequence.shape[0] - 2
            transitions.append((seq_bef, a, r, o, d, seq_aft))

        return transitions


class ReplayBuffer:
    """
    Replay Buffer storing transitions from trajectories.

     - capacity: int
        The number of transitions that the replay buffer can store.
    """

    def __init__(self, capacity):
        """
        See class documentation.
        """
        self.capacity = capacity
        self.buffer = []
        self.last = 0
        self.count = 0

    @property
    def is_full(self):
        return self.capacity == self.count

    def add_transition(self, transition):
        """
        Adds a transition in the replay buffer.
        """
        if self.count < self.capacity:
            self.buffer.append(transition)
            self.count += 1
        else:
            self.buffer[self.last] = transition
            self.last = (self.last + 1) % self.capacity

    def add(self, trajectory):
        """
        Adds all transitions of a trajectory in the replay buffer.
        """
        assert isinstance(trajectory, Trajectory)
        for transition in trajectory.get_transitions():
            self.add_transition(transition)

    def sample(self, number):
        """
        Sample `number` transitions in the replay buffer, with replacement.
        """
        return random.choices(self.buffer, k=number)

    def get_input_sequences(self):
        """
        Only returns the input histories (sequences of observations and
        actions).
        """
        return [transition[0] for transition in self.buffer]
