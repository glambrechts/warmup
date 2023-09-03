import random
import numpy as np


class Position:
    """
    Cartesian coordinates in a 2-dimensional space.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, p):
        return self.x == p.x and self.y == p.y


class State:
    """
    T-Maze state.
    """
    def __init__(self, goal, cell):
        self.goal = goal
        self.cell = cell


class TMaze:
    """
    T-Maze environment as described in the paper appendix.

    Arguments
    ---------
     - length: int
        T-Maze length $L$.
     - stochasticity: float
        T-Maze stochasticity rate $\lambda$.
    """
    observation_size = 4
    o_up, o_down, o_corridor, o_junction = range(observation_size)
    observations = np.eye(observation_size, dtype=np.float32)

    num_actions = 4
    a_right, a_up, a_left, a_down = range(num_actions)
    actions = list(range(num_actions))

    gamma = 0.98

    def __init__(self, length, stochasticity=0.0):
        """
        See class documentation.
        """
        if length < 2:
            raise ValueError('T-Maze length should be at least 2')

        self.length = length
        self.stochasticity = stochasticity

        self.goal_up = Position(self.length, 1)
        self.goal_down = Position(self.length, -1)

    def exploration_policy(self):
        """
        Returns an action sampled according to the exploration policy for this
        environment.
        """
        return np.random.choice(self.actions, p=[1/2, 1/6, 1/6, 1/6])

    def get_recommanded_horizon(self):
        """
        Returns the recommended time horizon for this environment
        """
        return 3 * self.length - 1

    def reset(self, p_up=0.5):
        """
        Resets the environment. Samples a new initial state and gives the
        returns the initial observation.
        """
        goal = self.goal_up if random.random() < p_up else self.goal_down

        cell = Position(0, 0)
        self.state = State(goal, cell)
        observation, terminal = self.observation_()

        return observation

    def observation_(self):
        """
        Returns the observation associated with the current state.
        """
        # If in first cell and goal up
        if self.state.cell.x == 0 and self.state.goal == self.goal_up:
            return self.observations[self.o_up, :], False
        # If in first cell and goal down
        if self.state.cell.x == 0 and self.state.goal == self.goal_down:
            return self.observations[self.o_down, :], False
        # If in corridor
        if self.state.cell.x < self.length:
            return self.observations[self.o_corridor, :], False
        # If at junction
        if self.state.cell.x == self.length and self.state.cell.y == 0:
            return self.observations[self.o_junction, :], False
        # If in terminal state
        if self.state.cell.x == self.length and self.state.cell.y in (-1, 1):
            return self.observations[self.o_junction, :], True
        # Otherwise, error
        raise ValueError('Invalid environment state')

    def reward_(self, a):
        """
        Returns the reward associated with the current state and action `a`.
        """
        # If in terminal state
        if self.state.cell.x == self.length and self.state.cell.y in (-1, 1):
            return 0.
        # If bouncing onto a wall
        if ((self.state.cell.x == 0 and a == self.a_left) or
                (self.state.cell.x == self.length and a == self.a_right) or
                (self.state.cell.x < self.length and a in
                    (self.a_down, self.a_up))):
            return -.1
        # If in corridor
        if (self.state.cell.x <= self.length and a in
                (self.a_right, self.a_left)):
            return 0.
        # If finding the goal
        if (self.state.cell.x == self.length and
                ((a == self.a_up and self.state.goal == self.goal_up) or
                    (a == self.a_down and self.state.goal == self.goal_down))):
            return 4.
        # If missing the goal
        if (self.state.cell.x == self.length and
                ((a == self.a_up and self.state.goal == self.goal_down) or
                    (a == self.a_down and self.state.goal == self.goal_up))):
            return -.1
        # Otherwise, error
        raise ValueError('Invalid environment state')

    def transition_(self, a):
        """
        Update the state according to the current state and action `a`.
        """
        if (self.state.cell.x < self.length and
                random.random() < self.stochasticity):
            a = random.choice((self.a_right, self.a_up, self.a_left,
                               self.a_down))

        # If in a terminal state, we stay in a terminal state
        if self.state.cell.x == self.length and self.state.cell.y in (-1, 1):
            return
        # If in first cell
        if self.state.cell.x == 0:
            if a == self.a_right:
                self.state.cell.x += 1
            return
        # If in corridor
        if self.state.cell.x < self.length:
            if a == self.a_right:
                self.state.cell.x += 1
            elif a == self.a_left:
                self.state.cell.x -= 1
            return
        # If at junction
        if self.state.cell.x == self.length:
            if a == self.a_left:
                self.state.cell.x -= 1
            elif a == self.a_up:
                self.state.cell.y = 1
            elif a == self.a_down:
                self.state.cell.y = -1
            return
        # Otherwise, error
        raise ValueError('Invalid environment state')

    def step(self, action):
        """
        Returns the observation, reward and terminal indicator resulting from
        taking action `action` in the environment.
        """
        # r_t = R(s_t, a_t)
        reward = self.reward_(action)
        # s_{t+1} ~ T(.|s_t, a_t)
        self.transition_(action)
        # o_{t+1} ~ O(.|s_{t+1})
        observation, terminal = self.observation_()

        return observation, reward, terminal, None

    def render(self):
        """
        Print a representation of the current state of the environment in the
        terminal.
        """
        if self.state.goal == self.goal_up:
            mark_up, mark_down = ('G', ' ')
        else:
            mark_up, mark_down = (' ', 'G')

        if self.state.cell == self.goal_up:
            mark_up = 'x'
        elif self.state.cell == self.goal_down:
            mark_down = 'x'

        print('\n' + ' ' * 2 * self.length + '|{}|'.format(mark_up),
              flush=False)
        for i in range(self.length + 1):
            if self.state.cell.x == i and self.state.cell.y == 0:
                print('|x', end='', flush=False)
            else:
                print('| ', end='', flush=False)
        print('|\n' + ' ' * 2 * self.length + '|{}|\n'.format(mark_down),
              flush=True)
