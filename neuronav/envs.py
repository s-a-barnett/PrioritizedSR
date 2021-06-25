import numpy as np
import numpy.random as npr

class OpenRoom:

    def __init__(self, width=14, height=18, init_state=None, reward_loc=None):
        self.width      = width
        self.height     = height
        self.num_states = width * height

        if init_state is None:
            self.init_state = 0
        elif init_state == 'random':
            self.init_state = npr.choice(self.num_states)
        else:
            self.init_state = init_state

        if reward_loc is None:
            self.rewards = np.zeros(self.num_states)
        elif type(reward_loc) is int:
            self.rewards = np.eye(self.num_states)[reward_loc]
        else:
            self.rewards = np.eye(self.num_states)[self._coord2state(reward_loc)]

        # left, right, down, up, stay
        self.act2dir = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], int)
        self.num_actions = self.act2dir.shape[0]

        self.P = self._get_P()

    def reset(self):
        self.state = self.init_state

    def _state2coord(self, state):
        y, x = divmod(state, self.width)
        return np.array([x, y], int)

    def _coord2state(self, coord):
        return coord[1] * self.width + coord[0]

    def step(self, action):
        reward = self.rewards[self.state]
        coords = self._state2coord(self.state)
        dir    = self.act2dir[action]
        new_coords = np.clip(coords + dir, [0, 0], [self.width-1, self.height-1])
        new_state  = self._coord2state(new_coords)

        self.state = new_state

        return new_state, reward

    def _get_P(self):

        P = np.zeros((self.num_actions, self.num_states, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.state = s
                s_next, _ = self.step(a)
                P[a, s, s_next] = 1
        return P
