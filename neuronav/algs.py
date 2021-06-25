import numpy as np
import numpy.random as npr

class VI:

    def __init__(self, env, discount=0.99, err_tol=1e-2):
        self.env = env
        self.V   = np.zeros(env.num_states)
        self.pi  = np.zeros(env.num_states, env.num_actions)
        self.discount = discount
        self.err_tol = err_tol

    def _step(self):
        q = env.rewards + self.discount * (env.P @ self.V).T
        self.V = np.amax(q, axis=1)
        self.pi = (q == q.max(axis=1)[:, None]).astype(int)
        return self.V, self.pi

    def train(self):
        old_V = self.V

class SR_TD:

    def __init__(self, env, discount=0.99, num_steps=10000, lr_sr=0.1, lr_td=0.1, epsilon=5e-2):

        self.env = env
        env.reset()

        self.discount = discount
        self.num_steps = num_steps
        self.lr_sr = lr_sr
        self.lr_td = lr_td
        self.epsilon = epsilon

        self.w  = np.zeros(env.num_states)
        self.sr = np.eye(env.num_states)

    def _step(self):
        V = self.sr @ self.w
        s = env.state

        # currently only works for deterministic environments
        if npr.rand() < self.epsilon:
            a = npr.choice(self.num_actions)
        else:
            next_states = np.nonzero(env.P[:, s, :])[1]
            next_vals   = V[next_states]
            a           = np.argmax(next_vals)

        s_next, r = env.step(a)

        delta = r + self.discount * V[s_next] - V[s]

        self.w += lr_td * delta * self.sr[s, :]

        sr_error = np.eye(env.num_states)[s_next] + self.discount * self.sr[s_next] - self.sr[s]
        self.sr[s] += lr_sr * sr_error
