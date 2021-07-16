import numpy as np
import numpy.random as npr
from scipy.special import softmax
import utils

class TDSR:

    def __init__(self, state_size, action_size, learning_rate, gamma, M_init=None, poltype='egreedy', eig_size=None, eig_init=None):
        self.state_size = state_size
        self.action_size = action_size

        self.poltype = poltype

        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack([0.01 * npr.randn(state_size, state_size) for i in range(action_size)])
        else:
            self.M = M_init
        
        if eig_size is None:
            self.eig_size = state_size
        else:
            self.eig_size = eig_size

        if eig_init is None:
            self.eigs, _ = np.linalg.qr(npr.rand(state_size, self.eig_size))
        elif (eig_init == 'id'):
            self.eigs = np.eye(state_size, self.eig_size)
        else:
            assert eig_init.shape == (state_size, self.eig_size)
            self.eigs = eig_init

        self.w = np.zeros([state_size])
        self.learning_rate = learning_rate
        self.gamma = gamma

    def Q_estimates(self, state, goal=None):
        if goal is None:
            goal = self.w
        else:
            goal = utils.onehot(goal, self.state_size)
        return self.M[:, state, :] @ goal

    def sample_action(self, state, goal=None, epsilon=0.0, beta=1e6):
        if self.poltype == 'softmax':
            Qs = self.Q_estimates(state, goal)
            action = npr.choice(self.action_size, p=softmax(beta * Qs))
        else:
            if npr.rand() < epsilon:
                action = npr.choice(self.action_size)
            else:
                Qs = self.Q_estimates(state, goal)
                action = np.argmax(Qs)
        return action

    def update_w(self, current_exp):
        s_1 = current_exp[2]
        r = current_exp[3]
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error
        return error

    def update_sr(self, current_exp, next_exp, prospective=False):
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]
        s_a_1 = next_exp[1]
        r = current_exp[3]
        d = current_exp[4]
        I = utils.onehot(s, self.state_size)
        if d:
            td_error = (I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :])
        else:
            td_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])
        if not prospective:
            self.M[s_a, s, :] += self.learning_rate * td_error
        return td_error

    def evb(self, state, exp, epsilon=0.0, beta=1e6):
        s = exp[0]
        s_a = exp[1]
        s_1 = exp[2]
        r = exp[3]
        M_new = self.M.copy()
        M_new[s_a, s, :] += self.learning_rate * utils.memory_update(exp, self, prospective=True)
        w_new = self.w.copy()
        w_new[s_1] += self.learning_rate * (r - w_new[s_1])

        Q_new = M_new @ w_new
        pi_new = self.get_policy(M=M_new, goal=w_new, epsilon=epsilon, beta=beta)
        pi_old = self.get_policy(epsilon=epsilon, beta=beta)
        gain = (Q_new.T @ (pi_new - pi_old))[s, s]
        M_new_states = np.diagonal(np.tensordot(pi_new.T, M_new, axes=1), axis1=0, axis2=1).T
        need = M_new_states[state, s]
        return gain * need

    def update_eigs(self, current_exp):
        s = current_exp[0]
        s_1 = current_exp[2]
        diff = self.eigs[s] - self.eigs[s_1]
        grad = 2 * self.gamma * diff
        self.eigs[s] -= self.learning_rate * grad
        self.eigs[s_1] += self.learning_rate * grad
        return grad

    def get_policy(self, M=None, goal=None, epsilon=0.0, beta=1e6):
        if goal is None:
            goal = self.w

        if M is None:
            M = self.M

        Q = M @ goal
        if self.poltype == 'softmax':
            policy = softmax(beta * Q, axis=0)
        else:
            mask = (Q == Q.max(0))
            greedy = mask / mask.sum(0)
            policy = (1 - epsilon) * greedy + (1 / self.action_size) * epsilon * np.ones((self.action_size, self.state_size))
        return policy

    def get_M_states(self, goal=None, epsilon=0.0, beta=1e6):
        policy = self.get_policy(goal=goal, epsilon=epsilon, beta=beta)
        M = np.diagonal(np.tensordot(policy.T, self.M, axes=1), axis1=0, axis2=1).T
        return M

