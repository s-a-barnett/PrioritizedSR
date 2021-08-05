import numpy as np
import numpy.random as npr
from scipy.special import softmax
from collections import defaultdict
import utils

class TDQ:
    def __init__(self, state_size, action_size, learning_rate=1e-1, gamma=0.99, Q_init=None):
        self.state_size = state_size
        self.action_size = action_size

        if Q_init == None:
            self.Q = np.zeros((action_size, state_size))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size)
        else:
            self.Q = Q_init

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.prioritized_states = np.zeros(state_size, dtype=np.int)

    def sample_action(self, state, beta=1e6):
        Qs = self.Q[:, state]
        action = npr.choice(self.action_size, p=softmax(beta * Qs))
        return action

    def update_q(self, current_exp, next_exp=None, prospective=False):
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]

        # determines whether update is on-policy or off-policy
        if next_exp is None:
            s_a_1 = np.argmax(self.Q[:, s_1])
        else:
            s_a_1 = next_exp[1]

        r = current_exp[3]

        q_error = r + self.gamma * self.Q[s_a_1, s_1] - self.Q[s_a, s]

        if not prospective:
            # actually perform update to Q if not prospective
            self.Q[s_a, s] += self.learning_rate * q_error
        return q_error

    def update(self, current_exp, **kwargs):
        q_error = self.update_q(current_exp, **kwargs)
        td_error = {'q': np.linalg.norm(q_error)}
        return td_error

    def get_policy(self, beta=1e6):
        policy = softmax(beta * self.Q, axis=0)
        return policy

class DynaQ(TDQ):

    def __init__(self, state_size, action_size, num_recall, kappa=0.0,  **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.experiences = []
        self.kappa = kappa
        self.tau = np.zeros((state_size, action_size), dtype=np.int)
        
    def _get_dyna_indices(self):
        p = utils.exp_normalize(np.arange(len(self.experiences)))
        return npr.choice(len(self.experiences), self.num_recall, p=p, replace=True)

    def _add_intrinsic_reward(self, exp):
        state, action, state_next, reward, done = exp
        reward += self.kappa * np.sqrt(self.tau[state, action])
        return state, action, state_next, reward, done

    def update(self, current_exp, **kwargs):

        # update time elapsed since state-action pair last seen
        self.tau += 1
        self.tau[current_exp[0], current_exp[1]] = 0

        # remember sliding window of past 10 experiences
        self.experiences.append(current_exp)
        self.experiences = self.experiences[-10:]

        # perform online update first
        q_error = self.update_q(current_exp, **kwargs)

        mem_indices = self._get_dyna_indices()
        mem = [self.experiences[i] for i in mem_indices]
        for exp in mem:
            exp = self._add_intrinsic_reward(exp)
            self.prioritized_states[exp[0]] += 1
            # perform off-policy update using recalled memories
            q_error = self.update_q(exp)

        td_error = {'q': np.linalg.norm(q_error)}
        return td_error

class PSQ(TDQ):

    def __init__(self, state_size, action_size, num_recall, theta=1e-6, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.theta = theta
        self.pqueue = utils.PriorityQueue()
        self.predecessors = defaultdict(set)
        self.model = {}

    def _update_predecessors(self, state, action, next_state):
        self.predecessors[next_state].add((state, action))

    def priority(self, q_error):
        error = q_error
        return np.linalg.norm(error)
    
    def update(self, current_exp, **kwargs):
        state, action, next_state, reward, done = current_exp

        # update (deterministic) model and state predecessors
        self.model[(state, action)] = (next_state, reward, done)
        self._update_predecessors(state, action, next_state)

        # compute value of update
        q_error = self.update_q(current_exp, prospective=True, **kwargs)

        # compute priority for the update, add to queue
        priority = self.priority(q_error)
        if priority > self.theta:
            self.pqueue.push((state, action), -priority)

        for k in range(self.num_recall):
            if self.pqueue.is_empty():
                break

            # get highest priority experience
            state, action = self.pqueue.pop()
            self.prioritized_states[state] += 1
            exp = (state, action) + self.model[(state, action)]

            # update Q based on this experience
            q_error = self.update_q(exp)

            for s, a in self.predecessors[state]:
                # add predecessors to priority queue
                exp_pred = (s, a) + self.model[(s, a)]
                q_error = self.update_q(exp_pred, prospective=True)
                priority = self.priority(q_error)
                if priority > self.theta:
                    self.pqueue.push((s, a), -priority)

        td_error = {'q': np.linalg.norm(q_error)}
        return td_error


class TDSR:

    def __init__(self, state_size, action_size, learning_rate=1e-1, gamma=0.99, M_init=None, poltype='softmax'):
        self.state_size = state_size
        self.action_size = action_size

        self.poltype = poltype

        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack([M_init * npr.randn(state_size, state_size) for i in range(action_size)])
        else:
            self.M = M_init
        
        self.w = np.zeros(state_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.prioritized_states = np.zeros(state_size, dtype=np.int)

    def Q_estimates(self, state):
        return self.M[:, state, :] @ self.w

    def sample_action(self, state, epsilon=0.0, beta=1e6):
        if self.poltype == 'softmax':
            Qs = self.Q_estimates(state)
            action = npr.choice(self.action_size, p=softmax(beta * Qs))
        else:
            if npr.rand() < epsilon:
                action = npr.choice(self.action_size)
            else:
                Qs = self.Q_estimates(state)
                action = np.argmax(Qs)
        return action

    def update_w(self, current_exp):
        s_1 = current_exp[2]
        r = current_exp[3]
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error
        return error

    def update_sr(self, current_exp, next_exp=None, prospective=False):
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]

        # determines whether update is on-policy or off-policy
        if next_exp is None:
            s_a_1 = np.argmax(self.Q_estimates(s_1))
        else:
            s_a_1 = next_exp[1]

        r = current_exp[3]
        d = current_exp[4]
        I = utils.onehot(s, self.state_size)

        if d:
            m_error = (I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :])
        else:
            m_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])

        if not prospective:
            # actually perform update to SR if not prospective
            self.M[s_a, s, :] += self.learning_rate * m_error
        return m_error

    def update(self, current_exp, **kwargs):
        m_error = self.update_sr(current_exp, **kwargs)
        w_error = self.update_w(current_exp)
        td_error = {'m': np.linalg.norm(m_error), 'w': np.linalg.norm(w_error)}
        return td_error

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

    def get_M_states(self, epsilon=0.0, beta=1e6):
        # average M(a, s, s') according to policy to get M(s, s')
        policy = self.get_policy(epsilon=epsilon, beta=beta)
        M = np.diagonal(np.tensordot(policy.T, self.M, axes=1), axis1=0, axis2=1).T
        return M


class DynaSR(TDSR):
    
    def __init__(self, state_size, action_size, num_recall, kappa=0.0, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.experiences = []
        self.kappa = kappa
        self.tau = np.zeros((state_size, action_size), dtype=np.int)
        
    def _get_dyna_indices(self):
        p = utils.exp_normalize(np.arange(len(self.experiences)))
        return npr.choice(len(self.experiences), self.num_recall, p=p, replace=True)

    def _add_intrinsic_reward(self, exp):
        state, action, state_next, reward, done = exp
        reward += self.kappa * np.sqrt(self.tau[state, action])
        return state, action, state_next, reward, done

    def update(self, current_exp, **kwargs):

        # update time elapsed since state-action pair last seen
        self.tau += 1
        self.tau[current_exp[0], current_exp[1]] = 0

        # remember sliding window of past 10 experiences
        self.experiences.append(current_exp)
        self.experiences = self.experiences[-10:]

        # perform online update first
        m_error = self.update_sr(current_exp, **kwargs)
        w_error = self.update_w(current_exp)

        mem_indices = self._get_dyna_indices()
        mem = [self.experiences[i] for i in mem_indices]
        for exp in mem:
            exp = self._add_intrinsic_reward(exp)
            self.prioritized_states[exp[0]] += 1
            # perform off-policy update using recalled memories
            m_error = self.update_sr(exp)

        td_error = {'m': np.linalg.norm(m_error), 'w': np.linalg.norm(w_error)}
        return td_error


class PSSR(TDSR):

    def __init__(self, state_size, action_size, num_recall, theta=1e-6, goal_pri=True, online=True, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.theta = theta
        self.goal_pri = goal_pri
        self.online = online
        self.pqueue = utils.PriorityQueue()
        self.predecessors = defaultdict(set)
        self.model = {}

    def _update_predecessors(self, state, action, next_state):
        self.predecessors[next_state].add((state, action))

    def priority(self, m_error, current_exp):
        if self.goal_pri:
            # priority given by temporal difference of Q
            error = np.dot(m_error, self.w) - self.w[current_exp[0]] + current_exp[3]
        else:
            # priority given by temporal difference of successor representation
            error = m_error
        return np.linalg.norm(error)
    
    def update(self, current_exp, **kwargs):
        state, action, next_state, reward, done = current_exp

        # update (deterministic) model and state predecessors
        self.model[(state, action)] = (next_state, reward, done)
        self._update_predecessors(state, action, next_state)

        # compute value of update
        m_error = self.update_sr(current_exp, prospective=(not self.online), **kwargs)
        w_error = self.update_w(current_exp) if self.online else 0.0

        # compute priority for the update, add to queue
        priority = self.priority(m_error, current_exp)
        if priority > self.theta:
            self.pqueue.push((state, action), -priority)

        for k in range(self.num_recall):
            if self.pqueue.is_empty():
                break

            # get highest priority experience
            state, action = self.pqueue.pop()
            self.prioritized_states[state] += 1
            exp = (state, action) + self.model[(state, action)]

            # update M and w based on this experience
            m_error = self.update_sr(exp)
            w_error = self.update_w(exp)

            for s, a in self.predecessors[state]:
                # add predecessors to priority queue
                exp_pred = (s, a) + self.model[(s, a)]
                m_error = self.update_sr(exp_pred, prospective=(not self.online))
                priority = self.priority(m_error, exp_pred)
                if priority > self.theta:
                    self.pqueue.push((s, a), -priority)

        td_error = {'m': np.linalg.norm(m_error), 'w': np.linalg.norm(w_error)}
        return td_error


class MDSR(TDSR):

    def __init__(self, state_size, action_size, num_recall, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.experiences = []

    def evb(self, state, exp, epsilon=0.0, beta=1e6):
        s = exp[0]
        s_a = exp[1]
        s_1 = exp[2]
        r = exp[3]

        # compute new M and w based on experience to be evaluated
        M_new = self.M.copy()
        M_new[s_a, s, :] += self.learning_rate * self.update_sr(exp, prospective=True)
        w_new = self.w.copy()
        w_error = r - w_new[s_1]
        w_new[s_1] += self.learning_rate * w_error

        # get old policy as a baseline
        pi_old = self.get_policy(epsilon=epsilon, beta=beta)

        # compute gain term
        if w_error != 0:
            Q_new = M_new @ w_new
            pi_new = self.get_policy(M=M_new, goal=w_new, epsilon=epsilon, beta=beta)
            gain = (Q_new.T @ (pi_new - pi_old))[s, s]
        else:
            # if no change in w, this is a shortcut for computing gain
            q_new = M_new[:, s, :] @ self.w
            if self.poltype == 'softmax':
                pi_s_new = softmax(beta * q_new)
            else:
                mask = (q_new == q_new.max())
                greedy = mask / mask.sum()
                pi_s_new = (1 - epsilon) * greedy + (1 / self.action_size) * epsilon * np.ones(self.action_size)
            gain = q_new @ (pi_s_new - pi_old[:, s])
            pi_new = pi_old.copy()
            pi_new[:, s] = pi_s_new

        # compute need according to updated successor representation
        need = pi_new[:, state].T @ M_new[:, state, s]

        # return product of gain and need terms
        return gain * need

    def update(self, current_exp, **kwargs):
        self.experiences.append(current_exp)

        # perform online update of M and w
        m_error = self.update_sr(current_exp, **kwargs)
        w_error = self.update_w(current_exp)

        # list of possible experiences to update on
        unique_exps = list(set(self.experiences))

        for k in range(self.num_recall):
            # compute expected value of backup for every possible experience
            evbs = [{"exp": exp, "evb": self.evb(current_exp[0], exp)} for exp in unique_exps]

            # get experience with the best evb
            best = sorted(evbs, key=lambda x: x["evb"]).pop()
            best_exp = best["exp"]
            self.prioritized_states[best_exp[0]] += 1

            # update successor representation with this experience
            m_error = self.update_sr(best_exp)

        td_error = {'m': np.linalg.norm(m_error), 'w': np.linalg.norm(w_error)}
        return td_error
