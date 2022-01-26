import numpy as np
import numpy.random as npr
from scipy.special import softmax
from collections import defaultdict
import random
from . import utils

class TDQ:
    def __init__(self, state_size, action_size, lr=1e-1, gamma=0.99, poltype='softmax', Q_init=None, **kwargs):
        self.state_size = state_size
        self.action_size = action_size

        if Q_init is None:
            self.Q = np.zeros((action_size, state_size))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size)
        else:
            self.Q = Q_init

        self.lr = lr
        self.gamma = gamma
        self.prioritized_states = np.zeros((action_size, state_size), dtype=np.int)
        self.num_updates = 0
        self.poltype = poltype
        
    def sample_action(self, state, epsilon=0.0, beta=1e6):
        Qs = self.Q[:, state]
        if self.poltype == 'softmax':
            action = npr.choice(self.action_size, p=softmax(beta * Qs))
        else:
            if npr.rand() < epsilon:
                action = npr.choice(self.action_size)
            else:
                action = npr.choice(np.flatnonzero(np.isclose(Qs, Qs.max())))
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
            self.Q[s_a, s] += self.lr * q_error
        return q_error

    def update(self, current_exp, **kwargs):
        q_error = self.update_q(current_exp, **kwargs)
        self.num_updates += 1
        td_error = {'q': np.linalg.norm(q_error)}
        return td_error

    def get_policy(self, epsilon=0.0, beta=1e6):
        if self.poltype == 'softmax':
            policy = softmax(beta * agent.Q, axis=0)
        else:
            mask = (agent.Q == agent.Q.max(0))
            greedy = mask / mask.sum(0)
            policy = (1 - epsilon) * greedy + (1 / self.action_size) * epsilon * np.ones((self.action_size, self.state_size))
        return policy

    @property
    def V(self):
        Q = self.Q
        return Q.max(0)

class DynaQ(TDQ):

    def __init__(self, state_size, action_size, num_recall=10, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.model = {}

    def _sample_model(self):
        # sample state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[npr.choice(len(past_states))]
        # sample action previously taken from sampled state
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = past_actions[npr.choice(len(past_actions))]
        # get reward, state_next, done, and make exp
        exp = (sampled_state, sampled_action) + self.model[(sampled_state, sampled_action)][1]
        
        return exp
        
    def update(self, current_exp, **kwargs):
        # perform online update first
        td_error = super().update(current_exp, **kwargs)

        state, action, next_state, reward, done = current_exp

        # update (deterministic) model
        self.model[(state, action)] = self.num_updates, (next_state, reward, done)

        for i in range(self.num_recall):
            exp = self._sample_model()
            self.prioritized_states[exp[1], exp[0]] += 1
            super().update(exp)

        return td_error

class DynaQPlus(DynaQ):
    
    def __init__(self, state_size, action_size, num_recall=10, kappa=1e-4, **kwargs):
        super().__init__(state_size, action_size, num_recall=num_recall, **kwargs)
        self.kappa = kappa

    def _sample_model(self):
        # sample a state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[npr.choice(len(past_states))]

        # sample action from any possible action
        possible_actions = np.arange(self.action_size)
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = possible_actions[npr.choice(len(possible_actions))]
        if sampled_action not in past_actions:
            # initial model that action leads to same state with reward zero
            reward = 0
            next_state = sampled_state
            done = False
            # add state-action to model
            self.model[(sampled_state, sampled_action)] = 1, (next_state, reward, done)

        t_last_update, (next_state, reward, done) = self.model[(sampled_state, sampled_action)]

        # bonus reward for actions not tried in a while
        reward += self.kappa * np.sqrt(self.num_updates - t_last_update)

        exp = (sampled_state, sampled_action, next_state, reward, done)
        return exp

class PSQ(TDQ):

    def __init__(self, state_size, action_size, num_recall=10, theta=1e-6, **kwargs):
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
        if priority >= self.theta:
            self.pqueue.push((state, action), -priority)

        for k in range(self.num_recall):
            if self.pqueue.is_empty():
                break

            # get highest priority experience
            state, action = self.pqueue.pop()
            self.prioritized_states[action, state] += 1
            exp = (state, action) + self.model[(state, action)]

            # update Q based on this experience
            q_error = self.update_q(exp)

            for s, a in self.predecessors[state]:
                # add predecessors to priority queue
                exp_pred = (s, a) + self.model[(s, a)]
                q_error = self.update_q(exp_pred, prospective=True)
                priority = self.priority(q_error)
                if priority >= self.theta:
                    self.pqueue.push((s, a), -priority)

        td_error = {'q': np.linalg.norm(q_error)}
        return td_error

class MDQ(TDQ):
    def __init__(self, state_size, action_size, num_recall=10, online=False, set_need_to_one=False, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.online = online
        self.T = np.identity(state_size)
        self.experiences = []
        self.set_need_to_one = set_need_to_one
        self.gain_array = np.zeros((state_size, num_recall, state_size, action_size))

    def update_T(self, current_exp):
        state = current_exp[0]
        state_next = current_exp[2]
        T_error = utils.onehot(state_next, self.state_size) - self.T[state]
        self.T[state] += 0.9 * T_error # hard-coded learning rate
        self.T[state] /= self.T[state].sum() # renormalize T
        return T_error

    def update_M(self):
        self.M = np.linalg.pinv(np.eye(self.state_size) - self.gamma * self.T)

    def evb(self, state, exp, idx_recall, beta=5.0):
        s = exp[0]
        a = exp[1]

        # record pi_old as a baseline
        Q_s_old = self.Q[:, s]
        pi_s_old = softmax(beta * Q_s_old)

        # compute new Q based on experience to be evaluated
        Q_s_new = Q_s_old.copy()
        Q_s_new[a] += self.lr * self.update_q(exp, prospective=True)

        # compute gain term
        pi_s_new = softmax(beta * Q_s_new)
        gain = np.dot(Q_s_new, pi_s_new - pi_s_old)
        gain = np.maximum(gain, 1e-10) # minimum gain

        # record gain term in array for subsequent visualization
        self.gain_array[state, idx_recall, s, a] = gain

        # compute need term
        if not self.set_need_to_one:
            need = self.M[state, s]
        else:
            need = 1.0

        # return product of gain and need terms
        return gain * need

    def update_q_nstep(self, exps, prospective=False):
        n = len(exps)
        rewards = np.array([exp[3] for exp in exps]) 
        discounts = np.array([self.gamma ** i for i in range(n)])
        nstep_reward = np.dot(rewards, discounts)
        end_state = exps[-1][2]
        nstep_reward += (self.gamma ** n) * self.Q[:, end_state].max()
        st = exps[0][0]; action = exps[0][1]
        q_error = nstep_reward - self.Q[action, st]
        if not prospective:
            self.Q[action, st] += self.lr * q_error
        return q_error

    def multistep_evb(self, state, exps, idx_recall, beta=5.0):
        gains = np.zeros(len(exps))

        for i in range(len(exps)):
            s = exps[i][0]
            a = exps[i][1]

            # record pi_old as a baseline
            Q_s_old = self.Q[:, s]
            pi_s_old = softmax(beta * Q_s_old)

            # compute new Q based on experience to be evaluated
            Q_s_new = Q_s_old.copy()
            Q_s_new[a] += self.lr * self.update_q_nstep(exps[i:], prospective=True)

            # compute gain term
            pi_s_new = softmax(beta * Q_s_new)
            gain = np.dot(Q_s_new, pi_s_new - pi_s_old)
            gain = np.maximum(gain, 1e-10) # minimum gain
            gains[i] = gain

            # record gain term in array for subsequent visualization
            self.gain_array[state, idx_recall, s, a] += gain

        # use need from last appended state
        need = self.M[state, exps[-1][0]]

        # evb is sum-product of gain and need terms
        return np.sum(need * gains)

    def update(self, current_exp, **kwargs):
        self.experiences.append(current_exp)
        state = current_exp[0]

        # perform online update of Q, T, and M
        q_error = self.update_q(current_exp, prospective=(not self.online), **kwargs)
        T_error = self.update_T(current_exp)
        self.update_M()

        # list of possible experiences to update on
        unique_exps = list(set(self.experiences))
        # filter out 'same state' experiences
        unique_exps = [exp for exp in unique_exps if exp[0] != exp[2]]

        self.recalled = []

        for k in range(self.num_recall):
            # don't loop if there aren't any experiences
            if not unique_exps:
                break

            # compute expected value of backup for every possible experience
            evbs = np.array([self.evb(state, exp, k) for exp in unique_exps])

            # get single experience with the best evb (random tie-break)
            evb_max = evbs.max()
            best_exps = [unique_exps[i] for i in range(len(unique_exps)) if evbs[i] == evb_max]
            best_exp = [random.choice(best_exps)]

            if k > 0:
                # get previously recalled experience
                prev_exp = self.recalled[-1]
                # find experience with next optimal action
                final_state = prev_exp[-1][2]
                Qs = self.Q[:, final_state]
                next_action = npr.choice(np.flatnonzero(np.isclose(Qs, Qs.max())))
                cand_exps = [exp for exp in self.experiences if (exp[0] == final_state) and (exp[1] == next_action)]
                if not cand_exps:
                    # skip if no candidate experience with corresponding state and action exists
                    #   applies to terminal states
                    continue
                next_exp = cand_exps[-1] # use most recent compatible experience, if multiple
                # append to previously recalled exp, compute evb
                nstep_exp = prev_exp + [next_exp]
                nstep_evb = self.multistep_evb(state, nstep_exp, k)
                # use as best exp if better than best single backup
                if nstep_evb > evb_max:
                    best_exp = nstep_exp

            for i in range(len(best_exp)):
                q_error = self.update_q_nstep(best_exp[i:])
                self.prioritized_states[best_exp[i][1], best_exp[i][0]] += 1

            self.recalled.append(best_exp)

        td_error = {'q': np.linalg.norm(q_error), 'T': np.linalg.norm(T_error)}
        return td_error

class TDSR:

    def __init__(self, state_size, action_size, lr=1e-1, gamma=0.99, M_init=None, poltype='softmax', weights='direct', **kwargs):
        self.state_size = state_size
        self.action_size = action_size

        self.poltype = poltype
        self.weights = weights

        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack([M_init * npr.randn(state_size, state_size) for i in range(action_size)])
        else:
            self.M = M_init
        
        self.w = np.zeros(state_size)
        self.lr = lr
        self.gamma = gamma
        self.prioritized_states = np.zeros((action_size, state_size), dtype=np.int)
        self.num_updates = 0

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
                action = npr.choice(np.flatnonzero(np.isclose(Qs, Qs.max())))
        return action

    def update_w(self, current_exp):
        s, a, s_1, r, _ = current_exp
        if self.weights == 'direct':
            error = r - self.w[s_1]
            self.w[s_1] += self.lr * error
        elif self.weights == 'td':
            Vs = self.Q_estimates(s).max()
            Vs_1 = self.Q_estimates(s_1).max()
            delta = r + self.gamma * Vs_1 - Vs
            # epsilon and beta are hard-coded, need to improve this
            M = self.get_M_states(epsilon=1e-1, beta=5)
            error = delta * M[s]
            self.w += self.lr * error
        return np.linalg.norm(error)

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
            self.M[s_a, s, :] += self.lr * m_error
        return m_error

    def update(self, current_exp, **kwargs):
        m_error = self.update_sr(current_exp, **kwargs)
        w_error = self.update_w(current_exp)
        self.num_updates += 1
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

    @property
    def Q(self):
        return self.M @ self.w

    @property
    def V(self):
        Q = self.Q
        return Q.max(0)

class DynaSR(TDSR):
    
    def __init__(self, state_size, action_size, num_recall=10, recency='exponential', **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.model = {}
        self.recency = recency

    def _sample_model(self):
        # sample state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[npr.choice(len(past_states))]
        # sample action previously taken from sampled state
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = past_actions[npr.choice(len(past_actions))]
        key = (sampled_state, sampled_action)
        # get reward, state_next, done, and make exp
        if self.recency == 'exponential':
            successors = self.model[key][1]
            idx = np.minimum(len(successors)-1, int(npr.exponential(scale=5)))
            successor = successors[::-1][idx]
        else:
            successor = self.model[key][1]
        exp = key + successor 
        
        return exp

    def update(self, current_exp, **kwargs):
        # perform online update first
        td_error = super().update(current_exp, **kwargs)

        state, action, next_state, reward, done = current_exp

        # update (deterministic) model
        key = (state, action)
        value = (next_state, reward, done)
        if self.recency == 'exponential':
            if key in self.model:
                successors = self.model[key][1]
                successors.append(value)
                # shorten successors to capture >99% of probability mass
                successors = successors[-25:]
                self.model[key] = self.num_updates, successors
            else:
                self.model[key] = self.num_updates, [value]
        else:
            self.model[key] = self.num_updates, value

        for i in range(self.num_recall):
            exp = self._sample_model()
            self.prioritized_states[exp[1], exp[0]] += 1
            m_error = self.update_sr(exp, **kwargs)
            self.num_updates += 1

        return td_error

class DynaSRPlus(DynaSR):
    
    def __init__(self, state_size, action_size, num_recall=10, kappa=1e-4, **kwargs):
        super().__init__(state_size, action_size, num_recall=num_recall, **kwargs)
        self.kappa = kappa

    def _sample_model(self):
        # sample a state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[npr.choice(len(past_states))]

        # sample action from any possible action
        possible_actions = np.arange(self.action_size)
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = possible_actions[npr.choice(len(possible_actions))]
        key = (sampled_state, sampled_action)
        if sampled_action not in past_actions:
            # initial model that action leads to same state with reward zero
            reward = 0
            next_state = sampled_state
            done = False
            # add state-action to model
            value = (next_state, reward, done)
            if self.recency == 'exponential':
                value = [value]
            self.model[(sampled_state, sampled_action)] = 1, value

        if self.recency == 'exponential':
            t_last_update, successors = self.model[key]
            idx = np.minimum(len(successors)-1, int(npr.exponential(scale=5)))
            successor = successors[::-1][idx]
            (next_state, reward, done) = successor
        else:
            t_last_update, (next_state, reward, done) = self.model[key]

        # bonus reward for actions not tried in a while
        reward += self.kappa * np.sqrt(self.num_updates - t_last_update)

        exp = (sampled_state, sampled_action, next_state, reward, done)
        return exp

class PARSR(TDSR):

    def __init__(self, state_size, action_size, num_recall=10, theta=1e-6, goal_pri=True, online=False, sweeping=True, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.theta = theta
        self.goal_pri = goal_pri
        self.online = online
        self.sweeping = sweeping
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

    def _sample_model(self):
        # sample state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[npr.choice(len(past_states))]
        # sample action previously taken from sampled state
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = past_actions[npr.choice(len(past_actions))]
        key = (sampled_state, sampled_action)
        # get reward, state_next, done, and make exp
        successor = self.model[key]
        exp = key + successor 
        
        return exp
    
    def update(self, current_exp, **kwargs):
        state, action, next_state, reward, done = current_exp

        # update (deterministic) model and state predecessors
        self.model[(state, action)] = (next_state, reward, done)
        if self.sweeping:
            self._update_predecessors(state, action, next_state)

        # compute value of update
        m_error = self.update_sr(current_exp, prospective=(not self.online), **kwargs)
        w_error = self.update_w(current_exp) if self.online else 0.0

        # compute priority for the update, add to queue
        priority = self.priority(m_error, current_exp)
        if priority >= self.theta:
            self.pqueue.push((state, action), -priority)

        self.recalled = []

        for k in range(self.num_recall):
            if self.pqueue.is_empty():
                exp = self._sample_model()
            else:
                # get highest priority experience
                state, action = self.pqueue.pop()
                exp = (state, action) + self.model[(state, action)]
            self.prioritized_states[exp[1], exp[0]] += 1
            self.recalled.append([exp])

            # update M and w based on this experience
            m_error = self.update_sr(exp)
            # w_error = self.update_w(exp)

            if self.sweeping:
                for s, a in self.predecessors[state]:
                    # add predecessors to priority queue
                    exp_pred = (s, a) + self.model[(s, a)]
                    m_error = self.update_sr(exp_pred, prospective=(not self.online))
                    # w_error = self.update_w(current_exp)
                    priority = self.priority(m_error, exp_pred)
                    if priority >= self.theta:
                        self.pqueue.push((s, a), -priority)

        td_error = {'m': np.linalg.norm(m_error), 'w': np.linalg.norm(w_error)}
        return td_error

class PEPARSR(TDSR):
    
    def __init__(self, state_size, action_size, num_recall=10, pri_strength=5, bias=0, goal_pri=True, online=False, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.num_recall = num_recall
        self.pri_strength = pri_strength
        self.bias = bias
        self.goal_pri = goal_pri
        self.online = online
        self.model = {}
        self.priorities = {}

    def priority(self, m_error, current_exp):
        if self.goal_pri:
            # priority given by temporal difference of Q
            error = np.dot(m_error, self.w) - self.w[current_exp[0]] + current_exp[3]
        else:
            # priority given by temporal difference of successor representation
            error = m_error
        return np.linalg.norm(error)

    def _sample_model(self):
        sa_pairs_all = list(self.model.keys())
        N = len(sa_pairs_all)
        pri_vals = np.array(list(self.priorities.values()))
        pri_vals += 1e-10
        weights = np.power(N * pri_vals, -self.bias)
        weights /= weights.max()
        p = softmax(self.pri_strength * pri_vals)
        # p = pri_vals ** self.pri_strength
        # p /= p.sum()
        sampled_idxs = npr.choice(len(sa_pairs_all), p=p, replace=True, size=self.num_recall)
        sa_pairs = [sa_pairs_all[i] for i in sampled_idxs]
        exps = [sa + self.model[sa] for sa in sa_pairs]
        sampled_weights = [weights[i] for i in sampled_idxs]
        return exps, sampled_weights

    def update(self, current_exp, **kwargs):
        state, action, next_state, reward, done = current_exp

        # update (deterministic) model
        self.model[(state, action)] = (next_state, reward, done)

        # compute value of update
        m_error = self.update_sr(current_exp, prospective=(not self.online), **kwargs)
        w_error = self.update_w(current_exp) if self.online else 0.0

        # compute priority for the update, add to priorities
        priority = self.priority(m_error, current_exp)
        self.priorities[(state, action)] = priority

        self.recalled = []
        exps, weights = self._sample_model()

        for exp, weight in zip(exps, weights):
            self.recalled.append([exp])

            # compute importance weight
            self.lr *= weight

            # update M and w
            m_error = self.update_sr(exp)
            # w_error = self.update_w(exp)
            self.lr /= weight

        td_error = {'m': np.linalg.norm(m_error), 'w': np.linalg.norm(w_error)}
        return td_error

# DEPRECATED ALGORITHM
# class MDSR(TDSR):
# 
#     def __init__(self, state_size, action_size, num_recall, online=False, **kwargs):
#         super().__init__(state_size, action_size, **kwargs)
#         self.num_recall = num_recall
#         self.experiences = []
#         self.online = online
# 
#     def evb(self, state, exp, epsilon=0.0, beta=5.0):
#         s = exp[0]
#         s_a = exp[1]
#         s_1 = exp[2]
#         r = exp[3]
# 
#         # compute new M and w based on experience to be evaluated
#         M_new = self.M.copy()
#         M_new[s_a, s, :] += self.lr * self.update_sr(exp, prospective=True)
#         w_new = self.w.copy()
#         w_error = r - w_new[s_1]
#         w_new[s_1] += self.lr * w_error
# 
#         # get old policy as a baseline
#         pi_old = self.get_policy(epsilon=epsilon, beta=beta)
# 
#         # compute gain term
#         if w_error != 0:
#             Q_new = M_new @ w_new
#             pi_new = self.get_policy(M=M_new, goal=w_new, epsilon=epsilon, beta=beta)
#             gain = (Q_new.T @ (pi_new - pi_old))[s, s]
#         else:
#             # if no change in w, this is a shortcut for computing gain
#             q_new = M_new[:, s, :] @ self.w
#             if self.poltype == 'softmax':
#                 pi_s_new = softmax(beta * q_new)
#             else:
#                 mask = (q_new == q_new.max())
#                 greedy = mask / mask.sum()
#                 pi_s_new = (1 - epsilon) * greedy + (1 / self.action_size) * epsilon * np.ones(self.action_size)
#             gain = q_new @ (pi_s_new - pi_old[:, s])
#             pi_new = pi_old.copy()
#             pi_new[:, s] = pi_s_new
# 
#         # compute need according to updated successor representation
#         need = pi_new[:, state].T @ M_new[:, state, s]
# 
#         # return product of gain and need terms
#         return gain * need
# 
#     def update(self, current_exp, **kwargs):
#         self.experiences.append(current_exp)
# 
#         # perform online update of M and w
#         m_error = self.update_sr(current_exp, prospective=(not self.online), **kwargs)
#         w_error = self.update_w(current_exp) if self.online else 0.0
# 
#         # list of possible experiences to update on
#         unique_exps = list(set(self.experiences))
# 
#         self.recalled = []
# 
#         for k in range(self.num_recall):
#             # compute expected value of backup for every possible experience
#             evbs = [{"exp": exp, "evb": self.evb(current_exp[0], exp)} for exp in unique_exps]
# 
#             # get experience with the best evb
#             best = sorted(evbs, key=lambda x: x["evb"]).pop()
#             best_exp = best["exp"]
#             self.prioritized_states[best_exp[0]] += 1
#             self.recalled.append([best_exp])
# 
#             # update successor representation with this experience
#             m_error = self.update_sr(best_exp)
# 
#         td_error = {'m': np.linalg.norm(m_error), 'w': np.linalg.norm(w_error)}
#         return td_error
