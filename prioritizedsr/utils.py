import numpy as np
import numpy.random as npr
import heapq
from collections import defaultdict
from itertools import product
from . import algs

def agent_factory(args, state_size, action_size):
    if args.agent == 'tdq':
        agent = algs.TDQ(state_size, action_size, **vars(args))
    elif args.agent == 'dynaq':
        agent = algs.DynaQ(state_size, action_size, **vars(args))
    elif args.agent == 'dynaqplus':
        agent = algs.DynaQPlus(state_size, action_size, **vars(args))
    elif args.agent == 'psq':
        agent = algs.PSQ(state_size, action_size, **vars(args))
    elif args.agent == 'mdq':
        agent = algs.MDQ(state_size, action_size, online=True, **vars(args))
    elif args.agent == 'tdsr':
        agent = algs.TDSR(state_size, action_size, **vars(args))
    elif args.agent == 'dynasr':
        agent = algs.DynaSR(state_size, action_size, **vars(args))
    elif args.agent == 'dynasrplus':
        agent = algs.DynaSRPlus(state_size, action_size, **vars(args))
    elif args.agent == 'qparsr':
        agent = algs.PARSR(state_size, action_size, goal_pri=True, online=True, **vars(args))
    elif args.agent == 'mparsr':
        agent = algs.PARSR(state_size, action_size, goal_pri=False, online=True, **vars(args))
    elif args.agent == 'qpeparsr':
        agent = algs.PEPARSR(state_size, action_size, goal_pri=True, online=True, **vars(args))
    elif args.agent == 'mpeparsr':
        agent = algs.PEPARSR(state_size, action_size, goal_pri=False, online=True, **vars(args))
    else:
        raise ValueError('Invalid agent type: %s' % args.agent)

    return agent

def onehot(value, max_value):
    vec = np.zeros(max_value)
    vec[value] = 1
    return vec

def twohot(value, max_value):
    vec_1 = np.zeros(max_value)
    vec_2 = np.zeros(max_value)
    vec_1[value[0]] = 1
    vec_2[value[1]] = 1
    return np.concatenate([vec_1, vec_2])

def mask_grid(grid, blocks, mask_value=-100):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if [i,j] in blocks:
                grid[i,j] = mask_value
    grid = np.ma.masked_where(grid == mask_value, grid)
    return grid

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def compute_uniform_sr(env, gamma, goal_pos=None):
    if goal_pos is None:
        goal_pos = env.goal_pos

    T = np.zeros((env.state_size, env.state_size))

    for i, j, a in product(range(env.grid_size[0]), range(env.grid_size[1]), range(env.action_size)):
        if [i, j] not in env.blocks:
            env.reset(agent_pos=[i, j], goal_pos=goal_pos, reward_val=0)
            state = env.observation
            _ = env.step(a)
            state_next = env.observation
            T[state, state_next] += (1 / env.action_size)
        else:
            block_state = env.grid_to_state([i, j])
            T[block_state, block_state] = 1

    M = np.linalg.pinv(np.eye(env.state_size) - gamma * T)
    return M

def run_episode(agent, env, epsilon=0.0, beta=1e6, poltype='softmax', episode_length=None, agent_pos=None, goal_pos=None, stoch_goal_pos=None, reward_val=None, update=True, sarsa=False, pretrain=False):
    if episode_length is None:
        episode_length = 10 * int(np.sqrt(env.state_size))
    env.reset(agent_pos=agent_pos, goal_pos=goal_pos, reward_val=reward_val)
    state = env.observation
    experiences = []
    td_errors = []

    for j in range(episode_length):
        action = agent.sample_action(state, epsilon=epsilon, beta=beta)
        reward = env.step(action)
        state_next = env.observation
        done = env.done
        experiences.append((state, action, state_next, reward, done))
        state = state_next

        if update:
            update_fn = super(type(agent), agent).update if pretrain else agent.update

            if not sarsa:
                td_error = update_fn(experiences[-1])
                td_errors.append(td_error)
            else:
                if (j > 0):
                    td_error = update_fn(experiences[-2], next_exp=experiences[-1])
                    td_errors.append(td_error)
                if done:
                    td_error = update_fn(experiences[-1], next_exp=experiences[-1])
                    td_errors.append(td_error)

        if done:
            break

    return experiences, td_errors

# memory utils

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.key_index = {}
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        _, _, item = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for idx, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[idx]
                self.heap.append((priority, c, i))
                heapq.heapify(self.heap)
                break
            else:
                self.push(item, priority)

# logging utils

class Logger:
    def __init__(self, task_name, config):
        self.task_name = task_name
        self.config = config # dict of training hyperparameters
        self.logs = defaultdict(set)
