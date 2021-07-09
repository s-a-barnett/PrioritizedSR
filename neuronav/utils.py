import numpy as np
import numpy.random as npr
import heapq

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

# memory utils

def memory_update(exp, agent, epsilon=0, beta=1e6):
    exp1 = exp.copy()
    if not exp[-1]:
        # change to "best" action in hindsight
        exp1[1] = agent.sample_action(exp[0], epsilon=epsilon, beta=beta)
    td_sr = agent.update_sr(exp, exp1)
    return td_sr

def get_dyna_indices(experiences, weights, nsamples):
    p = exp_normalize(np.array(weights))
    p /= p.sum()
    return npr.choice(len(experiences), nsamples, p=p, replace=True)

def get_predecessors(state, experiences):
    preds_nonunique = [tuple(exp) for exp in experiences if (exp[2] == state)]
    preds_unique = list(set(preds_nonunique))
    return [list(pred) for pred in preds_unique]

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

