import numpy as np
import numpy.random as npr

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

def memory_update(exp, agent, epsilon, beta):
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
    return [exp for exp in experiences if (exp[2] == state)]

def queue_append(exp, priority, queue):
    
    already_in_queue = False
    # if exp is already in queue with a lower priority, replace with higher priority
    for mem in queue:
        if (mem["exp"] == exp):
            already_in_queue = True
            if (mem["priority"] < priority):
                mem["priority"] = priority
                                        
    if not already_in_queue:
        queue.append({"exp": exp, "priority": priority})
                                                    
    return queue
