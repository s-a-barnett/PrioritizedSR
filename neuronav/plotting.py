import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from representations import eig
import utils
import copy

def plot_place_fields(agent, env, epsilon=1.0, beta=3.0):
    M = agent.get_M_states(epsilon=epsilon, beta=beta).copy()
    M = np.reshape(M, [env.state_size, env.grid_size, env.grid_size])
    
    cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    cmap.set_bad(color='white')
    
    plt.figure(1, figsize=(env.grid_size*3, env.grid_size*3))
    for i in range(env.state_size):
        if env.state_to_point(i) not in env.blocks:
            ax = plt.subplot(env.grid_size, env.grid_size, i+1)
            ax.imshow(utils.mask_grid(M[i, :, :], env.blocks), cmap=cmap)
    
def plot_grid_fields(agent, env, online=False, epsilon=1.0, beta=3.0, nrows=None):
    if nrows is None:
        nrows = int(np.sqrt(env.state_size))
    if online:
        M_eigs = agent.eigs
    else:
        M_eigs, _ = eig(agent.get_M_states(epsilon=epsilon, beta=beta).copy())

    # threshold eigs at 0
    M_eigs = M_eigs.real * (M_eigs.real > 0)
    M_eigs = np.reshape(M_eigs.T, [np.minimum(*M_eigs.shape), env.grid_size, env.grid_size])
    
    cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    cmap.set_bad(color='white')

    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(env.grid_size*3, env.grid_size*3))

    for i in range(nrows):
        for j in range(nrows):
            k = i*nrows + j
            axes[i][j].imshow(utils.mask_grid(M_eigs[k, :, :], env.blocks), cmap=cmap)
