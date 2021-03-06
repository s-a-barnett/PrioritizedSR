import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .representations import eig
from . import utils
import copy
from itertools import product

def plot_V(Q, env, policy_arrows=True):
    plt.imshow(utils.mask_grid(Q.max(0).reshape(env.grid_size), env.blocks));
    if policy_arrows:
        actions = Q.argmax(0).reshape(env.grid_size)
        coords = {0: (0, 0.5, 0, -0.5), 1: (0, -0.5, 0, 0.5), 2: (0.5, 0, -0.5, 0), 3: (-0.5, 0, 0.5, 0)}
        for i, j in product(range(env.grid_size[0]), range(env.grid_size[1])):
            if [i, j] not in env.blocks:
                x, y, dx, dy = coords[actions[i, j]]
                x += j
                y += i
                plt.arrow(x, y, dx, dy, width=0.1, head_length=0.3, head_starts_at_zero=True, color='r');

def plot_place_fields(M, env):
    M = np.reshape(M, [env.state_size, env.grid_size[0], env.grid_size[1]])
    
    cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    cmap.set_bad(color='white')
    
    plt.figure(1, figsize=(env.grid_size[0]*3, env.grid_size[1]*3))
    for i in range(env.state_size):
        if env.state_to_point(i) not in env.blocks:
            ax = plt.subplot(env.grid_size[0], env.grid_size[1], i+1)
            ax.imshow(utils.mask_grid(M[i, :, :], env.blocks), cmap=cmap)
    
def plot_grid_fields(M, env, nrows=None):
    if nrows is None:
        nrows = int(np.sqrt(env.state_size))

    M_eigs, _ = eig(M)

    # threshold eigs at 0
    M_eigs = M_eigs.real * (M_eigs.real > 0)
    M_eigs = np.reshape(M_eigs.T, [np.minimum(*M_eigs.shape), env.grid_size[0], env.grid_size[1]])
    
    cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    cmap.set_bad(color='white')

    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(env.grid_size[0]*3, env.grid_size[1]*3))

    for i in range(nrows):
        for j in range(nrows):
            k = i*nrows + j
            axes[i][j].imshow(utils.mask_grid(M_eigs[k], env.blocks), cmap=cmap)
