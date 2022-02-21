import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from prioritizedsr import utils
from prioritizedsr.gridworld import SimpleGrid

env = SimpleGrid(11, block_pattern='six_rooms')
env.reset()

grid = 0.4 * np.ones_like(env.grid)
start_pos = (env.onesixth, env.onesixth)
mid_pos = [(env.mid, env.onesixth), (env.onesixth, env.mid)]
goal_pos = [(env.fivesixth, env.onesixth), (env.mid, env.mid), (env.onesixth, env.fivesixth)]

grid[start_pos + (1,)] = 1
for mp in mid_pos:
    grid[mp + (2,)] = 1
for gp in goal_pos:
    grid[gp + (0,)] = 1
for block in env.blocks:
    grid[block[0], block[1]] = 0

plt.imshow(grid)
plt.annotate('1', (0.9, 1.1))
plt.annotate('2', (0.9, 5.1))
plt.annotate('3', (4.9, 1.1))
plt.annotate('4', (0.9, 9.1))
plt.annotate('5', (4.9, 5.1))
plt.annotate('6', (8.9, 1.1))
plt.arrow(3, 1, 0.5, 0, color='w', head_starts_at_zero=True, width=0.1, head_length=0.3)
plt.arrow(1, 3, 0, 0.5, color='w', head_starts_at_zero=True, width=0.1, head_length=0.3)
plt.arrow(7, 1, 0.5, 0, color='w', head_starts_at_zero=True, width=0.1, head_length=0.3)
plt.arrow(1, 7, 0, 0.5, color='w', head_starts_at_zero=True, width=0.1, head_length=0.3)
plt.arrow(3, 5, 0.5, 0, color='w', head_starts_at_zero=True, width=0.1, head_length=0.3)
plt.arrow(5, 3, 0, 0.5, color='w', head_starts_at_zero=True, width=0.1, head_length=0.3)
plt.axis('off')

plt.savefig('figures/1d.png')
