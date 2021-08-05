import numpy as np
import pandas as pd

recalls = [1, 3, 10, 30, 100]
betas   = [5]
seeds   = list(range(10))
agents  = ['tdq', 'dynaq', 'dynaqplus', 'psq', 'tdsr', 'dynasr', 'dynasrplus', 'pssr', 'mpssr']
grid_sizes = list(range(7, 17, 2))

index = pd.MultiIndex.from_product([recalls, betas, seeds, agents, grid_sizes], names=['num_recall', 'beta', 'seed', 'agent', 'grid_size'])
df = pd.DataFrame(index = index).reset_index()
df.to_csv('hyperparameters.csv')
