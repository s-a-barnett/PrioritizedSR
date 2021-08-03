import numpy as np
import pandas as pd

recalls = [1, 3, 10, 30, 100]
betas   = [0, 5]
seeds   = list(range(10))
agents  = ['nomem', 'dyna', 'ps', 'mps', 'md']

index = pd.MultiIndex.from_product([recalls, betas, seeds, agents], names=['num_recall', 'beta', 'seed', 'agent'])
df = pd.DataFrame(index = index).reset_index()
df.to_csv('hyperparameters.csv')
