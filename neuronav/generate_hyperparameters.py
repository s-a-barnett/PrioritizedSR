import numpy as np
import pandas as pd

seeds   = list(range(10))
agents  = ['dynaq', 'psq']
resolutions = list(range(1, 12))

index = pd.MultiIndex.from_product([seeds, agents, resolutions], names=['seed', 'agent', 'res'])
df = pd.DataFrame(index = index).reset_index()
df.to_csv('hyperparameters.csv')
