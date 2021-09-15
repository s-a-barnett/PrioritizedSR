import numpy as np
import pandas as pd
import os

output_file = 'hyperparameters_four_rooms.csv'

agents  = ['mparsr', 'qparsr', 'dynasr', 'mdq', 'tdsr']
lrs     = [0.1]
num_recalls = [10, 100, 1000, 10000]
seeds = list(range(10))

index = pd.MultiIndex.from_product([agents, lrs, num_recalls, seeds], names=['agent', 'lr', 'num_recall', 'seed'])
df = pd.DataFrame(index = index).reset_index()
df.to_csv(os.path.join('hyperparameters', output_file), index=False)
