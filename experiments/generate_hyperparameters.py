import numpy as np
import pandas as pd
import os

output_file = 'hyperparameters_rm.csv'

agents  = ['mparsr', 'qparsr', 'dynasr', 'mdq', 'tdsr']
lrs     = [0.1]
num_recalls = [1000, 3000, 10000, 30000, 100000]
seeds = list(range(10))
ress = [1, 2, 3, 4]
betas = [0, 5]

index = pd.MultiIndex.from_product([agents, lrs, num_recalls, seeds, ress, betas], names=['agent', 'lr', 'num_recall', 'seed', 'res', 'beta'])
df = pd.DataFrame(index = index).reset_index()
df.to_csv(os.path.join('hyperparameters', output_file), index=False)
