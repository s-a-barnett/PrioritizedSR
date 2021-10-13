import numpy as np
import pandas as pd
import os

output_file = 'hyperparameters_six_rooms.csv'
real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

agents  = ['mparsr', 'qparsr', 'dynasr', 'tdsr', 'psq', 'dynaq', 'tdq']
lrs     = [0.3]
num_recalls = [1, 3, 10, 30, 100]
seeds = list(range(10))
betas = [0, 5]
conditions = ['control', 'reward', 'transition', 'policy']

index = pd.MultiIndex.from_product([agents, lrs, num_recalls, seeds, conditions, betas], names=['agent', 'lr', 'num_recall', 'seed', 'condition', 'beta'])
df = pd.DataFrame(index = index).reset_index()
df.to_csv(os.path.join(dir_path, 'hyperparameters', output_file), index=False)
