import numpy as np
import pandas as pd

agents  = ['dynasr', 'mparsr', 'qparsr', 'mdq']
lrs     = [0.1, 0.3, 0.5, 0.7, 0.9]
num_recalls = [100, 300, 1000, 3000, 10000]
seeds = list(range(10))

index = pd.MultiIndex.from_product([agents, lrs, num_recalls, seeds], names=['agent', 'lr', 'num_recall', 'seed'])
df = pd.DataFrame(index = index).reset_index()
df.to_csv('hyperparameters.csv')
