import os
import sys
import pandas as pd

script = sys.argv[1]
idx = int(sys.argv[2])
df = pd.read_csv('hyperparameters.csv')
if idx < len(df):
    hyperparameters = df.iloc[idx]
    num_recall = hyperparameters['num_recall']
    beta = hyperparameters['beta']
    seed = hyperparameters['seed']
    agent = hyperparameters['agent']
    grid_size = hyperparameters['grid_size']
    
    os.system(f'python {script} --num_recall {num_recall} --beta {beta} --seed {seed} --agent {agent} --grid_size {grid_size}') 
