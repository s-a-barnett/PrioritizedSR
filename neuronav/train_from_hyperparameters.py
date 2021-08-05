import os
import sys
import pandas as pd

idx = int(sys.argv[1])
df = pd.read_csv('hyperparameters.csv')
if idx < len(df):
    hyperparameters = df.iloc[idx]
    num_recall = hyperparameters['num_recall']
    beta = hyperparameters['beta']
    seed = hyperparameters['seed']
    agent = hyperparameters['agent']
    grid_size = hyperparameters['grid_size']
    
    os.system(f'python train_agent.py --sarsa False --num_recall {num_recall} --beta {beta} --seed {seed} --agent {agent} --grid_size {grid_size}') 
