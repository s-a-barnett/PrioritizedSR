import os
import sys
import pandas as pd

hyperparameters = pd.read_csv('hyperparameters.csv').iloc[int(sys.argv[1])]
num_recall = hyperparameters['num_recall']
beta = hyperparameters['beta']
seed = hyperparameters['seed']
agent = hyperparameters['agent']

os.system(f'python train_agent.py --num_recall {num_recall} --beta {beta} --seed {seed} --agent {agent}') 
