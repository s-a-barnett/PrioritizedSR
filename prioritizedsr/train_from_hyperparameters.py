import os
import sys
import pandas as pd

script = sys.argv[1]
idx = int(sys.argv[2])
df = pd.read_csv('hyperparameters.csv')
if idx < len(df):
    hyperparameters = df.iloc[idx]
    agent = hyperparameters['agent']
    seed = hyperparameters['seed']
    lr = hyperparameters['lr']
    num_recall = hyperparameters['num_recall']
    
    os.system(f'python {script} --agent {agent} --seed {seed} --lr {lr} --num_recall {num_recall} --output results_policy_reval_online.csv') 
