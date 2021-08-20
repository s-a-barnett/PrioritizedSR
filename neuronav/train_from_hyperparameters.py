import os
import sys
import pandas as pd

script = sys.argv[1]
idx = int(sys.argv[2])
df = pd.read_csv('hyperparameters.csv')
if idx < len(df):
    hyperparameters = df.iloc[idx]
    seed = hyperparameters['seed']
    agent = hyperparameters['agent']
    res = hyperparameters['res']
    
    os.system(f'python {script} --seed {seed} --agent {agent} --res {res}') 
