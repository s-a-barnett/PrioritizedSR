import os
import sys
import pandas as pd

script = sys.argv[1]
hyp_file = sys.argv[2]
idx = int(sys.argv[3])
output = sys.argv[4]
df = pd.read_csv(hyp_file)
if idx < len(df):
    hyperparameters = df.iloc[idx]
    script = f'python {script} '
    script += ' '.join([f'--{key} {hyperparameters[key]}' for key in df.columns])
    script += f' --output {output}'
    
    os.system(script) 
