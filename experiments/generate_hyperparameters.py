import numpy as np
import pandas as pd
import os

output_file = 'hyperparameters_rm_detour.csv'
real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

hyp_dict = {
    'agent': ['psq', 'dynaq', 'dynasr', 'mparsr_nosweep', 'qparsr_nosweep'],
    'lr': [0.1],
    'epsilon': [0.1, 0.3, 0.5, 1.0],
    'poltype': ['egreedy'],
    'num_recall': [100],
    'seed': list(range(10)),
    # 'condition': ['control', 'reward', 'transition', 'policy']
}

index = pd.MultiIndex.from_product(hyp_dict.values(), names=hyp_dict.keys())
df = pd.DataFrame(index = index).reset_index()
print(f'number of unique hyperparameter settings: {len(df)}')
df.to_csv(os.path.join(dir_path, 'hyperparameters', output_file), index=False)
