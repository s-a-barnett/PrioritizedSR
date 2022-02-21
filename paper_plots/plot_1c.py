import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import progressbar
from itertools import product
import networkx as nx
import tqdm
sys.path.append(os.path.join(os.getcwd(), '..'))
from prioritizedsr.gridworld import Sequential

df = pd.read_csv('../experiments/outputs/results_sequential.csv')
df.rename(columns={'num_recall': 'num_replay_cycles'}, inplace=True)
df['succ_per_epi'] = df['learns_p3'] / df['num_eps_phase2']
df['reval_prob'] = df['learns_p3']
df.loc[df['condition'] == 'control', 'reval_prob'] = (1 - df['learns_p3'])

new_algs = {
        'dynaq': 'Dyna-Q',
            'dynasr_old': 'Dyna-SR',
                'psq': 'PS',
                    'mparsr_nosweep': 'M-PARSR',
                        'qparsr_nosweep': 'Q-PARSR'
}

new_conds = {
        'reward': 'Reward',
            'transition': 'Transition',
                'policy': 'Policy',
                    'control': 'Control'
}

df = df.replace({'agent': new_algs, 'condition': new_conds})

sns.set(font_scale = 2)
g = sns.catplot(x='condition', y='reval_prob',
               col='agent', col_wrap=3,
               data=df[df['num_replay_cycles'] == 10],
               kind='bar',
               order=list(new_conds.values()),
               col_order=list(new_algs.values()))

g.set_titles('{col_name}')
g.set_xlabels('Condition')
g.set(ylim=(0,1))
g.set_xticklabels(rotation=45)
g.set_ylabels('Revaluation score')

g.savefig('figures/1c.png')
