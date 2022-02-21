import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import progressbar
from itertools import product
import tqdm

df = pd.read_csv('../experiments/outputs/results_six_rooms.csv')
df.rename(columns={'num_recall': 'num_replay_cycles'}, inplace=True)
df['succ_per_epi'] = df['learns_p3'] / df['num_eps_phase2']
df['reval_prob'] = df['learns_p3']
df.loc[df['condition'] == 'control', 'reval_prob'] = (1 - df['learns_p3'])

new_algs = {
    'dynaq': 'Dyna-Q',
    'dynasr': 'Dyna-SR',
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

conds = ['control', 'reward', 'transition', 'policy']
algs = ['dynaq', 'psq', 'dynasr_old', 'dynasr', 'mparsr_nosweep', 'qparsr_nosweep']

old_algs = dict(zip(new_algs.values(), new_algs.keys()))
old_conds = dict(zip(new_conds.values(), new_conds.keys()))

cells = list(product(new_conds.values(), new_algs.values()))
ps_dict = dict(zip(cells, [np.zeros((4, 11, 11))] * len(cells)))

for idx in tqdm.tqdm(range(len(df))):
    row = df.iloc[idx]
    agent = row['agent']
    if row['num_replay_cycles'] != 1000:
        continue
    else:
        agent_name = old_algs[agent]
    condition = old_conds[row['condition']]
    dirname = agent_name + '_' + row['exp_id']
    ps_loc = os.path.join('../experiments/outputs', dirname, 'prioritized_states2.npy')
    ps = np.load(ps_loc)
    ps_dict[(new_conds[condition], agent)] = 0.5*(ps.mean(0).reshape(4, 11, 11) + ps_dict[(new_conds[condition], agent)])

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6))

for k in range(len(new_algs)):
    i, j = divmod(k, 3)
    alg = list(new_algs.values())[k]
    axs[i][j].imshow(ps_dict[('Transition', alg)].sum(0) / ps_dict[('Transition', alg)].max())
    axs[i][j].grid()
    axs[i][j].set_xticks([])
    axs[i][j].set_yticks([])    
    axs[i][j].set_title(alg)
axs[1][2].remove()

fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
fig.colorbar(mpl.cm.ScalarMappable(), cax=cax)

fig.savefig('figures/1e.png')
