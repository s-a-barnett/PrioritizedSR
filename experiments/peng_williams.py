import numpy as np
import numpy.random as npr
import progressbar
from itertools import product
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prioritizedsr import algs, utils
from prioritizedsr.gridworld import SimpleGrid

# paramaters fixed from original experiment
epsilon = 0.1
poltype = 'egreedy'
alpha = 0.5
theta = 1e-4
reward_val = 100
num_recall = 5
gamma = 0.95

MAX_TRAINING_EPISODES = int(1e5)

def agent_factory(args, state_size, action_size):
    if args.agent == 'dynaq':
        agent = algs.DynaQ(state_size, action_size, num_recall=num_recall, learning_rate=alpha, gamma=gamma, poltype='egreedy')
    elif args.agent == 'psq':
        agent = algs.PSQ(state_size, action_size, num_recall=num_recall, learning_rate=alpha, gamma=gamma, theta=theta, poltype='egreedy')
    else:
        raise ValueError('Invalid agent type: %s' % args.agent)

    return agent

def main(args):
    npr.seed(args.seed)
    grid_size = (6 * args.res, 9 * args.res)
    agent_pos = [2 * args.res, 0]
    goal_pos = [[x, 8*args.res + y] for (x, y) in product(range(args.res), range(args.res))]
    optimal_episode_length = 13 * args.res + 1

    env = SimpleGrid(grid_size, block_pattern='sutton')
    num_backups = 0
    agent = agent_factory(args, env.state_size, env.action_size)

    for ep in progressbar.progressbar(range(MAX_TRAINING_EPISODES)):
        experiences_train, _ = utils.run_episode(agent, env, epsilon=epsilon, poltype=poltype, agent_pos=agent_pos, goal_pos=goal_pos, reward_val=reward_val)

        num_backups += len(experiences_train)

        experiences_test, _ = utils.run_episode(agent, env, epsilon=0.0, poltype=poltype, agent_pos=agent_pos, goal_pos=goal_pos, reward_val=reward_val, update=False)

        if len(experiences_test) == optimal_episode_length:
            break

    if args.agent == 'dynaq':
        num_backups *= num_recall + 1

    with open(args.output, 'a') as f:
        f.write(','.join([str(args.agent), str(args.res), str(args.seed), str(num_backups)]) + '\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--res', type=int, default=1, help='grid resolution')
    parser.add_argument('--agent', type=str, default='dynaq', help='algorithm')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    main(args)

