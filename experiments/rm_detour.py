import numpy as np
import numpy.random as npr
import progressbar
import argparse
import itertools
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prioritizedsr import algs, utils
from prioritizedsr.gridworld import SimpleGrid

# fixed parameters
gamma = 0.95
beta  = 5

NUM_RUNS = 10
NUM_INTRO = 100
NUM_TRAINING_EPISODES = 100
EPISODE_LENGTH = 25000

def agent_factory(args, state_size, action_size):
    if args.agent == 'dynasr':
        agent = algs.DynaSR(state_size, action_size, num_recall=args.num_recall, learning_rate=args.lr, gamma=gamma)
    elif args.agent == 'qparsr':
        agent = algs.PARSR(state_size, action_size, num_recall=args.num_recall, learning_rate=args.lr, gamma=gamma, goal_pri=True, online=True)
    elif args.agent == 'mparsr':
        agent = algs.PARSR(state_size, action_size, num_recall=args.num_recall, learning_rate=args.lr, gamma=gamma, goal_pri=False, online=True)
    elif args.agent == 'mdq':
        agent = algs.MDQ(state_size, action_size, num_recall=args.num_recall, learning_rate=args.lr, gamma=gamma)
    else:
        raise ValueError('Invalid agent type: %s' % args.agent)

    return agent

def test_from_Q(Q, env, ap, gp, rw, args):
    agent_test = algs.TDQ(env.state_size, env.action_size, learning_rate=args.lr, gamma=gamma, Q_init=Q)
    experiences, _ = utils.run_episode(agent_test, env, agent_pos=ap, goal_pos=gp, reward_val=rw, update=False)
    return len(experiences)

def main(args):
    npr.seed(args.seed)
    r = args.res
    grid_size = (10*r, 10*r)
    agent_pos = [4*r, 0]
    goal_pos = [[4*r + x, 9*r + y] for x, y in itertools.product(range(r), range(r))]
    reward_val = 10
    optimal_episode_lengths = {'before': 9*r, 'after': 11*r + 2}

    env = SimpleGrid(grid_size, block_pattern='detour_before')
    Qs_before = np.zeros((NUM_RUNS, env.action_size, env.state_size))
    Qs_after = np.zeros_like(Qs_before)

    for i in progressbar.progressbar(range(NUM_RUNS)):
        agent = agent_factory(args, env.state_size, env.action_size)

        # train on original task
        for j in range(NUM_TRAINING_EPISODES):
            _, _ = utils.run_episode(agent, env, beta=beta, agent_pos=agent_pos, goal_pos=goal_pos, reward_val=reward_val, episode_length=EPISODE_LENGTH)

        # record Q value for original task
        Qs_before[i] = agent.Q.copy()

        # introduce wall
        env = SimpleGrid(grid_size, block_pattern='detour_after')

        for j in range(NUM_INTRO):
            for x in range(r):
                env.reset(agent_pos=[4*r + x, 5*r -1], goal_pos=goal_pos, reward_val=reward_val)
                state = env.observation
                action = 3
                reward = env.step(action)
                state_next = env.observation
                done = env.done
                agent.update((state, action, state_next, reward, done))

        # record Q value for detour
        Qs_after[i] = agent.Q.copy()
        
    # get run lengths for each of these tasks
    before_lengths = np.array([test_from_Q(Qs_before[i], env, agent_pos, goal_pos, reward_val, args) for i in range(NUM_RUNS)])
    after_lengths = np.array([test_from_Q(Qs_after[i], env, agent_pos, goal_pos, reward_val, args) for i in range(NUM_RUNS)])

    learns_before = (before_lengths == optimal_episode_lengths['before'])
    learns_after  = (after_lengths  == optimal_episode_lengths['after'])
    learns_both   = learns_before * learns_after

    with open(args.output, 'a') as f:
        f.write(','.join([str(args.agent), str(args.seed), str(args.res), str(args.lr), str(args.num_recall), str(np.mean(learns_before)), str(np.mean(learns_after)), str(np.mean(learns_both))]) + '\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--res', type=int, default=1, help='resolution of gridworld')
    parser.add_argument('--num_recall', type=int, default=10000, help='number of recalls')
    parser.add_argument('--agent', type=str, default='dynasr', help='algorithm')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    main(args)

