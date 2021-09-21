import numpy as np
import numpy.random as npr
import progressbar
import argparse
import itertools
import uuid
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prioritizedsr import algs, utils
from prioritizedsr.gridworld import SimpleGrid

NUM_RUNS = 10
NUM_INTRO = 100
NUM_TRAINING_EPISODES = 100
EPISODE_LENGTH = 25000

def agent_factory(args, state_size, action_size):
    if args.agent == 'tdsr':
        agent = algs.TDSR(state_size, action_size, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynasr':
        agent = algs.DynaSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'qparsr':
        agent = algs.PARSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, goal_pri=True, online=True)
    elif args.agent == 'mparsr':
        agent = algs.PARSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, goal_pri=False, online=True)
    elif args.agent == 'mdq':
        agent = algs.MDQ(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, online=True)
    else:
        raise ValueError('Invalid agent type: %s' % args.agent)

    return agent

def test_from_Q(Q, env, ap, gp, rw, args):
    agent_test = algs.TDQ(env.state_size, env.action_size, learning_rate=args.lr, gamma=args.gamma, Q_init=Q)
    experiences, _ = utils.run_episode(agent_test, env, agent_pos=ap, goal_pos=gp, reward_val=rw, update=False)
    return len(experiences)

def main(args):
    args_dict = vars(args).copy()
    del args_dict['output']
    args_keys = list(args_dict.keys())

    if not os.path.exists(args.output):
        with open(args.output, 'a') as f:
            f.write(','.join(args_keys + ['learns_before', 'learns_after', 'learns_both', 'exp_id']) + '\n')

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
            _, _ = utils.run_episode(agent, env, beta=args.beta, agent_pos=agent_pos, goal_pos=goal_pos, reward_val=reward_val, episode_length=EPISODE_LENGTH)

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

    output_dir = os.path.dirname(args.output)
    exp_id = str(uuid.uuid4())
    log_name = args.agent + '_' + exp_id
    log_dir = os.path.join(output_dir, log_name)
    os.mkdir(log_dir)
    # np.save(os.path.join(log_dir, 'prioritized_states.npy'), agent.prioritized_states)

    # if 'sr' in args.agent:
    #     M = agent.get_M_states(beta=args.beta).copy()
    # elif args.agent == 'mdq':
    #     M = agent.M.copy()

    # np.save(os.path.join(log_dir, 'M.npy'), M)
    np.save(os.path.join(log_dir, 'Qs_before.npy'), Qs_before)
    np.save(os.path.join(log_dir, 'Qs_after.npy'), Qs_after)

    args_vals = [str(val) for val in args_dict.values()]
    with open(args.output, 'a') as f:
        f.write(','.join(args_vals + [str(np.mean(learns_before)), str(np.mean(learns_after)), str(np.mean(learns_both)), exp_id]) + '\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--res', type=int, default=1, help='resolution of gridworld')
    parser.add_argument('--num_recall', type=int, default=10000, help='number of recalls')
    parser.add_argument('--agent', type=str, default='dynasr', help='algorithm')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--beta', type=float, default=5, help='softmax inverse temp')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for agent')
    parser.add_argument('--output', type=str, help='path to results file')
    args = parser.parse_args()
    main(args)

