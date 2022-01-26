import numpy as np
import numpy.random as npr
import tqdm
import argparse
import itertools
import uuid
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prioritizedsr import algs, utils
from prioritizedsr.gridworld import SimpleGrid

def test_from_Q(Q, env, ap, gp, rw, args):
    agent_test = algs.TDQ(env.state_size, env.action_size, Q_init=Q)
    experiences, _ = utils.run_episode(agent_test, env, agent_pos=ap, goal_pos=gp, reward_val=rw, update=False)
    return len(experiences)

def main(args):
    args_dict = vars(args).copy()
    del args_dict['output']
    args_keys = list(args_dict.keys())

    columns_string = ','.join(args_keys + ['learns_before', 'learns_after', 'learns_both', 'exp_id']) + '\n'
    if not args.debug:
        if not os.path.exists(args.output):
            with open(args.output, 'a') as f:
                f.write(columns_string)

    npr.seed(args.seed)
    r = args.res
    grid_size = (10*r, 10*r)
    agent_start = [4*r, 0]
    goal_pos = [[4*r + x, 9*r + y] for x, y in itertools.product(range(r), range(r))]
    reward_val = 10
    optimal_episode_lengths = {'before': 9*r, 'after': 11*r + 2}

    env_before = SimpleGrid(grid_size, block_pattern='detour_before')
    Qs_before = np.zeros((args.num_runs, env_before.action_size, env_before.state_size))
    Qs_after = np.zeros_like(Qs_before)
    Ms_before = np.zeros((args.num_runs, env_before.action_size, env_before.state_size, env_before.state_size))
    Ms_after = np.zeros_like(Ms_before)
    pss_before = np.zeros((args.num_runs, env_before.action_size, env_before.state_size))
    pss_after = np.zeros_like(pss_before)

    for i in tqdm.tqdm(range(args.num_runs)):
        agent = utils.agent_factory(args, env_before.state_size, env_before.action_size)

        # train on original task
        for j in range(args.max_episodes):
            _, _ = utils.run_episode(agent, env_before, beta=args.beta, epsilon=args.epsilon, poltype=args.poltype,  agent_pos=agent_start, goal_pos=goal_pos, reward_val=reward_val, episode_length=args.max_episode_length)

        # record Q value for original task
        Qs_before[i] = agent.Q.copy()
        if 'sr' in args.agent:
            Ms_before[i] = agent.M.copy()
        pss_before[i] = agent.prioritized_states.copy()

        # introduce wall
        env_after = SimpleGrid(grid_size, block_pattern='detour_after')

        for j in range(args.max_episodes):
            for x in range(r):
                env_after.reset(agent_pos=[4*r + x, 5*r -1], goal_pos=goal_pos, reward_val=reward_val)
                state = env_after.observation
                action = 3
                reward = env_after.step(action)
                state_next = env_after.observation
                done = env_after.done
                agent.update((state, action, state_next, reward, done))

        # record Q value for detour
        Qs_after[i] = agent.Q.copy()
        if 'sr' in args.agent:
            Ms_after[i] = agent.M.copy()
        pss_after[i] = agent.prioritized_states.copy()
        
    # get run lengths for each of these tasks
    before_lengths = np.array([test_from_Q(Qs_before[i], env_before, agent_start, goal_pos, reward_val, args) for i in range(args.num_runs)])
    after_lengths = np.array([test_from_Q(Qs_after[i], env_after, agent_start, goal_pos, reward_val, args) for i in range(args.num_runs)])

    learns_before = (before_lengths == optimal_episode_lengths['before'])
    learns_after  = (after_lengths  == optimal_episode_lengths['after'])
    learns_both   = learns_before * learns_after

    exp_id = str(uuid.uuid4())
    args_vals = [str(val) for val in args_dict.values()]
    results_string = ','.join(args_vals + [str(np.mean(learns_before)), str(np.mean(learns_after)), str(np.mean(learns_both)), exp_id]) + '\n'

    if args.debug:
        print(columns_string)
        print(results_string)
    else:
        output_dir = os.path.dirname(args.output)
        log_name = args.agent + '_' + exp_id
        log_dir = os.path.join(output_dir, log_name)
        os.mkdir(log_dir)
        np.save(os.path.join(log_dir, 'prioritized_states_before.npy'), pss_before)
        np.save(os.path.join(log_dir, 'prioritized_states_after.npy'), pss_after)

        if 'sr' in args.agent:
            np.save(os.path.join(log_dir, 'Ms_before.npy'), Ms_before)
            np.save(os.path.join(log_dir, 'Ms_after.npy'), Ms_after)

        np.save(os.path.join(log_dir, 'Qs_before.npy'), Qs_before)
        np.save(os.path.join(log_dir, 'Qs_after.npy'), Qs_after)

        with open(args.output, 'a') as f:
            f.write(results_string)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--res', type=int, default=1, help='resolution of gridworld')
    parser.add_argument('--num_recall', type=int, default=10, help='number of recalls')
    parser.add_argument('--agent', type=str, default='dynasr', help='algorithm')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--beta', type=float, default=5, help='softmax inverse temp')
    parser.add_argument('--epsilon', type=float, default=1e-1, help='epsilon greedy exploration parameter')
    parser.add_argument('--poltype', type=str, default='softmax', help='egreedy or softmax')
    parser.add_argument('--theta', type=float, default=1e-6, help='ps theta')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for agent')
    parser.add_argument('--num_runs', type=int, default=10, help='experiment runs')
    parser.add_argument('--max_episodes', type=int, default=1000, help='phase iter size')
    parser.add_argument('--max_episode_length', type=int, default=10000, help='length of a single episode')
    parser.add_argument('--debug', help='only print output', action='store_true')
    parser.add_argument('--output', type=str, help='path to results file')
    args = parser.parse_args()
    main(args)

