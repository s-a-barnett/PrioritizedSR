import copy
import argparse
import numpy as np
import numpy.random as npr
import tqdm
import time
import uuid
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prioritizedsr import algs, utils
from prioritizedsr.gridworld import SimpleGrid, StochasticSimpleGrid

MAX_TRAINING_EPISODES = 10000

def main(args):
    args_dict = vars(args).copy()
    del args_dict['output']
    args_keys = list(args_dict.keys())

    if not os.path.exists(args.output):
        with open(args.output, 'a') as f:
            f.write(','.join(args_keys + ['num_eps_to_optimality', 'wall_clock_time', 'exp_id']) + '\n')

    npr.seed(args.seed)
    if args.stochastic:
        env = StochasticSimpleGrid(args.grid_size, block_pattern='four_rooms')
    else:
        env = SimpleGrid(args.grid_size, block_pattern='four_rooms')
    agent = utils.agent_factory(args, env.state_size, env.action_size)
    optimal_episode_length = 2 * (args.grid_size - 1)

    iterations = []

    agent_pos = [0, 0]
    goal_pos = [args.grid_size -1, args.grid_size -1]
    stoch_goal_pos = [args.grid_size - 1, 0]

    tic = time.time()

    for i in tqdm.tqdm(range(MAX_TRAINING_EPISODES)):
        if args.stochastic:
            # train for an episode
            experiences, td_errors = utils.run_episode(agent, env, beta=args.beta, agent_pos=agent_pos, goal_pos=goal_pos, stoch_goal_pos=stoch_goal_pos)

            # every episode, test to see whether optimal policy is achieved
            experiences, _ = utils.run_episode(agent, env, beta=1e6, agent_pos=agent_pos, goal_pos=goal_pos, update=False, stoch_goal_pos=stoch_goal_pos)
        else:
                # train for an episode
            experiences, td_errors = utils.run_episode(agent, env, beta=args.beta, agent_pos=agent_pos, goal_pos=goal_pos)

            # every episode, test to see whether optimal policy is achieved
            experiences, _ = utils.run_episode(agent, env, beta=1e6, agent_pos=agent_pos, goal_pos=goal_pos, update=False)
        iterations.append(i)
        if len(experiences) == optimal_episode_length:
            break

    toc = time.time()

    num_eps_to_optimality = i
    wall_clock_time = toc - tic

    print('\n\n Number of episodes to optimality: %d' % num_eps_to_optimality)
    print('\n Wall clock time to convergence: %.2fs' % wall_clock_time)

    output_dir = os.path.dirname(args.output)
    exp_id = str(uuid.uuid4())
    log_name = args.agent + '_' + exp_id
    log_dir = os.path.join(output_dir, log_name)
    os.mkdir(log_dir)
    np.save(os.path.join(log_dir, 'prioritized_states.npy'), agent.prioritized_states)

    if 'sr' in args.agent:
        M = agent.get_M_states(beta=args.beta).copy()
    elif args.agent == 'mdq':
        M = agent.M.copy()

    np.save(os.path.join(log_dir, 'M.npy'), M)
    np.save(os.path.join(log_dir, 'Q.npy'), agent.Q)

    args_vals = [str(val) for val in args_dict.values()]
    with open(args.output, 'a') as f:
        f.write(','.join(args_vals + [str(num_eps_to_optimality), str(wall_clock_time), exp_id]) + '\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--agent', type=str, default='tdsr', help='agent type')
    parser.add_argument('--beta', type=float, default=5.0, help='softmax inverse temperature')
    parser.add_argument('--num_recall', type=int, default=10, help='number of recall steps')
    parser.add_argument('--grid_size', type=int, default=7, help='size of grid')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate for agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for agent')
    parser.add_argument('--stochastic', type=bool, default=False, help='stochastic environment')
    parser.add_argument('--output', type=str, help='path to results file')
    args = parser.parse_args()
    main(args)
