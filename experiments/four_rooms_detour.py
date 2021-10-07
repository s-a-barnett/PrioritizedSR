import argparse
import numpy as np
import numpy.random as npr
import tqdm
import time
import pickle
import uuid
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prioritizedsr import algs, utils
from prioritizedsr.gridworld import SimpleGrid

NUM_RUNS = 10
MAX_TRAINING_EPISODES = 1000
NUM_BLOCKSTEPS        = 40

def test_from_Q(Q, env, ap, gp, args):
    agent_test = algs.TDQ(env.state_size, env.action_size, Q_init=Q)
    experiences, _ = utils.run_episode(agent_test, env, agent_pos=ap, goal_pos=gp, update=False)
    return len(experiences)

def main(args):
    args_dict = vars(args).copy()
    del args_dict['output']
    args_keys = list(args_dict.keys())

    if not os.path.exists(args.output):
        with open(args.output, 'a') as f:
            f.write(','.join(args_keys + ['learns_before', 'learns_after', 'learns_both', 'exp_id']) + '\n')

    npr.seed(args.seed)
    env = SimpleGrid(args.grid_size, block_pattern='four_rooms', obs_mode='index')
    mid = int(args.grid_size // 2)
    earl_mid = int(mid // 2)
    late_mid = mid+earl_mid + 1
    optimal_before_length = args.grid_size - 1 + (2 * earl_mid)
    optimal_after_length = args.grid_size - 1 + (2 * late_mid)

    Qs_before = np.zeros((NUM_RUNS, env.action_size, env.state_size))
    Qs_after = np.zeros_like(Qs_before)

    tic = time.time()

    for i in tqdm.tqdm(range(NUM_RUNS)):
        agent = utils.agent_factory(args, env.state_size, env.action_size)

        agent_pos = [0, 0]
        goal_pos = [0, args.grid_size -1]

        ## train on the regular four rooms task
        agent.num_recall = 0
        for j in range(MAX_TRAINING_EPISODES):
            experiences, td_errors = utils.run_episode(agent, env, beta=args.beta, goal_pos=goal_pos)

        Qs_before[i] = agent.Q.copy()

        ## introduce the block
        env = SimpleGrid(args.grid_size, block_pattern='four_rooms_blocked', obs_mode='index')
        agent.num_recall = args.num_recall
        agent_detour_pos = [earl_mid, mid-1]

        for j in range(NUM_BLOCKSTEPS):
            # start agent to left of wall
            env.reset(agent_pos=agent_detour_pos, goal_pos=goal_pos)
            state = env.observation
            # make agent try to go right
            action = 3
            reward = env.step(action)
            state_next = env.observation
            done = env.done
            exp = (state, action, state_next, reward, done)
            # perform update
            td_error = agent.update(exp)
            
        Qs_after[i] = agent.Q.copy()

    # get run lengths for each of these tasks
    before_lengths = np.array([test_from_Q(Qs_before[i], env, agent_pos, goal_pos, args) for i in range(NUM_RUNS)])
    after_lengths = np.array([test_from_Q(Qs_after[i], env, agent_pos, goal_pos, args) for i in range(NUM_RUNS)])

    learns_before = (before_lengths == optimal_before_length)
    learns_after  = (after_lengths  == optimal_after_length)
    learns_both   = learns_before * learns_after

    toc = time.time()

    wall_clock_time = toc - tic

    output_dir = os.path.dirname(args.output)
    exp_id = str(uuid.uuid4())
    log_name = args.agent + '_' + exp_id
    log_dir = os.path.join(output_dir, log_name)
    os.mkdir(log_dir)

    np.save(os.path.join(log_dir, 'Qs_before.npy'), Qs_before)
    np.save(os.path.join(log_dir, 'Qs_after.npy'), Qs_after)

    args_vals = [str(val) for val in args_dict.values()]
    with open(args.output, 'a') as f:
        f.write(','.join(args_vals + [str(np.mean(learns_before)), str(np.mean(learns_after)), str(np.mean(learns_both)), exp_id]) + '\n')
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--agent', type=str, default='tdsr', help='algorithm')
    parser.add_argument('--beta', type=float, default=5, help='softmax inverse temperature')
    parser.add_argument('--num_recall', type=int, default=10, help='number of recall steps')
    parser.add_argument('--grid_size', type=int, default=7, help='size of grid')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate for agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for agent')
    parser.add_argument('--output', type=str, help='path to results file')
    args = parser.parse_args()
    main(args)
