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

def main(args):
    learns_before, learns_after = 0, 0
    args_dict = vars(args).copy()
    del args_dict['output']
    args_keys = list(args_dict.keys())

    columns_string = ','.join(args_keys + ['learns_before', 'learns_after', 'num_eps_before', 'num_eps_after', 'exp_id']) + '\n'
    if not args.debug:
        if not os.path.exists(args.output):
            with open(args.output, 'a') as f:
                f.write(columns_string)

    # sarsa = (args.agent == 'dynasr')
    sarsa = False

    npr.seed(args.seed)
    r = args.res
    grid_size = (10*r, 10*r)
    agent_start = [4*r, 0]
    goal_pos = [[4*r + x, 9*r + y] for x, y in itertools.product(range(r), range(r))]
    reward_val = 10
    optimal_episode_lengths = {'before': 9*r, 'after': 11*r + 2}

    def test_agent(agent, env, ap, gp, rw, phase):
        experiences, _ = utils.run_episode(agent, env, agent_pos=ap, goal_pos=gp, reward_val=rw, update=False)
        success = (len(experiences) == optimal_episode_lengths[phase])
        return success

    env_before = SimpleGrid(grid_size, block_pattern='detour_before')
    Qs_before = np.zeros((args.num_runs, env_before.action_size, env_before.state_size))
    Qs_after = np.zeros_like(Qs_before)
    Ms_before = np.zeros((args.num_runs, env_before.action_size, env_before.state_size, env_before.state_size))
    Ms_after = np.zeros_like(Ms_before)
    pss_before = np.zeros((args.num_runs, env_before.action_size, env_before.state_size))
    pss_after = np.zeros_like(pss_before)

    num_eps_before = 0
    num_eps_after = 0

    for i in tqdm.tqdm(range(args.num_runs)):
        agent = utils.agent_factory(args, env_before.state_size, env_before.action_size)

        before_results = []
        passed_before = False

        # train on original task
        for j in range(args.max_episodes):
            # train for an episode
            _, _ = utils.run_episode(agent, env_before, beta=args.beta, epsilon=args.epsilon, poltype=args.poltype,  agent_pos=agent_start, goal_pos=goal_pos, reward_val=reward_val, episode_length=args.max_episode_length, sarsa=sarsa)
            # test agent
            before_results.append(test_agent(agent, env_before, agent_start, goal_pos, reward_val, 'before'))
            if np.sum(np.array(before_results)[-3:]) == 3:
                passed_before = True
                break

        # record Q value for original task
        Qs_before[i] = agent.Q.copy()
        if 'sr' in args.agent:
            Ms_before[i] = agent.M.copy()
        pss_before[i] = agent.prioritized_states.copy()

        num_eps_before += (j+1) / args.num_runs
        if passed_before:
            learns_before += 1 / args.num_runs
        else:
            continue

        # introduce wall
        env_after = SimpleGrid(grid_size, block_pattern='detour_after')

        after_results = []
        passed_after = False

        for j in range(args.max_episodes):
            # train agent
            for x in range(r):
                env_after.reset(agent_pos=[4*r + x, 5*r -1], goal_pos=goal_pos, reward_val=reward_val)
                state = env_after.observation
                action = 3
                reward = env_after.step(action)
                state_next = env_after.observation
                done = env_after.done
                exp = (state, action, state_next, reward, done)
                if sarsa:
                    next_action = agent.sample_action(state_next, epsilon=args.epsilon, beta=args.beta)
                    next_exp = (None, next_action, None, None, None)
                else:
                    next_exp = None
                agent.update(exp, next_exp=next_exp)
            # test agent
            after_results.append(test_agent(agent, env_after, agent_start, goal_pos, reward_val, 'after'))
            if np.sum(np.array(after_results)[-3:]) == 3:
                passed_after = True
                break

        # record Q value for detour
        Qs_after[i] = agent.Q.copy()
        if 'sr' in args.agent:
            Ms_after[i] = agent.M.copy()
        pss_after[i] = agent.prioritized_states.copy()

        num_eps_after += (j+1) / args.num_runs
        if passed_after:
            learns_after += 1 / args.num_runs
        
    # lazy fix for bug in recording number of episodes to learn each phase
    if learns_before > 0:
        num_eps_before /= learns_before
    if learns_after > 0:
        num_eps_after /= learns_after

    exp_id = str(uuid.uuid4())
    args_vals = [str(val) for val in args_dict.values()]
    results_string = ','.join(args_vals + [str(np.round_(learns_before, 2)), str(np.round_(learns_after, 2)), str(np.round_(num_eps_before, 2)), str(np.round_(num_eps_after, 2)), exp_id]) + '\n'

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

