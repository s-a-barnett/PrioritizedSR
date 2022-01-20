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
from prioritizedsr.gridworld import SequentialNoAct

condition_rewards = {
    'control': [10.0, 1.0],
    'transition': [10.0, 1.0],
    'reward': [1.0, 10.0],
}

def state_weights(agent):
    if hasattr(agent, 'w'):
        return agent.w.copy()
    else:
        return agent.Q.max(0).copy()

def test_agent(agent, phase, condition):
    Q = agent.Q
    V = Q.max(0)
    if phase == 1:
        return V[0] > V[1]
    elif phase == 2:
        if condition != 'control':
            return V[2] < V[3]
        else:
            return V[2] > V[3]
    elif phase == 3:
        if condition != 'control':
            return V[0] < V[1]
        else:
            return V[0] > V[1]
    else:
        raise ValueError('phase must be 1, 2, or 3')

def main(args):
    assert args.condition in ['control', 'transition', 'reward']
    learns_p1, learns_p2, learns_p3 = 0, 0, 0
    args_dict = vars(args).copy()
    del args_dict['output']
    args_keys = list(args_dict.keys())

    if not os.path.exists(args.output):
        with open(args.output, 'a') as f:
            f.write(','.join(args_keys + ['learns_p1', 'learns_p2', 'learns_p3', 'num_eps_phase1', 'num_eps_phase2', 'exp_id']) + '\n')

    npr.seed(args.seed)
    agent_pos = 0
    env = SequentialNoAct()

    Qs = np.zeros((args.num_runs, env.action_size, env.state_size))
    Ms = np.zeros((args.num_runs, env.action_size, env.state_size, env.state_size))
    pss = np.zeros((args.num_runs, env.state_size))

    num_eps_phase1 = 0
    num_eps_phase2 = 0

    weights_allruns = []
    srs_allruns = []
    max_run_length = 0

    for i in tqdm.tqdm(range(args.num_runs)):
        env = SequentialNoAct()
        agent = utils.agent_factory(args, env.state_size, env.action_size)

        # == PHASE 1: TRAIN ==
        reward_val = [10.0, 1.0]

        phase1_results = []
        passed_phase1 = False

        weights = []
        srs = []

        for j in range(args.max_episodes):
            # train for an episode
            _, _ = utils.run_episode(agent, env, agent_pos=(j%2), reward_val=reward_val)
            weights.append(state_weights(agent))
            srs.append(agent.M[0])
            # test agent
            phase1_results.append(test_agent(agent, 1, args.condition))
            if np.sum(np.array(phase1_results)[-3:]) == 3:
                passed_phase1 = True
                break

        # == PHASE 1: TEST ==
        num_eps_phase1 += (j+1) / args.num_runs
        if passed_phase1:
            learns_p1 += 1 / args.num_runs
        else:
            continue

        # == PHASE 2: TRAIN ==
        reward_val = condition_rewards[args.condition]

        if args.condition == 'transition':
            env = SequentialNoAct(transition_pattern='reval')

        phase2_results = []
        passed_phase2 = False

        introstates = [2, 3]

        for j in range(args.max_episodes):
            # train agent
            for state in introstates:
                env.reset(agent_pos=state, reward_val=reward_val)
                reward = env.step(0)
                state_next = env.observation
                done = env.done
                agent.update((state, 0, state_next, reward, done))
            weights.append(state_weights(agent))
            srs.append(agent.M[0])
            # test agent
            phase2_results.append(test_agent(agent, 2, args.condition))
            if np.sum(np.array(phase2_results)[-3:]) == 3:
                passed_phase2 = True
                break

        weights_allruns.append(np.stack(weights))
        srs_allruns.append(np.stack(srs))
        max_run_length = np.maximum(max_run_length, len(srs))

        # == PHASE 2: TEST ==
        num_eps_phase2 += (j+1) / args.num_runs
        if passed_phase2:
            learns_p2 += 1 / args.num_runs
        else:
            continue

        # == PHASE 3: TEST ==
        learns_p3 += test_agent(agent, 3, args.condition) / args.num_runs

        Qs[i] = agent.Q.copy()
        if 'sr' in args.agent:
            Ms[i] = agent.M.copy()
        pss[i] = agent.prioritized_states.copy()

    output_dir = os.path.dirname(args.output)
    exp_id = str(uuid.uuid4())
    log_name = args.agent + '_' + exp_id
    log_dir = os.path.join(output_dir, log_name)
    os.mkdir(log_dir)
    np.save(os.path.join(log_dir, 'prioritized_states.npy'), pss)

    for idx in range(len(weights_allruns)):
        weights = weights_allruns[idx]
        pad_width = int(max_run_length - weights.shape[0])
        weights = np.pad(weights, ((0, pad_width), (0, 0)))
        weights_allruns[idx] = weights
        if 'sr' in args.agent:
            srs = srs_allruns[idx]
            srs = np.pad(srs, ((0, pad_width), (0, 0), (0, 0)))
            srs_allruns[idx] = srs

    if 'sr' in args.agent:
        np.save(os.path.join(log_dir, 'Ms.npy'), Ms)
        np.save(os.path.join(log_dir, 'srs.npy'), np.stack(srs_allruns))

    np.save(os.path.join(log_dir, 'Qs.npy'), Qs)
    np.save(os.path.join(log_dir, 'weights.npy'), np.stack(weights_allruns))

    args_vals = [str(val) for val in args_dict.values()]

    with open(args.output, 'a') as f:
        f.write(','.join(args_vals + [str(np.round_(learns_p1, 2)), str(np.round_(learns_p2, 2)), str(np.round_(learns_p3, 2)), str(np.round_(num_eps_phase1, 2)), str(np.round_(num_eps_phase2, 2)), exp_id]) + '\n')

    return 0

if __name__ == '__main__':
    default_path = os.path.join(os.path.expanduser('~'), 
                                'scr/parsr/out.csv')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--condition', type=str, default='control', help='experiment type')
    parser.add_argument('--num_recall', type=int, default=10, help='number of recalls')
    parser.add_argument('--agent', type=str, default='dynasr', help='algorithm')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--theta', type=float, default=1e-6, help='ps theta')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for agent')
    parser.add_argument('--num_runs', type=int, default=100, help='experiment runs')
    parser.add_argument('--max_episodes', type=int, default=10000, help='phase iter size')
    parser.add_argument('--output', type=str, default=default_path,  help='path to results file')
    args = parser.parse_args()
    main(args)

