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
from prioritizedsr.gridworld import Sequential

NUM_RUNS = 100
MAX_TRAINING_EPISODES = 100
MAX_INTRO = 100
condition_rewards = {
    'control': [15.0, 0.0, 45.0],
    'transition': [15.0, 0.0, 30.0],
    'reward': [45.0, 0.0, 30.0],
    'policy': [45.0, 15.0, 30.0]
}
condition_p1acts = {
    'control': [1, 0, 1],
    'transition': [1, 0, 1],
    'reward': [1, 0, 1],
    'policy': [1, 1, 1]
}
condition_p23acts = {
    'control': [1, 0, 1],
    'transition': [0, 1, 0],
    'reward': [0, 0, 1],
    'policy': [0, 0, 1]
}

def test_agent(agent, phase, condition):
    policy = agent.Q.argmax(0)
    if phase == 1:
        correct_acts = condition_p1acts[condition]
        return np.allclose(policy[:3], correct_acts)
    elif phase == 2:
        correct_acts = condition_p23acts[condition]
        return np.allclose(policy[1:3], correct_acts[1:])
    elif phase == 3:
        correct_acts = condition_p23acts[condition]
        return np.allclose(policy[0], correct_acts[0])
    else:
        return ValueError('phase must be 1, 2, or 3')

def main(args):
    assert args.condition in ['control', 'transition', 'reward', 'policy']
    learns_p1, learns_p2, learns_p3 = 0, 0, 0
    args_dict = vars(args).copy()
    del args_dict['output']
    args_keys = list(args_dict.keys())

    if not os.path.exists(args.output):
        with open(args.output, 'a') as f:
            f.write(','.join(args_keys + ['learns_p1', 'learns_p2', 'learns_p3', 'num_eps_phase1', 'num_eps_phase2', 'exp_id']) + '\n')

    npr.seed(args.seed)
    agent_pos = 0
    env = Sequential()

    Qs = np.zeros((NUM_RUNS, env.action_size, env.state_size))
    Ms = np.zeros((NUM_RUNS, env.action_size, env.state_size, env.state_size))
    pss = np.zeros((NUM_RUNS, env.state_size))

    num_eps_phase1 = 0
    num_eps_phase2 = 0

    for i in tqdm.tqdm(range(NUM_RUNS)):
        env = Sequential()
        agent = utils.agent_factory(args, env.state_size, env.action_size)

        # == PHASE 1: TRAIN ==
        if args.condition == 'policy':
            reward_val = [0.0, 15.0, 30.0]
        else:    
            reward_val = [15.0, 0.0, 30.0]

        phase1_results = []
        passed_phase1 = False

        for j in range(MAX_TRAINING_EPISODES):
            # train for an episode
            _, _ = utils.run_episode(agent, env, beta=args.beta, reward_val=reward_val)
            # test agent
            phase1_results.append(test_agent(agent, 1, args.condition))
            if np.sum(np.array(phase1_results)[-3:]) == 3:
                passed_phase1 = True
                break

        # == PHASE 1: TEST ==
        num_eps_phase1 += (j+1) / NUM_RUNS
        if passed_phase1:
            learns_p1 += 1 / NUM_RUNS
        else:
            continue

        # == PHASE 2: TRAIN ==
        reward_val = condition_rewards[args.condition]

        if args.condition == 'transition':
            env = Sequential(transition_pattern='reval')

        phase2_results = []
        passed_phase2 = False

        for j in range(MAX_INTRO):
            # train agent
            for state in [1, 2]:
                for action in [0, 1]:
                    env.reset(agent_pos=state, reward_val=reward_val)
                    reward = env.step(action)
                    state_next = env.observation
                    done = env.done
                    agent.update((state, action, state_next, reward, done))
            # test agent
            phase2_results.append(test_agent(agent, 2, args.condition))
            if np.sum(np.array(phase2_results)[-3:]) == 3:
                passed_phase2 = True
                break

        # == PHASE 2: TEST ==
        num_eps_phase2 += (j+1) / NUM_RUNS
        if passed_phase2:
            learns_p2 += 1 / NUM_RUNS
        else:
            continue

        # == PHASE 3: TEST ==
        learns_p3 += test_agent(agent, 3, args.condition) / NUM_RUNS

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

    if 'sr' in args.agent:
        np.save(os.path.join(log_dir, 'Ms.npy'), Ms)

    np.save(os.path.join(log_dir, 'Qs.npy'), Qs)

    args_vals = [str(val) for val in args_dict.values()]

    with open(args.output, 'a') as f:
        f.write(','.join(args_vals + [str(np.round_(learns_p1, 2)), str(np.round_(learns_p2, 2)), str(np.round_(learns_p3, 2)), str(np.round_(num_eps_phase1, 2)), str(np.round_(num_eps_phase2, 2)), exp_id]) + '\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--condition', type=str, default='control', help='experiment type')
    parser.add_argument('--num_recall', type=int, default=10, help='number of recalls')
    parser.add_argument('--agent', type=str, default='dynasr', help='algorithm')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--beta', type=float, default=5, help='softmax inverse temp')
    parser.add_argument('--theta', type=float, default=1e-6, help='ps theta')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for agent')
    parser.add_argument('--output', type=str, help='path to results file')
    args = parser.parse_args()
    main(args)

