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

NUM_RUNS = 10
MAX_EPISODE_LENGTH = 10000
MAX_TRAINING_EPISODES = 1000
MAX_INTRO = 10
condition_rewards = {
    'control': [15.0, 0.0, 45.0],
    'transition': [15.0, 0.0, 30.0],
    'reward': [45.0, 0.0, 30.0],
    'policy': [45.0, 15.0, 30.0]
}
condition_p1goalidxs = {
    'control': [2, 0, 2],
    'transition': [2, 0, 2],
    'reward': [2, 0, 2],
    'policy': [2, 1, 2]
}
condition_p3goalidxs = {
    'control': [2, 0, 2],
    'transition': [2, 2, 0],
    'reward': [0, 0, 2],
    'policy': [0, 0, 2]
}

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
    env = SimpleGrid(args.grid_size, block_pattern='six_rooms')
    room_length = env.mid - env.onesixth
    poss = [[env.onesixth, env.onesixth], [env.mid, env.onesixth], [env.onesixth, env.mid],
            [env.fivesixth, env.onesixth], [env.mid, env.mid], [env.onesixth, env.fivesixth]]
    agent_pos = poss[0]
    goal_pos = poss[3:]
    goal_states = np.array([env.grid_to_state(pos) for pos in goal_pos])
    
    def test_agent(agent, env, phase, condition, reward_val):
        ex1, _ = utils.run_episode(agent, env, agent_pos=poss[0], reward_val=reward_val, goal_pos=goal_pos, update=False)
        ex2, _ = utils.run_episode(agent, env, agent_pos=poss[1], reward_val=reward_val, goal_pos=goal_pos, update=False)
        ex3, _ = utils.run_episode(agent, env, agent_pos=poss[2], reward_val=reward_val, goal_pos=goal_pos, update=False)
        reached_goals = [ex[-1][2] for ex in [ex1, ex2, ex3]]
        trajlengths = [len(ex) for ex in [ex1, ex2, ex3]]
        correct_trajlengths = [2 * room_length, room_length, room_length]

        if phase == 1:
            correct_goals = goal_states[condition_p1goalidxs[condition]]
            return np.allclose(reached_goals, correct_goals) and np.allclose(trajlengths, correct_trajlengths)
        elif phase == 2:
            # not sure how to test for phase 2 consistently among conditions!
            #   instead just testing phase 3 from different conditions
            return True
        elif phase == 3:
            correct_goals = goal_states[condition_p3goalidxs[condition]]
            return np.allclose(reached_goals, correct_goals) and np.allclose(trajlengths, correct_trajlengths)
        else:
            raise ValueError('phase must be 1, 2, or 3')

    Qs = np.zeros((NUM_RUNS, env.action_size, env.state_size))
    Ms = np.zeros((NUM_RUNS, env.action_size, env.state_size, env.state_size))
    pss = np.zeros((NUM_RUNS, env.state_size))

    num_eps_phase1 = 0
    num_eps_phase2 = 0

    for i in tqdm.tqdm(range(NUM_RUNS)):
        env = SimpleGrid(args.grid_size, block_pattern='six_rooms')
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
            _, _ = utils.run_episode(agent, env, beta=args.beta, reward_val=reward_val, goal_pos=goal_pos, episode_length=MAX_EPISODE_LENGTH)
            # test agent
            phase1_results.append(test_agent(agent, env, 1, args.condition, reward_val))
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
            env = SimpleGrid(args.grid_size, block_pattern='six_rooms_tr')
            intro_poss = env.teleports.keys()
        else:
            intro_poss = goal_pos
        intro_states = [env.grid_to_state(pos) for pos in intro_poss]

        phase2_results = []
        # passed_phase2 = False
        passed_phase2 = True

        for j in range(MAX_INTRO):
            # train agent
            for pos in intro_poss:
                for action in range(env.action_size):
                    env.reset(agent_pos=pos, reward_val=reward_val, goal_pos=goal_pos)
                    state = env.observation
                    reward = env.step(action)
                    state_next = env.observation
                    done = env.done
                    agent.update((state, action, state_next, reward, done))
            # # test agent
            # phase2_results.append(test_agent(agent, env, 2, args.condition, reward_val))
            # if np.sum(np.array(phase2_results)[-3:]) == 3:
            #     passed_phase2 = True
            #     break

        # == PHASE 2: TEST ==
        num_eps_phase2 += (j+1) / NUM_RUNS
        if passed_phase2:
            learns_p2 += 1 / NUM_RUNS
        else:
            continue

        # == PHASE 3: TEST ==
        learns_p3 += test_agent(agent, env, 3, args.condition, reward_val) / NUM_RUNS

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
    parser.add_argument('--gamma', type=float, default=0.3, help='discount factor for agent')
    parser.add_argument('--grid_size', type=int, default=11, help='size of grid')
    parser.add_argument('--output', type=str, help='path to results file')
    args = parser.parse_args()
    main(args)

