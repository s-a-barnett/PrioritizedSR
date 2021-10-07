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
EPISODE_LENGTH = 25000
NUM_BLOCKSTEPS = 40

def test_from_Q(Q, env, ap, gp, rw, args):
    agent_test = algs.TDQ(env.state_size, env.action_size, Q_init=Q)
    experiences, _ = utils.run_episode(agent_test, env, agent_pos=ap, goal_pos=gp, reward_val=rw, update=False)
    return len(experiences)

def main(args):
    args_dict = vars(args).copy()
    del args_dict['output']
    args_keys = list(args_dict.keys())

    if not os.path.exists(args.output):
        with open(args.output, 'a') as f:
            f.write(','.join(args_keys + ['learns_latent', 'learns_reval', 'learns_both', 'exp_id']) + '\n')

    npr.seed(args.seed)
    # agent has chance to be introduced to every square in goal_pos
    agent_pos = [0, 0]
    goal_pos = [[], []]
    reward_val = [10, 20]
    optimal_episode_lengths = {'S1R1': 16*r + 3, 'S1R2': 18*r + 2, 'S2R2': r}

    env = SimpleGrid(args.grid_size, block_pattern='four_rooms')
    Qs_latent = np.zeros((NUM_RUNS, env.action_size, env.state_size))
    Qs_reval = np.zeros_like(Qs_latent)

    for i in tqdm.tqdm(range(NUM_RUNS)):
        agent = utils.agent_factory(args, env.state_size, env.action_size)

        # train on latent learning task
        agent.num_recall = 0

        env.reset(agent_pos=agent_pos[0], reward_val=0)
        state = env.observation
        experiences = []
        for j in range(EPISODE_LENGTH):
            action = agent.sample_action(state, beta=args.beta)
            reward = env.step(action)
            state_next = env.observation
            done = env.done
            experiences.append((state, action, state_next, reward, done))
            state = state_next

            if j > 0:
                agent.update(experiences[-2], next_exp=experiences[-1])

        # add in reward R1
        agent.num_recall = args.num_recall

        for j in range(NUM_INTRO):
            env.reset(agent_pos=goal_pos[0][j % (r**2)], goal_pos=goal_pos[0], reward_val=reward_val[0])
            state = env.observation
            action = npr.choice(env.action_size)
            reward = env.step(action)
            state_next = env.observation
            done = env.done
            experience = (state, action, state_next, reward, done)
            _ = agent.update(experience)

        # record Q value for latent task
        Qs_latent[i] = agent.Q.copy()

        # train with alternating starting states
        agent.num_recall = 0

        for j in range(NUM_INTRO):
            env.reset(agent_pos=agent_pos[j % 2], goal_pos=goal_pos[0], reward_val=reward_val[0])
            state = env.observation
            experiences = []
            for k in range(EPISODE_LENGTH):
                action = agent.sample_action(state, beta=args.beta)
                reward = env.step(action)
                state_next = env.observation
                done = env.done
                experiences.append((state, action, state_next, reward, done))
                state = state_next

                if k > 0:
                    agent.update(experiences[-2], next_exp=experiences[-1])
                if done:
                    agent.update(experiences[-1], next_exp=experiences[-1])
                    break

        # introduce agent to new goal
        agent.num_recall = args.num_recall

        for j in range(NUM_INTRO):
            env.reset(agent_pos=goal_pos[1][j % (r**2)], goal_pos=goal_pos[1], reward_val=reward_val[1])
            state = env.observation
            action = npr.choice(env.action_size)
            reward = env.step(action)
            state_next = env.observation
            done = env.done
            experience = (state, action, state_next, reward, done)
            _ = agent.update(experience)

        # record Q value after policy reval
        Qs_reval[i] = agent.Q.copy()

    # get run lengths for each of these tasks
    latent_lengths = np.array([test_from_Q(Qs_latent[i], env, agent_pos[0], goal_pos[0], reward_val[0], args) for i in range(NUM_RUNS)])
    reval_lengths = np.array([test_from_Q(Qs_reval[i], env, agent_pos[0], goal_pos[1], reward_val[1], args) for i in range(NUM_RUNS)])

    learns_latent = (latent_lengths == optimal_episode_lengths['S1R1'])
    learns_reval  = (reval_lengths  == optimal_episode_lengths['S1R2'])
    learns_both   = learns_latent * learns_reval

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
    np.save(os.path.join(log_dir, 'Qs_latent.npy'), Qs_latent)
    np.save(os.path.join(log_dir, 'Qs_reval.npy'), Qs_reval)

    args_vals = [str(val) for val in args_dict.values()]
    with open(args.output, 'a') as f:
        f.write(','.join(args_vals + [str(np.mean(learns_latent)), str(np.mean(learns_reval)), str(np.mean(learns_both)), exp_id]) + '\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--grid_size', type=int, default=7, help='size of grid')
    parser.add_argument('--num_recall', type=int, default=10000, help='number of recalls')
    parser.add_argument('--agent', type=str, default='dynasr', help='algorithm')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--beta', type=float, default=5, help='softmax inverse temp')
    parser.add_argument('--theta', type=float, default=1e-6, help='ps theta')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for agent')
    parser.add_argument('--output', type=str, help='path to results file')
    args = parser.parse_args()
    main(args)

