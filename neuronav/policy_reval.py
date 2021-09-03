import numpy as np
import numpy.random as npr
import algs
import progressbar
import utils
from gridworld import SimpleGrid
import argparse

# fixed parameters
gamma = 0.95
beta  = 5

NUM_RUNS = 10
NUM_INTRO = 20
EPISODE_LENGTH = 25000

def agent_factory(args, state_size, action_size):
    if args.agent == 'dynasr':
        agent = algs.DynaSR(state_size, action_size, num_recall=args.num_recall, learning_rate=args.lr, gamma=gamma)
    elif args.agent == 'qparsr':
        agent = algs.PARSR(state_size, action_size, num_recall=args.num_recall, learning_rate=args.lr, gamma=gamma, goal_pri=True)
    elif args.agent == 'mparsr':
        agent = algs.PARSR(state_size, action_size, num_recall=args.num_recall, learning_rate=args.lr, gamma=gamma, goal_pri=False)
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
    grid_size = (10, 10)
    agent_pos = [[7, 0], [8, 8]]
    goal_pos = [[1, 9], [8, 9]]
    reward_val = [10, 20]
    optimal_episode_lengths = {'S1R1': 19, 'S1R2': 20, 'S2R2': 1}

    env = SimpleGrid(grid_size, block_pattern='tolman')
    Qs_latent = np.zeros((NUM_RUNS, env.action_size, env.state_size))
    Qs_reval = np.zeros_like(Qs_latent)

    for i in progressbar.progressbar(range(NUM_RUNS)):
        agent = agent_factory(args, env.state_size, env.action_size)

        # train on latent learning task
        agent.num_recall = 0
        if ('PARSR' in type(agent).__name__) or (type(agent).__name__ == 'MDQ'):
            agent.online = True

        env.reset(agent_pos=agent_pos[0], reward_val=0)
        state = env.observation
        experiences = []
        for j in range(EPISODE_LENGTH):
            action = agent.sample_action(state, beta=beta)
            reward = env.step(action)
            state_next = env.observation
            done = env.done
            experiences.append((state, action, state_next, reward, done))
            state = state_next

            if j > 0:
                agent.update(experiences[-2], next_exp=experiences[-1])

        # add in reward R1
        agent.num_recall = args.num_recall
        if ('PARSR' in type(agent).__name__) or (type(agent).__name__ == 'MDQ'):
            agent.online = True

        for j in range(NUM_INTRO):
            env.reset(agent_pos=goal_pos[0], goal_pos=goal_pos[0], reward_val=reward_val[0])
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
        if ('PARSR' in type(agent).__name__) or (type(agent).__name__ == 'MDQ'):
            agent.online = True

        for j in range(NUM_INTRO):
            env.reset(agent_pos=agent_pos[j % 2], goal_pos=goal_pos[0], reward_val=reward_val[0])
            state = env.observation
            experiences = []
            for k in range(EPISODE_LENGTH):
                action = agent.sample_action(state, beta=beta)
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
        if ('PARSR' in type(agent).__name__) or (type(agent).__name__ == 'MDQ'):
            agent.online = True

        for j in range(NUM_INTRO):
            env.reset(agent_pos=goal_pos[1], goal_pos=goal_pos[1], reward_val=reward_val[1])
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

    with open(args.output, 'a') as f:
        f.write(','.join([str(args.agent), str(args.seed), str(args.lr), str(args.num_recall), str(np.mean(learns_latent)), str(np.mean(learns_reval)), str(np.mean(learns_both))]) + '\n')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_recall', type=int, default=10000, help='number of recalls')
    parser.add_argument('--agent', type=str, default='dynasr', help='algorithm')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    main(args)

