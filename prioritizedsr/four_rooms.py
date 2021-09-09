import copy
import argparse
import numpy as np
import numpy.random as npr
import algs
import progressbar
import utils
from gridworld import SimpleGrid
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import pickle
import uuid

MAX_TRAINING_EPISODES = 10000

def agent_factory(args, state_size, action_size):
    if args.agent == 'tdsr':
        agent = algs.TDSR(state_size, action_size, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynasr':
        agent = algs.DynaSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynasrplus':
        agent = algs.DynaSR(state_size, action_size, args.num_recall, kappa=1e-3, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'qparsr':
        agent = algs.PARSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, goal_pri=True, online=True)
    elif args.agent == 'mparsr':
        agent = algs.PARSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, goal_pri=False, online=True)
    elif args.agent == 'tdq':
        agent = algs.TDQ(state_size, action_size, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynaq':
        agent = algs.DynaQ(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynaqplus':
        agent = algs.DynaQ(state_size, action_size, args.num_recall, kappa=1e-3, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'psq':
        agent = algs.PSQ(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'mdq':
        agent = algs.MDQ(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, online=True)
    else:
        raise ValueError('Invalid agent type: %s' % args.agent)

    return agent

def main(args):
    logger = utils.Logger('four_rooms', vars(args))
    npr.seed(args.seed)
    env = SimpleGrid(args.grid_size, block_pattern='four_rooms', obs_mode='index')
    agent = agent_factory(args, env.state_size, env.action_size)
    optimal_episode_length = 2 * (args.grid_size - 1)

    logger.logs['ep_test_cum_reward'] = []
    logger.logs['ep_train_td_errors'] = []

    iterations = []
    optimal_return = args.gamma ** optimal_episode_length

    agent_pos = [0, 0]
    goal_pos = [args.grid_size -1, args.grid_size -1]

    tic = time.time()

    for i in progressbar.progressbar(range(MAX_TRAINING_EPISODES + args.num_pretrain)):
        # train for an episode
        pretrain = True if i < args.num_pretrain else False
        experiences, td_errors = utils.run_episode(agent, env, beta=args.beta, agent_pos=agent_pos, goal_pos=goal_pos, sarsa=args.sarsa, pretrain=pretrain)

        logger.logs['ep_train_td_errors'].append(td_errors)
        
        # every episode, test to see whether optimal policy is achieved
        if i >= args.num_pretrain:
            experiences, _ = utils.run_episode(agent, env, beta=1e6, agent_pos=agent_pos, goal_pos=goal_pos, update=False)
            iterations.append(i)
            logger.logs['ep_test_cum_reward'].append(args.gamma ** len(experiences))
            if len(experiences) == optimal_episode_length:
                break

    toc = time.time()

    logger.logs['num_eps_to_optimality'] = i - args.num_pretrain
    logger.logs['wall_clock_time'] = toc - tic

    print('\n\n Number of episodes to optimality: %d' % logger.logs['num_eps_to_optimality'])
    print('\n Wall clock time to convergence: %.2fs' % logger.logs['wall_clock_time'])

    if not args.no_logs:
        ## uncomment the line below for more comprehensive (but more memory-intensive) logging
        if args.full_agent_logging:
            logger.logs['agent'] = agent

        logger.logs['agent_prioritized_states'] = agent.prioritized_states
        if 'sr' in args.agent:
            logger.logs['agent_M'] = agent.get_M_states(beta=args.beta)
            logger.logs['agent_w'] = agent.w

        logger.logs['agent_Q'] = agent.Q

        if args.log_name is None:
            log_name = args.agent + '_' + str(uuid.uuid4())
        else:
            log_name = args.log_name

        picklefile = open('logs/four_rooms/' + log_name + '.pkl', 'wb')
        pickle.dump(logger, picklefile)
        picklefile.close()
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-a', '--agent', type=str, default='tdsr', help='agent type')
    parser.add_argument('-b', '--beta', type=float, default=5.0, help='softmax inverse temperature')
    parser.add_argument('-r', '--num_recall', type=int, default=10, help='number of recall steps')
    parser.add_argument('-g', '--grid_size', type=int, default=7, help='size of grid')
    parser.add_argument('--num_pretrain', type=int, default=0, help='pretraining episodes before recall')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate for agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for agent')
    parser.add_argument('--no_logs', action='store_true', help='storing logs as pkl')
    parser.add_argument('--full_agent_logging', action='store_true', help='whether to store full agent object')
    parser.add_argument('--log_name', type=str, help='custom name for log file')
    parser.add_argument('--sarsa', type=bool, default=True, help='are updates on-policy or off-policy')
    args = parser.parse_args()
    main(args)
