import copy
import argparse
import numpy as np
import numpy.random as npr
import progressbar
import time
import pickle
import uuid
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prioritizedsr import algs, utils
from prioritizedsr.gridworld import SimpleGrid

MAX_TRAINING_EPISODES = 100
NUM_BLOCKSTEPS        = 40

def agent_factory(args, state_size, action_size):
    if args.agent == 'tdsr':
        agent = algs.TDSR(state_size, action_size, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynasr':
        agent = algs.DynaSR(state_size, action_size, 10, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynasrplus':
        agent = algs.DynaSR(state_size, action_size, 10, kappa=1e-3, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'pssr':
        agent = algs.PSSR(state_size, action_size, 10, learning_rate=args.lr, gamma=args.gamma, goal_pri=True)
    elif args.agent == 'mpssr':
        agent = algs.PSSR(state_size, action_size, 10, learning_rate=args.lr, gamma=args.gamma, goal_pri=False)
    elif args.agent == 'tdq':
        agent = algs.TDQ(state_size, action_size, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynaq':
        agent = algs.DynaQ(state_size, action_size, 10, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'dynaqplus':
        agent = algs.DynaQ(state_size, action_size, 10, kappa=1e-3, learning_rate=args.lr, gamma=args.gamma)
    elif args.agent == 'psq':
        agent = algs.PSQ(state_size, action_size, 10, learning_rate=args.lr, gamma=args.gamma)
    else:
        raise ValueError('Invalid agent type: %s' % args.agent)

    return agent

def main(args):
    logger = utils.Logger('four_rooms_detour', vars(args))
    npr.seed(args.seed)
    env = SimpleGrid(args.grid_size, block_pattern='four_rooms', obs_mode='index')
    agent = agent_factory(args, env.state_size, env.action_size)

    logger.logs['ep_train_td_errors'] = []

    iterations = []

    agent_pos = [0, 0]
    goal_pos = [0, args.grid_size -1]

    tic = time.time()

    ## train on the regular four rooms task
    print('Initial training...')
    for i in progressbar.progressbar(range(MAX_TRAINING_EPISODES)):
        experiences, td_errors = utils.run_episode(agent, env, beta=args.beta, goal_pos=goal_pos, sarsa=False)
        logger.logs['ep_train_td_errors'].append(td_errors)

    ## introduce the block
    env = SimpleGrid(args.grid_size, block_pattern='four_rooms_blocked', obs_mode='index')
    # set recall back
    agent.num_recall = args.num_recall
    mid = int(args.grid_size // 2)
    earl_mid = int(mid // 2)
    agent_detour_pos = [earl_mid, mid-1]

    print('\nIntroducing wall...')
    for i in progressbar.progressbar(range(NUM_BLOCKSTEPS)):
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
        logger.logs['ep_train_td_errors'].append([td_error])

    toc = time.time()

    experiences, td_errors = utils.run_episode(agent, env, beta=1e6, agent_pos=agent_pos, goal_pos=goal_pos, update=False)

    logger.logs['zero_shot_ep_length'] = len(experiences)
    logger.logs['wall_clock_time'] = toc - tic

    print('\n\n Zero shot episode length: %d' % logger.logs['zero_shot_ep_length'])
    print('\n Wall clock time to convergence: %.2fs' % logger.logs['wall_clock_time'])

    if not args.no_logs:
        ## uncomment the line below for more comprehensive (but more memory-intensive) logging
        # logger.logs['agent'] = agent

        logger.logs['agent_prioritized_states'] = agent.prioritized_states
        if 'sr' in args.agent:
            logger.logs['agent_M'] = agent.get_M_states(beta=args.beta)
            logger.logs['agent_w'] = agent.w
            logger.logs['agent_Q'] = agent.M @ agent.w
        else:
            logger.logs['agent_Q'] = agent.Q

        if args.log_name is None:
            log_name = args.agent + '_' + str(uuid.uuid4())
        else:
            log_name = args.log_name

        picklefile = open('logs/four_rooms_detour/' + log_name + '.pkl', 'wb')
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
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate for agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for agent')
    parser.add_argument('--no_logs', action='store_true', help='storing logs as pkl')
    parser.add_argument('--log_name', type=str, help='custom name for log file')
    args = parser.parse_args()
    main(args)
