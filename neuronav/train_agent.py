import copy
import argparse
import numpy as np
import numpy.random as npr
import algs
import progressbar
import utils
from gridworld import SimpleGrid
from plotting import plot_place_fields
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import pickle
import uuid

MAX_TRAINING_EPISODES = 1000

def agent_factory(args, state_size, action_size):
    if args.agent == 'nomem':
        agent = algs.TDSR(state_size, action_size, learning_rate=args.lr, gamma=args.gamma, M_init=args.M_init)
    elif args.agent == 'dyna':
        agent = algs.DynaSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, M_init=args.M_init)
    elif args.agent == 'ps':
        agent = algs.PSSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, M_init=args.M_init)
    elif args.agent == 'mps':
        agent = algs.PSSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, M_init=args.M_init, goal_pri=False)
    elif args.agent == 'md':
        agent = algs.MDSR(state_size, action_size, args.num_recall, learning_rate=args.lr, gamma=args.gamma, M_init=args.M_init)
    else:
        raise ValueError('Invalid agent type: %s' % args.agent)

    return agent

def main(args):
    logger = utils.Logger(vars(args))
    npr.seed(args.seed)
    env = SimpleGrid(args.grid_size, block_pattern=args.env, obs_mode='index')
    agent = agent_factory(args, env.state_size, env.action_size)
    optimal_episode_length = 2 * (args.grid_size - 1)

    logger.logs['ep_test_cum_reward'] = []
    logger.logs['ep_train_m_errors'] = []
    logger.logs['ep_train_w_errors'] = []

    iterations = []
    optimal_return = args.gamma ** optimal_episode_length

    agent_pos = [0, 0]
    goal_pos = [args.grid_size -1, args.grid_size -1]

    tic = time.time()

    for i in progressbar.progressbar(range(MAX_TRAINING_EPISODES)):
        # train for an episode
        pretrain = True if i < args.num_pretrain else False
        experiences, m_errors, w_errors = utils.run_episode(agent, env, beta=args.beta, agent_pos=agent_pos, goal_pos=goal_pos, sarsa=True, pretrain=pretrain)

        logger.logs['ep_train_m_errors'].append(np.mean(m_errors))
        logger.logs['ep_train_w_errors'].append(np.mean(w_errors))
        
        # every 1 episode, test to see whether optimal policy is achieved
        if i % 1 == 0:
            experiences, _, _ = utils.run_episode(agent, env, beta=1e6, agent_pos=agent_pos, goal_pos=goal_pos, update=False)
            iterations.append(i)
            logger.logs['ep_test_cum_reward'].append(args.gamma ** len(experiences))
            if len(experiences) == optimal_episode_length:
                break

    toc = time.time()

    logger.logs['num_eps_to_optimality'] = i
    logger.logs['wall_clock_time'] = toc - tic

    print('\n\n Number of episodes to optimality: %d' % i)
    print('\n Wall clock time to convergence: %.2fs' % (toc - tic))

    if not args.no_logs:
        logger.logs['agent_w'] = agent.w
        logger.logs['agent_M'] = agent.get_M_states(beta=args.beta)
        logger.logs['agent_V'] = (agent.M @ agent.w).max(0)
        logger.logs['agent_prioritized_states'] = agent.prioritized_states

        if args.log_name is None:
            log_name = args.agent + '_' + str(uuid.uuid4())
        else:
            log_name = args.log_name

        picklefile = open('logs/' + log_name + '.pkl', 'wb')
        pickle.dump(logger, picklefile)
        picklefile.close()

    if args.save_figs:
        cmap = copy.copy(mpl.cm.get_cmap('viridis'))
        cmap.set_bad(color='white')

        Q = agent.M @ agent.w

        plt.imshow(utils.mask_grid(Q.max(0).reshape(args.grid_size, args.grid_size), env.blocks), cmap=cmap);
        plt.title(f'V(S) for {args.agent}');
        plt.savefig('figures/vs.png');
        plt.close();

        plot_place_fields(agent, env, beta=args.beta);
        plt.suptitle(f'Place fields for {args.agent}');
        plt.savefig('figures/placefields.png');
        plt.close();

        plt.plot(iterations, logger.logs['ep_test_cum_reward']);
        plt.plot([iterations[0], iterations[-1]], [optimal_return, optimal_return]);
        plt.title(f'Cumulative episode reward for {args.agent}');
        plt.savefig('figures/training_returns.png');
        plt.close();

        plt.imshow(utils.mask_grid(agent.prioritized_states.reshape(args.grid_size, args.grid_size), env.blocks), cmap=cmap);
        plt.title(f'Prioritized states for {args.agent}');
        plt.savefig('figures/prioritizedstates.png');
        plt.close();

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-a', '--agent', type=str, default='nomem', help='agent type')
    parser.add_argument('-b', '--beta', type=float, default=5.0, help='softmax inverse temperature')
    parser.add_argument('-r', '--num_recall', type=int, default=10, help='number of recall steps')
    parser.add_argument('-g', '--grid_size', type=int, default=7, help='size of grid')
    parser.add_argument('-M', '--M_init', default=None, help='SR initialization type')
    parser.add_argument('-e', '--env', type=str, default='four_rooms', help='environment')
    parser.add_argument('--num_pretrain', type=int, default=0, help='pretraining episodes before recall')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate for agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for agent')
    parser.add_argument('--save_figs', action='store_true', help='storing figures')
    parser.add_argument('--no_logs', action='store_true', help='storing logs as pkl')
    parser.add_argument('--log_name', type=str, help='custom name for log file')
    args = parser.parse_args()
    main(args)
