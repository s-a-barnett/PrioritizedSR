import numpy as np
import numpy.random as npr
import algs
import progressbar
import utils
from gridworld import SimpleGrid
import argparse
import itertools

# fixed parameters
gamma = 0.9
lr = 1.0
beta  = 5
num_recall = 20

NUM_RUNS = 100
NUM_INTRO = 20
NUM_TRAIN = 50
MAX_EPISODE_LENGTH = 10000
streak_length = 5

grid_size = (6, 9)

def agent_factory(args, state_size, action_size):
    if args.agent == 'mdq':
        agent = algs.MDQ(state_size, action_size, num_recall=num_recall, learning_rate=lr, gamma=gamma, online=True)
    elif args.agent == 'qparsr':
        agent = algs.PARSR(state_size, action_size, num_recall=num_recall, learning_rate=lr, gamma=gamma, goal_pri=True)
    elif args.agent == 'mparsr':
        agent = algs.PARSR(state_size, action_size, num_recall=num_recall, learning_rate=lr, gamma=gamma, goal_pri=False)
    else:
        raise ValueError('Invalid agent type: %s' % args.agent)

    return agent

def count_replay_events(recalls_list):
    replay_counts = {'f': 0, 'r': 0}

    def flatten_recalls(recalls):
        return [item for elem in recalls for item in elem]

    def classify_recall(exp, next_exp):
        if exp[2] == next_exp[0]:
            recall_type = 'f'
        elif exp[0] == next_exp[2]:
            recall_type = 'r'
        else:
            recall_type = 'n'
        return recall_type

    def get_recall_string(recalls):
        string = ''
        for i in range(len(recalls)-1):
            string += classify_recall(recalls[i], recalls[i+1])
        return string

    def count_recall_streaks(recalls):
        streaks = {'f': 0, 'r': 0}
        recall_string = get_recall_string(recalls)
        streak_tuples = list((c, len(list(y))) for (c, y) in itertools.groupby(recall_string) if c in ['f', 'r'])
        for tup in streak_tuples:
            if tup[1] >= streak_length:
                streaks[tup[0]] += 1
        return streaks

    for recalls in recalls_list:
        recalls = flatten_recalls(recalls)
        streaks = count_recall_streaks(recalls)
        replay_counts['f'] += streaks['f']
        replay_counts['r'] += streaks['r']

    return replay_counts
    

def openmaze_sim(args):
    env = SimpleGrid(grid_size, block_pattern='sutton')
    goal_pos = [0, 8]
    agent = agent_factory(args, env.state_size, env.action_size)

    # accumulate every possible experience not in goal state
    agent.num_recall = 0
    for i, j, a in itertools.product(range(grid_size[0]), range(grid_size[1]), range(env.action_size)):
        if [i, j] not in (env.blocks + [goal_pos]):
            env.reset(agent_pos=[i, j], goal_pos=goal_pos, reward_val=0)
            state = env.observation
            action = a
            reward = env.step(action)
            state_next = env.observation
            done = env.done
            exp = (state, action, state_next, reward, done)
            agent.update(exp)

    # reinitialize T and Q for MDQ agent
    # T assumes a uniformly random policy
    # Q goes back to zeros
    if type(agent).__name__ == 'MDQ':
        agent.T = np.zeros((env.state_size, env.state_size))

        for i, j, a in itertools.product(range(grid_size[0]), range(grid_size[1]), range(env.action_size)):
            if [i, j] not in (env.blocks + [goal_pos]):
                env.reset(agent_pos=[i, j], goal_pos=goal_pos, reward_val=0)
                state = env.observation
                _ = env.step(a)
                state_next = env.observation
                agent.T[state, state_next] += (1 / env.action_size)

        # transitions from goal state to random valid starting state
        invalid_states = [env.grid_to_state(pos) for pos in (env.blocks + [goal_pos])]
        for i in range(env.state_size):
            if i not in invalid_states:
                agent.T[env.grid_to_state(goal_pos), i] = 1 / (env.state_size - len(invalid_states))

        agent.update_M()
        agent.Q = np.zeros_like(agent.Q)

    # complete training episodes
    start_recalls = []
    end_recalls = []
    reward_observed = False

    for i in range(NUM_TRAIN):
        env.reset(goal_pos=goal_pos, reward_val=(1+0.1*npr.randn()))
        state = env.observation
        for j in range(MAX_EPISODE_LENGTH):
            action = agent.sample_action(state, beta=beta)
            reward = env.step(action)
            if reward != 0:
                reward_observed = True
            state_next = env.observation
            done = env.done
            exp = (state, action, state_next, reward, done)
            state = state_next

            if ((j == 0) or done) and reward_observed:
                agent.num_recall = 20
            else:
                agent.num_recall = 0

            agent.update(exp)

            # capture recall information
            if j == 0:
                start_recalls.append(agent.recalled)
            if done:
                end_recalls.append(agent.recalled)
                break

    return start_recalls, end_recalls

def main(args):
    npr.seed(0)
    replay_events = {'start_forward': 0, 'end_forward': 0, 'start_reverse': 0, 'end_reverse': 0}
    for i in progressbar.progressbar(range(NUM_RUNS)):
        start_recalls, end_recalls = openmaze_sim(args)
        start_counts = count_replay_events(start_recalls)
        replay_events['start_forward'] += start_counts['f']
        replay_events['start_reverse'] += start_counts['r']
        end_counts = count_replay_events(end_recalls)
        replay_events['end_forward'] += end_counts['f']
        replay_events['end_reverse'] += end_counts['r']

    print(replay_events)

    with open(args.output, 'a') as f:
        f.write(str(replay_events))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='mdq', help='algorithm')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    main(args)

