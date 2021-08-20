import numpy as np
import utils

class SimpleGrid:
    def __init__(self, size, block_pattern='empty', 
                 verbose=False, obs_mode='index'):
        self.verbose = verbose
        if np.isscalar(size):
            self.grid_size = (size, size)
        else:
            self.grid_size = size
        self.action_size = 4
        self.obs_mode = obs_mode
        self.state_size = self.grid_size[0] * self.grid_size[1]
        self.blocks = self.make_blocks(block_pattern)
        self.goal_pos = [[]]
        self.agent_pos = []
        self.done = None
        self.observations = None
        self.num_steps = 0
            
    def reset(self, goal_pos=None, agent_pos=None, reward_val=None):
        self.done = False
        if reward_val != None:
            self.reward_val = reward_val
        else:
            self.reward_val = 1.0
        if goal_pos != None:
            # turn goal_pos into list of goal_pos if just one given
            if type(goal_pos[0]) == int:
                goal_pos = [goal_pos]
            self.goal_pos = goal_pos
        else:
            self.goal_pos = [self.get_free_spot()]
        if agent_pos != None:
            self.agent_pos = agent_pos
        else:
            self.agent_pos = self.get_free_spot()
        
    def get_free_spot(self):
        free = False
        possible_x = np.arange(0, self.grid_size[0])
        possible_y = np.arange(0, self.grid_size[1])
        while not free:
            try_x = np.random.choice(possible_x, replace=False)
            try_y = np.random.choice(possible_y, replace=False)
            try_position = [try_x, try_y]
            if try_position not in self.all_positions:
                return try_position
        
    def make_blocks(self, pattern):
        if (pattern == 'four_rooms') or (pattern == 'four_rooms_blocked'):
            mid = int(self.grid_size[0] // 2)
            earl_mid = int(mid // 2)
            late_mid = mid+earl_mid + 1
            blocks_a = [[mid,i] for i in range(self.grid_size[0])]
            blocks_b = [[i,mid] for i in range(self.grid_size[0])]
            blocks = blocks_a + blocks_b
            self.bottlenecks = [[mid,earl_mid],[mid,late_mid],[earl_mid,mid],[late_mid,mid]]
            if pattern == 'four_rooms_blocked':
                self.bottlenecks.remove([earl_mid, mid])
            for bottleneck in self.bottlenecks:
                blocks.remove(bottleneck)
            return blocks
        if pattern == 'empty':
            self.bottlenecks = []
            return []
        if pattern == 'random':
            blocks = []
            for i in range(self.state_size // 10):
                blocks.append([np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])])
            self.bottlenecks = []
            return blocks
        if 'two_rooms' in pattern:
            mid = int(self.grid_size[0] // 2)
            blocks = [[mid,i] for i in range(self.grid_size[1])]
            if pattern == 'two_rooms_left':
                self.bottlenecks = [[mid, 0]]
            elif pattern == 'two_rooms_right':
                self.bottlenecks = [[mid, self.grid_size[1]-1]]
            else:
                self.bottlenecks = [[mid, 0], [mid, self.grid_size[1]-1]]
            for bottleneck in self.bottlenecks:
                blocks.remove(bottleneck)
            return blocks
        if pattern == 'sutton':
            assert self.grid_size[0] % 6 == 0
            assert self.grid_size[1] % 9 == 0
            k = self.grid_size[0] // 6
            assert k == (self.grid_size[1] // 9)

            blocks_a = []; blocks_b = []; blocks_c = []
            for x in range(k):
                for y in range(k):
                    blocks_b.append([4*k + x, 5*k + y])
                    for i in range(1, 4):
                        blocks_a.append([i*k + x, 2*k + y])
                    for i in range(3):
                        blocks_c.append([i*k + x, 7*k + y])

            blocks = blocks_a + blocks_b + blocks_c
            return blocks
        
    @property
    def grid(self):
        grid = np.zeros([self.grid_size[0], self.grid_size[1], 3])
        grid[self.agent_pos[0], self.agent_pos[1], 0] = 1
        for goal_pos in self.goal_pos:
            grid[goal_pos[0], goal_pos[1], 1] = 1
        for block in self.blocks:
            grid[block[0], block[1], 2] = 1
        return grid
    
    def move_agent(self, direction):
        if self.agent_pos not in self.goal_pos:
            new_pos = self.agent_pos + direction
            if self.check_target(new_pos):
                self.agent_pos = list(new_pos)
            
    def simulate(self, action):
        agent_old_pos = self.agent_pos
        reward = self.step(action)
        state = self.state
        self.agent_pos = agent_old_pos
        return state
        
    def check_target(self, target):
        x_check = target[0] > -1 and target[0] < self.grid_size[0]
        y_check = target[1] > -1 and target[1] < self.grid_size[1]
        block_check = list(target) not in self.blocks
        if x_check and y_check and block_check:
            return True
        else:
            return False
        
    @property
    def observation(self):
        if self.obs_mode == 'onehot':
            return utils.onehot(self.agent_pos[0] * self.grid_size[1] + self.agent_pos[1], self.state_size)
#        if self.obs_mode == 'twohot':
#            return self.twohot(self.agent_pos, self.grid_size)
#        if self.obs_mode == 'geometric':
#            return (2 * np.array(self.agent_pos) / (self.grid_size-1)) - 1 
        if self.obs_mode == 'visual':
            return env.grid
        if self.obs_mode == 'index':
            return self.agent_pos[0] * self.grid_size[1] + self.agent_pos[1]

    @property
    def goal(self):
        if self.obs_mode == 'onehot':
            goal_list = [utils.onehot(goal_pos[0] * self.grid_size[1] + goal_pos[1], self.state_size) for goal_pos in self.goal_pos]
            return goal_list
#        if self.obs_mode == 'twohot':
#            return self.twohot(self.goal_pos, self.grid_size)
#        if self.obs_mode == 'geometric':
#            return (2 * np.array(self.goal_pos) / (self.grid_size-1)) - 1 
        if self.obs_mode == 'visual':
            return env.grid
        if self.obs_mode == 'index':
            goal_list = [goal_pos[0] * self.grid_size[1] + goal_pos[1] for goal_pos in self.goal_pos]
        
    @property
    def all_positions(self):
        all_positions = self.blocks + self.goal_pos + [self.agent_pos]
        return all_positions
    
    def state_to_grid(self, state):
        vec_state = np.zeros([self.state_size])
        vec_state[state] = 1
        vec_state = np.reshape(vec_state, [self.grid_size[0], self.grid_size[1]])
        return vec_state
    
    def state_to_goal(self, state):
        return utils.onehot(state, self.state_size)
    
    def state_to_point(self, state):
        a = self.state_to_grid(state)
        b = np.where(a==1)
        c = [b[0][0],b[1][0]]
        return c
    
    def state_to_obs(self, state):
        if self.obs_mode == 'onehot':
            point = self.state_to_point(state)
            return utils.onehot(point[0] * self.grid_size[0] + point[1], self.state_size)
#        if self.obs_mode == 'twohot':
#            point = self.state_to_point(state)
#            return self.twohot(point, self.grid_size)
#        if self.obs_mode == 'geometric':
#            point = self.state_to_point(state)
#            return (2 * np.array(point) / (self.grid_size-1)) - 1 
        if self.obs_mode == 'visual':
            return self.state_to_grid(state)
        if self.obs_mode == 'index':
            return state

    def grid_to_state(self, coords):
        state = coords[0] * self.grid_size[1] + coords[1]
        return state
        
    def step(self, action):
        # 0 - Up
        # 1 - Down
        # 2 - Left
        # 3 - Right
        move_array = np.array([0,0])
        if action == 2:
            move_array = np.array([0,-1])
        if action == 3:
            move_array = np.array([0,1])
        if action == 0:
            move_array = np.array([-1,0])
        if action == 1:
            move_array = np.array([1,0])
        self.move_agent(move_array)
        self.num_steps += 1
        if self.agent_pos in self.goal_pos:
            self.done = True
            return self.reward_val
        else:
            return 0.0

    def state_to_goal(self, state):
        return self.state_to_obs(state)

class MultiGoalGrid(SimpleGrid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

