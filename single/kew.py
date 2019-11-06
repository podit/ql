import gym
import math
import numpy as np

class Kew:
    def __init__(self, dis, verboseFlag):
        # Set discretisation factor
        self.dis = dis
        self.verbose = verboseFlag

    # Initialize environment and Q-table
    def init_env(self, initialisation, cont_os, cont_as, environment,
            resolution):

        # Create numpy array to store rewards for use in statistical tracking
        self.timestep_reward_res = np.zeros(resolution)

        # Set variables to be used to flag specific behaviour in the learning
        self.environment = environment

        self.cont_os = cont_os
        self.cont_as = cont_as

        # Initialize environment
        env = gym.make(environment).env
        env.reset()
        
        # If observation space is continuous do calculations to create
        #   corresponding bins for use with Q table
        if cont_os:
            self.os_high = env.observation_space.high
            self.os_low = env.observation_space.low

            # Set bounds for infinite observation spaces in 'CartPole-v1'
            if environment == 'CartPole-v1':
                self.os_high[1], self.os_high[3] = 5, 5
                self.os_low[1], self.os_low[3] = -5, -5

            # Discretize the observation space
            self.discrete_os_size = [self.dis] * len(self.os_high)
            self.discrete_os_win_size = (self.os_high\
                    - self.os_low) / self.discrete_os_size
        else:
            # Use number of observations if no discretization is required
            self.discrete_os_size = [env.observation_space.n]
        
        # The same for action space
        if cont_as:
            self.dis_centre = self.dis / 2

            self.as_high = env.action_space.high
            self.as_low = env.action_space.low

            self.discrete_as_size = [self.dis] * len(self.as_high)
            self.discrete_as_win_size = (self.as_high\
                    - self.as_low) / self.discrete_as_size
            self.action_n = self.dis
        else:
            self.discrete_as_size = [env.action_space.n]
            self.action_n = env.action_space.n
        
        # Initialise q-table with supplied type
        if initialisation == 'uniform':
            Q = np.random.uniform(low = -2, high = 0, size=(
                self.discrete_os_size + self.discrete_as_size))
        elif initialisation == 'random':
            Q = np.random.uniform((self.discrete_os_size,
                self.discrete_as_size))
        elif initialisation == 'zeros':
            Q = np.zeros((self.discrete_os_size,
                self.discrete_as_size))
        elif initialisation == 'ones':
            Q = np.ones((self.discrete_os_size,
                self.discrete_as_size))
        else:
            print('initialisation method not valid')

        return Q, env

    # Get the discrete state from the state supplied by the environment
    def get_discrete_state(self, state, env):
        discrete_state = ((state - self.os_low) / self.discrete_os_win_size)# - 0.5
        return tuple(discrete_state.astype(np.int))

    # Get the continuous action from the discrete action supplied by e-greedy
    def get_continuous_action(self, discrete_action):
        continuous_action = (discrete_action - self.dis_centre) *\
                self.discrete_as_win_size
        return continuous_action

    # e-Greedy algorithm for action selection from the q table by state with
    #   flag to force greedy method for testing. Takes input for decaying
    #   epsilon value. Gets the continuous action if needed
    def e_greedy(self, Q, epsilon, s, greedy=False):
        if greedy or np.random.rand() > epsilon:
            d_a = np.argmax(Q[s])
        else:
            d_a = np.random.randint(0, self.action_n)

        if self.cont_as:
            a = self.get_continuous_action(d_a)
        else:
            a = d_a

        return a, d_a

    # Perform training on the Q table for the given environment, called once per
    #   episode taking variables to control the training process
    def lrn(self, Q, env, epsilon, episode, resolution, res, policy, mode, pen,
            alpha, gamma, maxSteps, renderFlag):

        # Set vars used for checks in training
        steps = 0
        maxS = False
        done = False
        render = False
        
        # Reset environment for new episode and get initial discretized state
        if self.cont_os:
            d_s = self.get_discrete_state(env.reset(), env)
        else:
            s = env.reset()
            d_s = s

        if mode == 'log':
            modeL = True
            history_o = np.zeros((10, len(d_s)))
            history_a = np.zeros(10)
        else:
            modeL = False

        # Report episode and epsilon and set the episode to be rendered
        #   if the resolution is reached
        if episode % resolution == 0 and episode != 0:
            if self.verbose:
                print(episode, epsilon)
            if renderFlag:
                render = True

        # Create values for recording rewards and task completion
        total_reward = 0
        
        # Get initial action using e-Greedy method for SARSA policy
        if policy == 'sarsa':
            a, d_a = self.e_greedy(Q, epsilon, d_s)

        # Loop the task until task is completed or max steps are reached
        while not done:
            steps += 1
            if render:
                env.render()
            
            # Get initial action using e-Greedy method for Q-Lrn policy
            if policy == 'q-lrn':
                a, d_a = self.e_greedy(Q, epsilon, d_s)

            # Get next state from the chosen action and record reward
            s_, reward, done, info = env.step(a)
            total_reward += reward

            # Discretise state if observation space is continuous
            if self.cont_os:
                d_s_ = self.get_discrete_state(s_, env)
            else:
                d_s_ = s_

            if maxS:
                done = True

            # If the task is not completed update Q by max future Q-values
            if not done:
                max_future_q = np.max(Q[d_s_])

                # Select next action based on next discretized state using
                #   e-Greedy method for SARSA policy
                if policy == 'sarsa':
                    a_, d_a_ = self.e_greedy(Q, epsilon, d_s)
                
                # Perform Bellman equation to update Q-values
                Q[d_s + (d_a, )] = (1 - alpha) * Q[d_s + (d_a, )] + alpha *\
                        (reward + gamma * max_future_q)
            
            # If task is completed set Q-value to zero so no penalty is applied
            if done:
                if maxS:
                    pass
                elif modeL and steps >= 10 and epsilon == 0:
                    for i in range(10):
                        Q[history_o[i].astype(np.int) + (int(history_a[i]), )]\
                                += pen * math.exp(-.75) ** i
                else:
                    Q[d_s + (d_a, )] = pen
                
                env.reset()

                if res == resolution:
                    res = 0
                else:
                    self.timestep_reward_res[res] = total_reward
                    res += 1

                if self.verbose and episode % resolution == 0 and episode != 0:
                    print(np.average(self.timestep_reward_res),
                            np.min(self.timestep_reward_res),
                            np.max(self.timestep_reward_res))
                
                if render:
                    env.close()

            if modeL:
                history_o = np.roll(history_o, 1)
                history_a = np.roll(history_a, 1)
                history_o[0, ...] = d_s
                history_a[0, ...] = d_a

            if policy == 'q-lrn':
                # Set next state to current state (Q-Learning) control policy
                d_s = d_s_

            if policy == 'sarsa':
                # Set next state and action to current state and action (SARSA)
                d_s, d_a, a = d_s_, d_a_, a_
            
            # If max steps are reached complete episode and set max step flag
            if steps == maxSteps:
                maxS = True
        
        return Q, env, res

    # Test function to test the Q-table
    def test_qtable(self, Q, env, n_tests, maxSteps):
        #print('A1')
        # Create array to store total rewards and steps for each test
        rewards = np.zeros(n_tests)

        # Iterate through each test
        for test in range(n_tests):
            #print('A2')
            # Reset the environment and get the initial state
            d_s = self.get_discrete_state(env.reset(), env)
            
            # Set step and reward counters to zero and done flag
            steps = 0
            total_reward = 0
            done = False
            
            # Set greedy flag sets greedy method to be used by e-Greedy
            epsilon = 0
            greedy = True
            
            # Loop until test conditions are met iterating the steps counter
            while not done:
                #env.render()
                steps += 1
                #print('A3')
                # Get action by e-greedy method
                a, d_a = self.e_greedy(Q, epsilon, d_s, greedy)

                # Get state by applying the action to the environment and
                #   add reward
                s, reward, done, info = env.step(a)
                total_reward += reward
                d_s = self.get_discrete_state(s, env)

                if steps == maxSteps:
                    #print('cheater')
                    done = True
            #print('A4')
            # Record total rewards and steps
            rewards[test] = total_reward

        #env.close()

        # Get averages of the steps and rewards and failure percentage for tests
        avg_rwd = np.average(rewards)
        std_rwd = np.std(rewards)

        # Print average test values for all tests
        #print(f'Average reward:{avg_rwd}, std:{std_rwd}')

        return avg_rwd, std_rwd


