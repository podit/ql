import sys
import gym
import numpy as np
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Initialize environment and Q-table
def init_env(initialisation = 'uniform'):
    env = gym.make("MountainCar-v0").env
    env.reset()

    os_high = env.observation_space.high
    os_low = env.observation_space.low

    # Discretize the observation space
    discrete_os_size = [21] * len(os_high)
    discrete_os_win_size = (os_high\
            - os_low) / discrete_os_size
    
    # Initialise q-table with supplied type
    if initialisation == 'ones':
        Q = np.ones((n_states, n_actions))
    elif initialisation == 'random':
        Q = np.random.uniform((n_states, n_actions))
    elif initialisation == 'zeros':
        Q = np.zeros((n_states, n_actions))
    elif initialisation == 'uniform':
        Q = np.random.uniform(low = -2, high = 0, size=(discrete_os_size\
                + [env.action_space.n]))

    return Q, env, os_high, os_low, discrete_os_size,\
            discrete_os_win_size, env.action_space.n

# Get the discrete state from the state supplied by the environment
def get_discrete_state(state, env, low, discrete_os_win_size):
    #print(state, low, discrete_os_win_size)
    discrete_state = ((state - low) / discrete_os_win_size)# - 0.5
    #print(discrete_state)
    return tuple(discrete_state.astype(np.int))

# e-Greedy algorithm for action selection from the q table by state with flag
#   to force greedy method for testing. Edited for decaying epsilon value
def e_greedy(Q, epsilon, n_ac, s, greedy=False):
    if greedy or np.random.rand() > epsilon:
        a = np.argmax(Q[s])
    else:
        a = np.random.randint(0, n_ac)

    return a

# Standard Q-Learning control method 
def q_lrn(alpha, gamma, epsilon, e_decay_rate, eps_start, eps_end, episodes,\
        max_steps, n_tests, initialisation, dis_st, dis_ac,\
        renderFlag=True, test=True):
    # Initialise environment to get Q-table and action and state information
    Q, env, s_os_high, s_os_low, s_os_size, s_os_win, n_actions =\
            init_env(initialisation)
    
    # Create list to store the timestep rewards
    timestep_reward = []
    timestep_reward_min = []
    timestep_reward_max = []
    resolution = 100
    res = 0
    timestep_reward_resolution = np.zeros(resolution)
    
    # Iterate through episodes
    for episode in range(episodes):
        # Reset environment for new episode and get initial discretized state
        d_s = get_discrete_state(env.reset(), env, s_os_low, s_os_win)
        
        # Reset render to false
        render = False
        # Every <resolution> episodes apart from the 1st check if render flag
        #   is set. Print the episode and epsilon values
        if episode % resolution == 0 and episode != 0:
            print(episode, epsilon)

            if renderFlag == True:
                render = True
        
        # Create values for recording rewards and task completion
        total_reward = 0
        done = False
        
        # Loop the task until task is completed or max steps are reached
        while True:
            # If render is true render environment
            if render:
                env.render()

            # Get action using e-Greedy method and discretized state
            a = e_greedy(Q, epsilon, n_actions, d_s) 

            # Get discretized next state from the chosen action and record reward
            s_, reward, done, info = env.step(a)
            total_reward += reward
            d_s_ = get_discrete_state(s_, env, s_os_low, s_os_win)

            # If the task is not completed update Q by max future Q-values
            if not done:
                # Get the maximum future Q value for the next state
                max_future_q = np.max(Q[d_s_])
                
                # Perform Bellman equation to back-propagate Q-values
                Q[d_s + (a, )] = (1 - alpha) * Q[d_s + (a, )] + alpha *\
                        (reward + gamma * max_future_q)
            
            # If task is completed set Q-value to zero so no penalty is applied
            if done:
                Q[d_s + (a, )] = 0
                env.reset()
                
                # Iterate until the resolution is hit
                if res == resolution:
                    res = 0
                else:
                    timestep_reward_resolution[res] = total_reward
                    res += 1
                
                # When the resolution is hit record average values
                if episode % resolution == 0 and episode != 0:
                    print(np.average(timestep_reward_resolution),\
                            np.min(timestep_reward_resolution),\
                            np.max(timestep_reward_resolution))
                    timestep_reward.append(np.average(\
                            timestep_reward_resolution))
                    timestep_reward_min.append(np.min(\
                            timestep_reward_resolution))
                    timestep_reward_max.append(np.max(\
                            timestep_reward_resolution))
                
                # If render is true close the environment rendering
                if render:
                    print(s_, d_s_, a, reward)
                    #input('lolo')
                    env.close()

                break    

            # Set next state to current state (Q-Learning) control policy
            d_s = d_s_

        # Decay epsilon values during epsilon decay range
        if eps_end >= episode >= eps_start:
            epsilon -= e_decay_rate
            if epsilon < 0:
                epsilon = 0

    # Check if testing flag is passed (on by default)
    if test:
        # Call testing class and return metrics for display
        avg_rwd, std_rwd = test_qtable(Q, env, n_tests, n_actions,\
                renderFlag, s_os_win, s_os_low)

    # Return the time stepped rewards averages, max and min values with test
    #   results
    return timestep_reward, timestep_reward_min, timestep_reward_max, avg_rwd,\
            std_rwd

# Adapted SARSA control method 
def sarsa(alpha, gamma, epsilon, e_decay_rate, eps_start, eps_end, episodes,\
        max_steps, n_tests, initialisation, dis_st, dis_ac,\
        renderFlag=True, test=True):
    # Initialise environment to get q table and action and state information
    Q, env, s_os_high, s_os_low, s_os_size, s_os_win, n_actions =\
            init_env(initialisation)
    
    # Create list to store the timestep rewards
    timestep_reward = []
    timestep_reward_min = []
    timestep_reward_max = []
    resolution = 100
    res = 0
    timestep_reward_resolution = np.zeros(resolution)

    # Iterate through episodes
    for episode in range(episodes):
        # Reset environment for new episode and get initial discretized state
        d_s = get_discrete_state(env.reset(), env, s_os_low, s_os_win)
        
        render = False

        if episode % resolution == 0 and episode != 0:
            print(episode, epsilon)

            if renderFlag:
                render = True

        # Create values for recording rewards and task completion
        total_reward = 0
        done = False
        
        # Get initial action using e-Greedy method
        a = e_greedy(Q, epsilon, n_actions, d_s) 

        # Loop the task until task is completed or max steps are reached
        while not done:
            if render == True:
                env.render()

            # Get next state from the chosen action and record reward
            s_, reward, done, info = env.step(a)
            total_reward += reward
            d_s_ = get_discrete_state(s_, env, s_os_low, s_os_win)

            # If the task is not completed update Q by max future Q-values
            if not done:
                max_future_q = np.max(Q[d_s_])

                # Select next action based on next discretized state using
                #   e-Greedy method
                a_ = e_greedy(Q, epsilon, n_actions, d_s_)
           
                # Perform Bellman equation to back-propagate Q-values
                Q[d_s + (a, )] = (1 - alpha) * Q[d_s + (a, )] + alpha *\
                        (reward + gamma * max_future_q)

            # If task is completed set Q-value to zero so no penalty is applied
            if done:
                Q[d_s + (a, )] = 0
                env.reset()

                if res == resolution:
                    res = 0
                else:
                    timestep_reward_resolution[res] = total_reward
                    res += 1

                if episode % resolution == 0 and episode != 0:
                    print(np.average(timestep_reward_resolution),\
                            np.min(timestep_reward_resolution),\
                            np.max(timestep_reward_resolution))
                    timestep_reward.append(np.average(\
                            timestep_reward_resolution))
                    timestep_reward_min.append(np.min(\
                            timestep_reward_resolution))
                    timestep_reward_max.append(np.max(\
                            timestep_reward_resolution))

                if render:
                    env.close()

                break

            # Set next state and action to current state and action (SARSA)
            d_s, a = d_s_, a_
 
        # Decay epsilon values during epsilon decay range
        if eps_end >= episode >= eps_start:
            epsilon -= e_decay_rate
            if epsilon < 0:
                epsilon = 0
    
    # Check if testing flag is passed (on by default)
    if test:
        # Call testing class and return metrics for display
        avg_rwd, std_rwd = test_qtable(Q, env, n_tests, n_actions,\
                renderFlag, s_os_win, s_os_low)

    # Return the time stepped rewards averages, max and min values with test
    #   results
    return timestep_reward, timestep_reward_min, timestep_reward_max, avg_rwd,\
            std_rwd

# Test function to test the Q-table
def test_qtable(Q, env, n_tests, n_actions, render, s_os_win, s_obs_low,\
        delay=0.1):
    
    #input('press enter to test')

    # Create array to store total rewards and steps for each test
    rewards = np.zeros(n_tests)
    # Set failed counter to zero
    #failed = 0

    # Iterate through each test
    for test in range(n_tests):

        # Reset the environment and get the initial state
        d_s = get_discrete_state(env.reset(), env, s_obs_low, s_os_win)
        # Set done flag to false
        done = False
        
        # Set step and reward counters to zero
        steps = 0
        total_reward = 0
        
        # Set e-greedy parameters greedy flag sets greedy method to be used
        epsilon = 0.9
        greedy = True
        
        # Loop until test conditions are met iterating the steps counter
        while not done:
            #steps += 1

            # Get action by e-greedy method
            a = e_greedy(Q, epsilon, n_actions, d_s, greedy)

            # Get state by applying the action to the environment and add reward
            s, reward, done, info = env.step(a)
            total_reward += reward
            d_s = get_discrete_state(s, env, s_obs_low, s_os_win)

            #env.render()

        # Record total rewards and steps
        rewards[test] = total_reward

    env.close()

    # Get averages of the steps and rewards and failure percentage for tests
    avg_rwd = np.average(rewards)
    std_rwd = np.std(rewards)

    # Print average test values for all tests
    print(f'Average reward:{avg_rwd}, std:{std_rwd}')

    return avg_rwd, std_rwd

# Plotting function to plot timesteo rewards to show how the average agent
#   reward increases over the training period by the specified resolution
def plot(rewards, mins, maxs, mthd):
    plt.title(mthd)
    plt.xlabel('Episodes')
    plt.ylabel('+Reward / -Steps')
    plt.plot(rewards)
    plt.plot(mins)
    plt.plot(maxs)

    plt.show()


# Start timer, set run length, set length of hyperparameters, and define
#   corresponding lists of hyperparameters to be used
start = timer()

runs = 3

initialisation = 'uniform'     # ones, zeros or random

max_steps = 500
n_tests = 100

dis_st = 21
dis_ac = 21

eps = 1
gammas = 1
alphas = 1
decays = 1
epsilons = 1

arr_eps = [1000, 7500, 10000, 15000]
arr_gamma = [0.95, 0.99, 0.9, 0.9999]
arr_alpha = [0.1, 0.02, 0.03, 0.04, 0.01, 0.06, 0.07, 0.08, 0.09, 0.1]
arr_decays = [1.1, 2, 1.5, 1.25]
arr_epsilons = [0.03, 0.2, 0.3, 0.5]

# Iterate through each level of hyperparameters, episodes, gammas, alphas,
# decay rates and epsilons
for e in range(eps):
    for g in range(gammas):
        for a in range(alphas):
            for d in range(decays):
                for l in range(epsilons):
                    start_split = timer()
                    # Set hyper parameters
                    episodes = arr_eps[e]
                    gamma = arr_gamma[g]
                    alpha = arr_alpha[a]
                    epsilon = arr_epsilons[l]
                    decay = arr_decays[d]

                    # Calculate the decay period
                    eps_start = 1
                    eps_end = episodes // decay
                    
                    # Calculate decay rate
                    e_decay_rate = epsilon / (eps_end - eps_start)
                    print(e_decay_rate)

                    # Check if runs is greater then 3 to a void indexing errors
                    if runs >= 3:
                        # If so create aggregate array to store values for runs
                        aggr_rewards = np.zeros(runs)
                        aggr_stds = np.zeros(runs)

                    # Iterate through runs
                    for i in range(runs):
                        # Check of first argument is passed
                        if len(sys.argv) > 1:
                            # Check method flag for s (SARSA) or q (Q-Learning)
                            if sys.argv[1] == 's':
                                # Check second argument is passed
                                if len(sys.argv) > 2:
                                    # Check if render flag is set or not
                                    if sys.argv[2] == 'r':
                                        # Start SARSA function with render flag
                                        rewards, mins, maxs, avg_rwd, std_rwd\
                                                = sarsa(alpha, gamma, epsilon,\
                                                e_decay_rate, eps_start,\
                                                eps_end, episodes, max_steps,\
                                                n_tests, initialisation, dis_st,\
                                                dis_ac, renderFlag=True)
                                    elif sys.argv[2] == 'n':
                                        rewards, mins, maxs, avg_rwd, std_rwd\
                                                = sarsa(alpha, gamma, epsilon,\
                                                e_decay_rate, eps_start,\
                                                eps_end, episodes, max_steps,\
                                                n_tests, initialisation, dis_st,\
                                                dis_ac, renderFlag=False)
                                    else:
                                        print(sys.argv[2], 'is not an option \
                                                use "r" to render tests')
                                else:
                                    print('Select render options "r" or "n" at \
                                            position 2')
                            
                            # Same as above for Q-Learning function
                            elif sys.argv[1] == 'q':
                                if len(sys.argv) > 2:
                                    if sys.argv[2] == 'r':
                                        rewards, mins, maxs, avg_rwd, std_rwd\
                                                = q_lrn(alpha, gamma, epsilon,\
                                                e_decay_rate, eps_start,\
                                                eps_end, episodes, max_steps,\
                                                n_tests, initialisation, dis_st,\
                                                dis_ac, renderFlag=True)
                                    elif sys.argv[2] == 'n':
                                        rewards, mins, maxs, avg_rwd, std_rwd\
                                                = q_lrn(alpha, gamma, epsilon,\
                                                e_decay_rate, eps_start,\
                                                eps_end, episodes, max_steps,\
                                                n_tests, initialisation, dis_st,\
                                                dis_ac, renderFlag=False)
                                    else:
                                        print(sys.argv[2], 'is not valid, use \
                                                "r" or "n" to render tests')
                                else:
                                    print('Select render options "r" or "n" at\
                                             position 2')

                            else:
                                print(sys.argv[1], 'is not valid, use "q" or\
                                         "s"')
                        
                        else:
                            print('Select a method "q" or "s" at position 1')

                        # Check of third argument is passed
                        if len(sys.argv) > 3:
                            # Check plot flag and plot data if so
                            if sys.argv[3] == 'p':
                                plot(rewards, mins, maxs, sys.argv[1])
                            elif sys.argv[3] == 'n':
                                pass
                            else:
                                print(sys.argv[3], 'is not valid, use "p" or \
                                        "n" to plot rewards')
                        else:
                            print('Select plot options "p" or "n" at position\
                                    3')
                    
                        # If the runs threshold is met record testing values
                        if runs >= 3:
                            aggr_rewards[i] = avg_rwd
                            aggr_stds[i] = std_rwd
 
                    # If runs threshold is met print the averages and standard
                    #   deviation of the average and standard deviations of
                    #   rewards and standard deviations for run length
                    if runs >= 3:
                        print('Total average reward:',\
                                np.average(aggr_rewards),\
                                np.std(aggr_rewards), 'Stds:',\
                                np.average(aggr_stds), np.std(aggr_stds))
                    
                    # Print hyper parameters for testing
                    print('Episodes:', episodes, 'Gamma:', gamma, 'Alpha:',\
                            alpha)
                    print('Epsilon:', epsilon, 'Decay:', decay, 'Rate:',\
                            e_decay_rate)
                    print('------------==========================------------')
                    
                    # Calculate split (total runs) time and report
                    end_split = timer()
                    segment = end_split - start_split
                    print('Split time:', segment)

# End timer and print time
end = timer()
print('Time:', end-start)
# Denote the method flag provided upon completion
print('Method used:', sys.argv[1])


