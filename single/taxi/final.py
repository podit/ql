import sys
import gym
import numpy as np
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Initialize environment and q table
def init_env(initialisation = 'ones'):
    env = gym.make("Taxi-v2").env
    n_states, n_actions = env.observation_space.n, env.action_space.n
    
    # Initialise q-table with supplied type
    if initialisation == 'ones':
        Q = np.ones((n_states, n_actions))
    elif initialisation == 'random':
        Q = np.random.uniform((n_states, n_actions))
    elif initialisation == 'zeros':
        Q = np.zeros((n_states, n_actions))

    return Q, env, n_states, n_actions

# e-Greedy algorithm for action selection from the q table by state with flag
#   to force greedy method for testing
def e_greedy(Q, epsilon, n_actions, s, greedy=False):
    if greedy or np.random.rand() < epsilon:
        a = np.argmax(Q[s])
    else:
        a = np.random.randint(0, n_actions)

    return a

# Standard Q-Learning control method 
def q_lrn(alpha, gamma, epsilon, episodes, max_steps, n_tests,\
        initialisation = 'ones', render=True, test=True):
    # Initialise environment to get q table and action and state information
    Q, env , n_states, n_actions = init_env(initialisation)
    
    # Create list to store the timestep rewards
    timestep_reward = []
    timestep_max = []
    timestep_min = []
    resolution = 10
    res = 0
    timestep_reward_resolution = np.zeros(resolution)

    # Iterate through episodes
    for episode in range(episodes):
        # Reset environment for new episode and get initial state
        s = env.reset()
        
        # Create values for recording rewards , steps and task completion
        total_reward = 0
        steps = 0
        done = False
        
        # Loop the task until task is completed or max steps are reached
        while True:
            steps += 1
            
            a = e_greedy(Q, epsilon, n_actions, s) 

            # Get next state from the chosen action and record reward
            s_, reward, done, info = env.step(a)
            total_reward += reward

            # Select next action greedily based on next state
            a_ = e_greedy(Q, epsilon, n_actions, s_, greedy=True)

            # If the task is completed do not use next q-value to update Q
            if done:
                Q[s, a] += alpha * (reward - Q[s, a])
            # If not done use the Bellman equation to update Q
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_]) - Q[s, a])

            # Set next state to current state (Q-Learning) control policy
            s = s_
            
            # If the task is completed break and record the reward for graphing
            if done:
                timestep_reward_resolution[res] = total_reward
                
                # If the resolution count is reached reset counter
                if res == resolution - 1:
                    res = 0
                # If not iterate the resolution counter
                else:
                    res += 1
                
                # Average the reward for the resolution and record for plotting
                if episode % resolution == 0:
                    timestep_reward.append(np.average(\
                            timestep_reward_resolution))
                    timestep_max.append(np.max(timestep_reward_resolution))
                    timestep_min.append(np.min(timestep_reward_resolution))
                
                break

            # If the max steps are reached (agent stuck in action loop)
            if steps == max_steps:
                timestep_reward_resolution[res] = total_reward

                if res == resolution - 1:
                    res = 0
                else:
                    res += 1

                if episode % resolution == 0:
                    timestep_reward.append(np.average(\
                            timestep_reward_resolution))
                
                break

    # Check if testing flag is passed (on by default)
    if test:
        # Call testing class and return metrics for display
        avg_rwd, avg_stp, failed = test_qtable(Q, env, n_tests, n_actions,\
                max_steps, render)
    
    # Return the time stepped rewards, averages for rewards and steps and the
    #   the failure percentage
    return timestep_reward, timestep_max, timestep_min, avg_rwd, avg_stp, failed

# Adapted SARSA control method 
def sarsa(alpha, gamma, epsilon, episodes, max_steps, n_tests,\
        initialisation = 'ones', render=True, test=True):
    # Initialise environment to get q table and action and state information
    Q, env , n_states, n_actions = init_env(initialisation)
    
    # Create list to store the timestep rewards
    timestep_reward = []
    resolution = 10
    res = 0
    timestep_reward_resolution = np.zeros(resolution)

    # Iterate through episodes
    for episode in range(episodes):
        # Reset environment for new episode and get initial state
        s = env.reset()
        
        # Create values for recording rewards , steps and task completion
        total_reward = 0
        steps = 0
        done = False
        
        # Get initial action using e_greedy method
        a = e_greedy(Q, epsilon, n_actions, s) 

        # Loop the task until task is completed or max steps are reached
        while True:
            steps += 1            

            # Get next state from the chosen action and record reward
            s_, reward, done, info = env.step(a)
            total_reward += reward

            # Select next action based on next state using 
            a_ = e_greedy(Q, epsilon, n_actions, s_)

            # If the task is completed do not use next q-value to update Q
            if done:
                Q[s, a] += alpha * (reward - Q[s, a])
            # If not done use the Bellman equation to update Q
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_]) - Q[s, a])

            # Set next state and action to current state and action
            s, a = s_, a_
            
            # If the task is completed break and record the reward for graphing
            if done:
                timestep_reward_resolution[res] = total_reward
                
                # If the resolution count is reached reset counter
                if res == resolution - 1:
                    res = 0
                # If not iterate the resolution counter
                else:
                    res += 1

                # Average the reward for the resolution and record for plotting
                if episode % resolution == 0:
                    timestep_reward.append(np.average(\
                            timestep_reward_resolution))
                
                break

            # If the max steps are reached (agent stuck in action loop)
            if steps == max_steps:
                timestep_reward_resolution[res] = total_reward
                
                if res == resolution - 1:
                    res = 0
                else:
                    res += 1

                if episode % resolution == 0:
                    timestep_reward.append(np.average(\
                            timestep_reward_resolution))
                
                break

    # Check if testing flag is passed (on by default)
    if test:
        # Call testing class and return metrics for display
        avg_rwd, avg_stp, failed = test_qtable(Q, env, n_tests, n_actions,\
                max_steps, render)
    
    # Return the time stepped rewards, averages for rewards and steps and the
    #   the failure percentage
    return timestep_reward, avg_rwd, avg_stp, failed

# Test function to test the Q-table
def test_qtable(Q, env, n_tests, n_actions, max_steps, render, delay=0.1):
    
    # Create array to store total rewards and steps for each test
    step_rewards = np.zeros([n_tests, n_tests])
    # Set failed counter to zero
    failed = 0

    # Iterate through each test
    for test in range(n_tests):

        # Reset the environment and get the initial state
        s = env.reset()
        # Set done flag to false
        done = False
        
        # Set step and reward counters to zero
        steps = 0
        total_reward = 0
        
        # Set e-greedy parameters greedy flag sets greedy method to be used
        epsilon = 0.9
        greedy = True
        
        # Loop until test conditions are met iterating the steps counter
        while True:
            steps += 1

            # Get action by e-greedy method
            a = e_greedy(Q, epsilon, n_actions, s, greedy)

            # If render flag has been passed render the environment
            if render:
                time.sleep(delay)
                env.render()
                print(f'Action {a} chosen for state {s}')

            # Get state by applying the action to the environment and add reward
            s, reward, done, info = env.step(a)
            total_reward += reward

            # If task is completed break
            if done:
                #print(f'Episode reward: {total_reward}, Steps: {steps}')
                # If render flag is set render and pause
                if render:
                    time.sleep(1)
                break

            # If step limit is reached end and record the failure
            if steps == max_steps:
                # If render flag is set render and wait for user input
                if render:
                    print(f'Failed on ep: {steps} performing action {a}')
                    env.render()
                    input()

                #total_reward = 0
                #steps = 0
                failed += 1
                break

        # Record total rewards and steps
        step_rewards[0, test] = total_reward
        step_rewards[1, test] = steps

    # Get averages of the steps and rewards and failure percentage for tests
    avg_rwd = np.average(step_rewards[0])
    avg_stp = np.average(step_rewards[1])
    fail_percent = (failed / n_tests) * 100

    # Print average test values for all tests
    print(f'Average reward:{avg_rwd},steps:{avg_stp},failed:{failed},\
            {fail_percent}%')

    return avg_rwd, avg_stp, failed

# Plotting function to plot timesteo rewards to show how the average agent
#   reward increases over the training period by the specified resolution
def plot(data, maxim, minim, mthd):
    plt.title(mthd)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.plot(data)
    plt.plot(maxim)
    plt.plot(minim)

    plt.show()


# Start timer, set run length, set length of hyperparameters, and define
#   corresponding lists of hyperparameters to be used
start = timer()

runs = 10

initialisation = 'ones'     # ones, zeros or random

epsilon = 0.9

max_steps = 500
n_tests = 1000

eps = 1
gammas = 1
alphas = 1

arr_eps = [50000, 2500, 5000, 10000, 50000]
arr_gamma = [0.999, 0.99, 0.9, 0.9999]
arr_alpha = [0.05, 0.02, 0.03, 0.04, 0.01, 0.06, 0.07, 0.08, 0.09, 0.1]

# Iterate through each level of hyperparameters, episodes, gammas and alphas
for e in range(eps):
    for g in range(gammas):
        for a in range(alphas):

            # Set hyper parameters
            episodes = arr_eps[e]
            gamma = arr_gamma[g]
            alpha = arr_alpha[a]
            
            # Check if runs os greater then 3 to a void indexing errors
            if runs >= 3:
                # If so create aggregate array to average the total runs value
                aggr = np.zeros([runs, runs, runs])

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
                                # Start SARSA function with render flag set
                                data, maxim, minim, avg_rwd, avg_stp, failed = sarsa(alpha,\
                                        gamma, epsilon, episodes, max_steps,\
                                        n_tests, initialisation, render=True)
                            elif sys.argv[2] == 'n':
                                data, maxim, minim, avg_rwd, avg_stp, failed = sarsa(alpha,\
                                        gamma, epsilon, episodes, max_steps,\
                                        n_tests, initialisation, render=False)
                            else:
                                print(sys.argv[2], 'is not an option use "r" \
                                        to render tests')
                        else:
                            print('Select render options "r" or "n" at \
                                    position 2')
                    
                    # Same as above for Q-Learning function
                    elif sys.argv[1] == 'q':
                        if len(sys.argv) > 2:
                            if sys.argv[2] == 'r':
                                data, maxim, minim, avg_rwd, avg_stp, failed = q_lrn(alpha,\
                                        gamma, epsilon, episodes, max_steps,\
                                        n_tests, initialisation, render=True)
                            elif sys.argv[2] == 'n':
                                data, maxim, minim, avg_rwd, avg_stp, failed = q_lrn(alpha,\
                                        gamma, epsilon, episodes, max_steps,\
                                        n_tests, initialisation, render=False)
                            else:
                                print(sys.argv[2], 'is not valid, use "r" or \
                                        "n" to render tests')
                        else:
                            print('Select render options "r" or "n" at \
                                    position 2')

                    else:
                        print(sys.argv[1], 'is not valid, use "q" or "s"')

                    # Check of third argument is passed
                    if len(sys.argv) > 3:
                        # Check plot flag and plot data if so
                        if sys.argv[3] == 'p':
                            plot(data, maxim, minim, sys.argv[1])
                        elif sys.argv[3] == 'n':
                            pass
                        else:
                            print(sys.argv[3], 'is not valid, use "p" or \
                                    "n" to plot rewards')
                    else:
                        print('Select plot options "p" or "n" at position 3')
                
                else:
                    print('Select a method "q" or "s" at position 1')

                # If the runs threshold is met record values from testing
                if runs >= 3:
                    aggr[0, i], aggr[1, i], aggr[2, i] = avg_rwd, avg_stp,\
                            failed

            # Print the method used for clarity
            print(sys.argv[1])
            
            # If runs threshold is met print the averages and standard
            #   deviation of the rewards, steps and failures
            if runs >= 3:
                print('Total average reward:', np.average(aggr[0]),\
                        np.std(aggr[0]), 'steps:', np.average(aggr[1]),\
                        np.std(aggr[1]), 'failures:', np.average(aggr[2]),\
                        np.std(aggr[2]))
            
            # Print hyper parameters for testing
            print('Episodes:', episodes, 'Gamma:', gamma, 'Alpha:', alpha)

# End timer and print time
end = timer()
print('Time:', end-start)
