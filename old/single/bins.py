import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from kew import Kew

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


timestep_reward = []
timestep_reward_min = []
timestep_reward_max = []

initialisation = 'uniform'      # uniform, ones, zeros, random
policy = 'q-lrn'                # q-lrn, sarsa
mode = 'none'                   # none, log
pen = -2

renderFlag = False

environment = 'CartPole-v1'     # CartPole-v1, 

cont_os = True
cont_as = False

dis = 26
#binn = 10
#bins = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

resolution = 100
res = 0

maxSteps = 1000
n_tests = 100

episodes = 500
gamma = 0.99
alpha = 0.5
decay = 2
epsilon_s = 1

# Calculate the decay period
eps_start = 1
eps_end = episodes // decay

# Calculate decay rate
e_decay_rate = epsilon_s / (eps_end - eps_start)

# Start timer, set run length, set length of hyperparameters, and define
#   corresponding lists of hyperparameters to be used
start = timer()

runs = 1000
    
# Check if runs is greater then 3 to a void indexing errors
if runs >= 3:
    # If so create aggregate array to store values for runs
    aggr_rewards = np.zeros(runs)
    aggr_stds = np.zeros(runs)

q = Kew(dis)

for r in range(runs):
    #print('Run: ', r)
    
    Q, env = q.init_env(initialisation, cont_os, cont_as, environment,
            resolution)
    
    start_split = timer()
    epsilon = epsilon_s
    for episode in range(episodes):
        episode += 1
        
        Q, env, res = q.lrn(Q, env, epsilon, episode, resolution, res, policy,
                mode, pen, alpha, gamma, maxSteps, renderFlag)

        # Decay epsilon values during epsilon decay range
        if eps_end >= episode >= eps_start:
            epsilon -= e_decay_rate
            if epsilon < 0:
                epsilon = 0

        if episode % resolution == 0 and episode != 0:
            timestep_reward.append(np.average(
                    q.timestep_reward_res))
            timestep_reward_min.append(np.min(
                    q.timestep_reward_res))
            timestep_reward_max.append(np.max(
                    q.timestep_reward_res))
    

    #plot(timestep_reward, timestep_reward_min, timestep_reward_max, policy)
    #input('press to tesst')
    avg_rwd, std_rwd = q.test_qtable(Q, env, n_tests, maxSteps) 
 
    # If the runs threshold is met record testing values
    if runs >= 3:
        aggr_rewards[r] = avg_rwd
        aggr_stds[r] = std_rwd
    
    # Calculate split (total runs) time and report
    end_split = timer()
    segment = end_split - start_split
    #print('Split time:', segment)
    #print('#--------========--------#')

# If runs threshold is met print the averages and standard
#   deviation of the average and standard deviations of
#   rewards and standard deviations for run length
if runs >= 3:
    print('Total average reward:',
            np.average(aggr_rewards),
            np.std(aggr_rewards), 'Stds:',
            np.average(aggr_stds), np.std(aggr_stds))

# Print hyper parameters for testing
print('Episodes:', episodes, 'Gamma:', gamma, 'Alpha:',
        alpha)
print('Epsilon:', epsilon, 'Decay:', decay, 'Rate:',
        e_decay_rate)
print('------------==========================------------')

# End timer and print time
end = timer()
print('Time:', end-start)
print(dis)
# Denote the method flag provided upon completion
print('Method used:', policy)


