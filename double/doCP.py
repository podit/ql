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

def plotAll(rewards, mins, maxs, mthd):
    shap = rewards.shape
    for r in range(shap[0]):
        plt.title(mthd)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.plot(rewards[r])
        plt.plot(mins[r])
        plt.plot(maxs[r])

    plt.show()


initialisation = 'uniform'      # uniform, ones, zeros, random
policy = 'q_lrn'                # q_lrn, sarsa

log = False
pen = -2                        # penalty value
exp = -0.5
length = 10

profileFlag = True
verboseFlag = False
renderFlag = False

renderTrain = False

environment = 'CartPole-v1'     # CartPole-v1, 

cont_os = True
cont_as = False

dis = 8

resolution = 25
res = 0

maxSteps = 2500
n_tests = 100

episodes = 1000
gamma = 0.99
alpha = 0.5
decay = 2
epsilon_s = 1

# Calculate the decay period
eps_start = 1
eps_end = episodes // decay

# Calculate decay rate
e_decay_rate = epsilon_s / (eps_end - eps_start)

data_points = episodes / resolution

# Start timer, set run length, set length of hyperparameters, and define
#   corresponding lists of hyperparameters to be used
start = timer()

runs = 1000
    
# Check if runs is greater then 3 to a void indexing errors
if runs >= 3:
    # If so create aggregate array to store values for runs
    aggr_rewards = np.zeros(runs)
    aggr_stds = np.zeros(runs)
    aggr_ts_r = np.zeros((runs, int(data_points)))
    aggr_ts_r_min = np.zeros((runs, int(data_points)))
    aggr_ts_r_max = np.zeros((runs, int(data_points)))

q = Kew(dis, policy, log, verboseFlag)

for r in range(runs):
    dp = 0
    timestep_reward = np.zeros(int(data_points))
    timestep_reward_min = np.zeros(int(data_points))
    timestep_reward_max = np.zeros(int(data_points))

    q.init_env(initialisation, cont_os, cont_as, environment,
            resolution)
    
    start_split = timer()
    epsilon = epsilon_s
    for episode in range(episodes):
        episode += 1
        
        q.lrn(epsilon, episode, pen, exp, length, alpha, gamma, maxSteps,
                renderTrain)

        # Decay epsilon values during epsilon decay range
        if eps_end >= episode >= eps_start:
            epsilon -= e_decay_rate
            if epsilon < 0:
                epsilon = 0

        if episode % resolution == 0 and episode != 0:
            timestep_reward[dp] = np.average(
                    q.timestep_reward_res)
            timestep_reward_min[dp] = np.min(
                    q.timestep_reward_res)
            timestep_reward_max[dp] = np.max(
                    q.timestep_reward_res)
            dp += 1
    

    #input('press to tesst')
    avg_rwd, std_rwd = q.test_qtable(n_tests, maxSteps, renderFlag) 
 
    # If the runs threshold is met record testing values
    if runs >= 3:
        aggr_rewards[r] = avg_rwd
        aggr_stds[r] = std_rwd
        aggr_ts_r[r] = timestep_reward
        aggr_ts_r_min[r] = timestep_reward_min
        aggr_ts_r_max[r] = timestep_reward_max
    
    if profileFlag:
        # Calculate split (total runs) time and report profiling values
        end_split = timer()
        segment = end_split - start_split
        print('Run:', r)
        print(f'Average reward:{avg_rwd}, std:{std_rwd}')
        print('Split time:', segment)
        print('#--------========--------#')

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
input('press enter to see plots')
plotAll(aggr_ts_r, aggr_ts_r_min, aggr_ts_r_max, policy)
plot(np.mean(aggr_ts_r, axis=0), np.mean(aggr_ts_r_min, axis=0),
        np.mean(aggr_ts_r_max, axis=0), policy)
