import numpy as np
from timeit import default_timer as timer

# Import plotting functions
import plotKew as plt

# Import single and double Q-Learning classes
from sinKew import SinKew
from dblKew import DblKew

initialisation = 'uniform'      # uniform, ones, zeros, random
policy = 'q_lrn'                # q_lrn, sarsa

doubleFlag = False
eDecay = False
log = False

profileFlag = True
verboseFlag = False
renderFlag = False

renderTrain = False

environment = 'MountainCar-v0'     # CartPole-v1, MountainCar-v0

cont_os = True
cont_as = False

dis = 8

resolution = 20
res = 0

maxSteps = 500
n_tests = 100

pen = 2                        # penalty value
exp = -0.75
length = 1

episodes = 1000
gamma = 0.99
alpha = 0.5
decay = 2
epsilon = 0.1

if eDecay:
    # Set epsilon start value
    epsilon_s = 0.5

    # Calculate the decay period
    eps_start = 1
    eps_end = episodes // decay

    # Calculate decay rate
    e_decay_rate = epsilon_s / (eps_end - eps_start)

data_points = episodes / resolution

# Start timer, set run length, set length of hyperparameters, and define
#   corresponding lists of hyperparameters to be used
start = timer()

runs = 100
    
# If so create aggregate array to store values for runs
aggr_rewards = np.zeros(runs)
aggr_stds = np.zeros(runs)
aggr_ts_r = np.zeros((runs, int(data_points)))
aggr_ts_r_min = np.zeros((runs, int(data_points)))
aggr_ts_r_max = np.zeros((runs, int(data_points)))

# Initialise double or single Q-learning class dependent on the flag provided
if doubleFlag:
    q = DblKew(dis, policy, log, verboseFlag)
else:
    q = SinKew(dis, policy, log, verboseFlag)

for r in range(runs):
    dp = 0
    timestep_reward = np.zeros(int(data_points))
    timestep_reward_min = np.zeros(int(data_points))
    timestep_reward_max = np.zeros(int(data_points))

    q.init_env(initialisation, cont_os, cont_as, environment,
            resolution)
    
    start_split = timer()
    if eDecay: epsilon = epsilon_s
    for episode in range(episodes):
        episode += 1
        
        q.lrn(epsilon, episode, pen, exp, length, alpha, gamma, maxSteps,
                renderTrain)

        if eDecay:
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
    

    if renderFlag: input('Start testing (rendered)')
    avg_rwd, std_rwd = q.test_qtable(n_tests, maxSteps, renderFlag) 
 
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

print('Total average reward:',
        np.average(aggr_rewards),
        np.std(aggr_rewards), 'Stds:',
        np.average(aggr_stds), np.std(aggr_stds))

# Print hyper parameters for testing
print('Episodes:', episodes, 'Gamma:', gamma, 'Alpha:',
        alpha)
if eDecay: print('Decaying Epsilon Start:', epsilon_s, 'Decay:', decay, 'Rate:',
        e_decay_rate)
else: print('Epsilon:', epsilon)
print('------------==========================------------')

# End timer and print time
end = timer()
print('Time:', end-start)
print('Discretisation Factor:', dis)
# Denote the method flag provided upon completion
print('Method used:', policy)
print('Double?:', doubleFlag)
input('Show plots')
plt.plotAll(aggr_ts_r, aggr_ts_r_min, aggr_ts_r_max, policy)
plt.plot(np.mean(aggr_ts_r, axis=0), np.mean(aggr_ts_r_min, axis=0),
        np.mean(aggr_ts_r_max, axis=0), policy)
