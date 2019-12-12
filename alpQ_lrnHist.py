import numpy as np
from timeit import default_timer as timer

import math

# Import control script
import do as d
# Import plotting functions
import plotKew as plt
# Import single and double Q-Learning classes
from sinKew import SinKew
from dblKew import DblKew

initialisation = 'uniform'      # uniform, ones, zeros, random
policy = 'q_lrn'                # q_lrn, sarsa

doubleFlag = False
eDecayFlag = True
logFlag = False

profileFlag = False
verboseFlag = False

renderTest = False
renderTrain = False

environment = 'CartPole-v1'     # CartPole-v1, MountainCar-v0

contOS = True
contAS = False

discretisation = 8

resolution = 5

maxSteps = 500
nTests = 100

penalty = 0                        # penalty value
exponent = -0.75
length = 1

episodes = 1000
runs = 1000
    
gamma = 0.99
alpha = 0.5

epsilon = 0.1

# Set decay coefficient
decay = 2
# Set epsilon start value
epsilonDecay = 0.5
# Calculate the decay period
eDecayStart = 1
eDecayEnd = episodes // decay
# Calculate decay rate
eDecayRate = epsilonDecay / eDecayEnd

dataPoints = episodes / resolution

# Start timer, set run length, set length of hyperparameters, and define
#   corresponding lists of hyperparameters to be used
start = timer()

experiments = 12

aggr_rewards = [None] * experiments
avg = [None] * experiments

ind = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#decays = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
decays = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

for e in range(experiments):

    alpha = decays[e]

    # Initialise double or single Q-learning class dependent on the flag provided
    if doubleFlag: q = DblKew(initialisation, policy, environment, contOS, contAS,
                discretisation, maxSteps, nTests, logFlag, verboseFlag, renderTest,
                renderTrain)
    else: q = SinKew(initialisation, policy, environment, contOS, contAS,
                discretisation, maxSteps, nTests, logFlag, verboseFlag, renderTest,
                renderTrain)

    aggr_rewards[e], aggr_stds, aggr_ts_r, aggr_ts_r_min, aggr_ts_r_max,\
            aggr_ts_r_uq, aggr_ts_r_lq =\
            d.do(q, runs, episodes, resolution, dataPoints, profileFlag, eDecayFlag,
            gamma, alpha, epsilon, decay, epsilonDecay, eDecayStart, eDecayEnd,
            eDecayRate, penalty, exponent, length, renderTest)

    avg[e] = np.average(aggr_rewards[e])

    print('Total average reward:',
            np.average(aggr_rewards[e]),
            np.std(aggr_rewards[e]), 'Stds:',
            np.average(aggr_stds), np.std(aggr_stds))
    
    print(e+1, '/', experiments)

    # Print hyper parameters for testing
    print('Episodes:', episodes, 'Gamma:', gamma, 'Alpha:',
            alpha)
    if eDecayFlag: print('Decaying Epsilon Start:', epsilonDecay, 'Decay:', decay, 'Rate:',
            eDecayRate)
    else: print('Epsilon:', epsilon)
    print('------------==========================------------')

# End timer and print time
end = timer()
print('Time:', end-start)
print('Discretisation Factor:', discretisation)
# Denote the method flag provided upon completion
print('Method used:', policy)
print('Double?:', doubleFlag)
print(decays)
input('Show plots')
data = aggr_rewards
#plt.boxPlot(data, avg, ind)
row = int(math.floor(math.sqrt(experiments)))
col = int(experiments/row)
plt.histExp(data, row, col, experiments)
#plt.plotAll(aggr_ts_r, aggr_ts_r_min, aggr_ts_r_max, policy)
#plt.plot(np.mean(aggr_ts_r, axis=0), np.mean(aggr_ts_r_min, axis=0),
#        np.mean(aggr_ts_r_max, axis=0), np.mean(aggr_ts_r_uq, axis=0),
#        np.mean(aggr_ts_r_lq, axis=0), policy)
