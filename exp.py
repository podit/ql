import numpy as np
from timeit import default_timer as timer

# Import control script
import do as d
# Import plotting functions
import plotKew as plt
# Import single and double Q-Learning classes
from sinKew import SinKew
from dblKew import DblKew

initialisation = 'uniform'      # uniform, ones, zeros, random
policy = 'q_lrn'                # q_lrn, sarsa

doubleFlag = True
eDecayFlag = False
logFlag = False

profileFlag = True
verboseFlag = False

renderTest = False
renderTrain = False

environment = 'CartPole-v1'     # CartPole-v1, MountainCar-v0

contOS = True
contAS = False

discretisation = 8

resolution = 20

maxSteps = 500
nTests = 100

penalty = 2                        # penalty value
exponent = -0.75
length = 1

episodes = 1000
runs = 100
    
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
eDecayRate = epsilonDecay / (eDecayStart - eDecayEnd)

dataPoints = episodes / resolution

# Start timer, set run length, set length of hyperparameters, and define
#   corresponding lists of hyperparameters to be used
start = timer()

# Initialise double or single Q-learning class dependent on the flag provided
if doubleFlag: q = DblKew(initialisation, policy, environment, contOS, contAS,
            discretisation, maxSteps, nTests, logFlag, verboseFlag, renderTest,
            renderTrain)
else: q = SinKew(initialisation, policy, environment, contOS, contAS,
            discretisation, maxSteps, nTests, logFlag, verboseFlag, renderTest,
            renderTrain)

aggr_rewards, aggr_stds, aggr_ts_r, aggr_ts_r_min, aggr_ts_r_max,\
        aggr_ts_r_uq, aggr_ts_r_lq =\
        d.do(q, runs, episodes, resolution, dataPoints, profileFlag, eDecayFlag,
        gamma, alpha, epsilon, decay, epsilonDecay, eDecayStart, eDecayEnd,
        eDecayRate, penalty, exponent, length, renderTest)

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
#plt.plotAll(aggr_ts_r, aggr_ts_r_min, aggr_ts_r_max, policy)
plt.plot(np.mean(aggr_ts_r, axis=0), np.mean(aggr_ts_r_min, axis=0),
        np.mean(aggr_ts_r_max, axis=0), np.mean(aggr_ts_r_uq, axis-0),
        np.mean(aggr_ts_r_lq, axis=0), policy)
