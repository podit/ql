# Import numpy for array managment and timeit to time execution
import numpy as np
from timeit import default_timer as timer

# Import control script
import doExp as d
# Import plotting functions
import plotKew as plt
# Import single and double Q-Learning classes
from expSinKew import SinKew
from dblKew import DblKew

# Set initialisation policy for Q-table
initialisation = 'uniform'      # uniform, ones, zeros, random

# Set on-policy (sarsa) or off-policy (q_lrn) control method for training
policy = 'q_lrn'                # q_lrn, sarsa

# Control flags for double Q-learning, epsilon decay and expontntial penalties
doubleFlag = False
eDecayFlag = False
logFlag = False

# Flags to report each run and each resolution step
# Report run number and average reward from test for each run
profileFlag = True
# Report episode and epsilon value as well as reward for each resolution step
verboseFlag = False

# Render flags for testing and training
renderTest = False
# Render training at each resolution step
renderTrain = False

# Set openai gym environment (CartPole and MountainCar have been tested)
environment = 'CartPole-v1'     # CartPole-v1, MountainCar-v0

# Flags for continuous observation and action spaces
contOS = True
contAS = False
# Discretisation factor to set number of bins for continuous
#   observation/action spaces depending on flags
discretisation = 8

# Set resolution for bins to record performance every <resolution> epsiodes
resolution = 5

# Set max steps in the environment
maxSteps = 500
# Set number of tests to be run (average is reported)
nTests = 100

# Set penalty to be applied at episode completion (positive acts as reward)
penalty = 0

# Used when logFlag is enabled
# Set exponent for exponential penalty and length of applied steps
exponent = -0.05
length = 5

# Set number of episodes and runs to be completed by the agent
episodes = 100
# Episodes constitute run length before testing
runs = 6

# Set hyper-parameters for use in bellman equation for updating Q table
# Discount factor
gamma = 0.99
# Learning rate
alpha = 0.5

# Set epsilon value for constant e-greedy method
epsilon = 0.1

# Used whrn eDecayFlag is enabled
# Set decay coefficient
decay = 1.5
# Set epsilon start value
epsilonDecay = 0.25
# Calculate the decay period
eDecayStart = 1
eDecayEnd = episodes // decay
# Calculate decay rate
eDecayRate = epsilonDecay / eDecayEnd

# Create number of individual data points for run length
dataPoints = episodes / resolution

# Start experiment timer
start = timer()

# Initialise double or single QL class with the doubleFlag value provided
if doubleFlag: q = DblKew(initialisation, policy, environment, contOS, contAS,
            discretisation, maxSteps, nTests, logFlag, verboseFlag, renderTest,
            renderTrain)
else: q = SinKew(initialisation, policy, environment, contOS, contAS,
            discretisation, maxSteps, nTests, logFlag, verboseFlag, renderTest,
            renderTrain)

# Run experiment passing relevent variables to do script to run QL,
#   recording performance of tests and training for plotting
aggr_rewards, aggr_stds, aggr_ts_r, aggr_ts_r_min, aggr_ts_r_max,\
        aggr_ts_r_uq, aggr_ts_r_lq =\
        d.do(q, runs, episodes, resolution, dataPoints, profileFlag, eDecayFlag,
        gamma, alpha, epsilon, decay, epsilonDecay, eDecayStart, eDecayEnd,
        eDecayRate, penalty, exponent, length, renderTest)

# Print the average reward and standard deviation of test results for
#   all the runs over the experiment
print('Total average reward:',
        np.average(aggr_rewards),
        np.std(aggr_rewards), 'Stds:',
        np.average(aggr_stds), np.std(aggr_stds))

# Print experiment number to show progress
#print('Expreiment:', e, 'of:', experiments)
# Print experiment parameters
print('Episodes:', episodes, 'Gamma:', gamma, 'Alpha:',
        alpha, 'Penalty:', penalty)
if eDecayFlag: print('Decaying Epsilon Start:', epsilonDecay, 'Decay:',
        decay, 'Rate:', eDecayRate)
else: print('Epsilon:', epsilon)
if logFlag: print('Exponential penalties exponent:', exponent, 'Length:',
        length)
print('------------==========================------------')

# End timer and print time
end = timer()
print('Time:', end-start)
print('Discretisation Factor:', discretisation)
# Denote the method flag and environment upon completion
print('Method used:', policy)
print('Double?:', doubleFlag)
print('Environment:', environment)
data = aggr_rewards
row = 1
col = 1
experiments = 1
input('hostograme')
print(np.max(q.N))
plt.hist(data)
