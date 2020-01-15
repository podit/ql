import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 11,
        }

# Plotting function to plot timestep rewards to show how the average agent
#   reward increases over the training period by the specified resolution
def plot(rewards, mins, maxs, uq, lq):
    #plt.ylim(0, 500)
    plt.xlabel('Episode', fontdict=font)
    plt.ylabel('Reward', fontdict=font)
    plt.plot(rewards, label='average')
    plt.plot(mins, label='minimum')
    plt.plot(maxs, label='maximum')
    plt.plot(uq, label='upr-qrt')
    plt.plot(lq, label='lwr-qrt')

    plt.legend()

    plt.show()

# Plot average reward against standard deviation of reward
def plotStd(rwd, std):
    plt.plot(rwd, std, ',b')
    plt.xlabel('Reward', fontdict=font)
    plt.ylabel('Standard Deviation', fontdict=font)

    plt.show()

# Plot notched box plot for rewards of each experiment along with average
def boxPlot(data, avg, ind):
    fig1, ax1 = plt.subplots()
    plt.xlabel('Experiment Index', fontdict=font)
    plt.ylabel('Reward', fontdict=font)
    ax1.boxplot(data, notch=True)
    ax1.plot(ind, avg, 'x--b', label='average')

    plt.legend()

    plt.show()

# Plot grid of histograms of frequency of average rewards for each experiment
def histExp(data, row, col, experiments):

    fig, ax = plt.subplots(row, col)
    e = 0

    for x in range(row):
        for y in range(col):
            ax[x, y].hist(data[e], 100)
            e += 1

    plt.show()

# Plot histogram of reward frequency
def hist(data):

    plt.hist(data, 100)

    plt.show()


