import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 11,
        }

# Plotting function to plot timesteo rewards to show how the average agent
#   reward increases over the training period by the specified resolution
def plot(rewards, mins, maxs, uq, lq):
    plt.ylim(0, 500)
    plt.xlabel('Episode', fontdict=font)
    plt.ylabel('Reward', fontdict=font)
    plt.plot(rewards, label='average')
    plt.plot(mins, label='minimum')
    plt.plot(maxs, label='maximum')
    plt.plot(uq, label='upper')
    plt.plot(lq, label='lower')

    plt.legend()
    #plt.setp(L.texts, fontdict=font)

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

def plotStd(rwd, std):
    plt.plot(rwd, std, ',b')
    plt.xlabel('reward', fontdict=font)
    plt.ylabel('standard deviation', fontdict=font)

    plt.show()

def boxPlot(data, avg, ind):
    fig1, ax1 = plt.subplots()
    ax1.boxplot(data, notch=True)
    ax1.plot(ind, avg)
    plt.show()

def histExp(data, row, col, experiments):

    fig, ax = plt.subplots(row, col)
    e = 0

    for x in range(row):
        for y in range(col):
            ax[x, y].hist(data[e], 100)
            e += 1
            #ax[x, y].title(str(e))

    plt.show()

def hist(data):

    plt.hist(data, 100)

    plt.show()

