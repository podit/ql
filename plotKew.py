import matplotlib.pyplot as plt


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


