import gym
import numpy as np

# Get the discrete state from the state supplied by the environment
def get_discrete_state(state, env, discrete_os_win_size):
    #print(state, env.observation_space.low, discrete_os_win_size)
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    #print(discrete_state)
    return tuple(discrete_state.astype(np.int))

env = gym.make("CartPole-v0").env
env.reset()

high = np.float64(env.observation_space.high)
low = np.float64(env.observation_space.low)

print(high, low)

disos = [50] * len(env.observation_space.high)
disoswin = (high - low) / disos

print(disos, disoswin)

while True:
    s, r, d, i = env.step(env.action_space.sample())

    d_s = get_discrete_state(s, env, disoswin)

    print(s, d_s)

    input('...')
    
    if d:
        print('--------------------------------------====================#####')
        env.reset()



