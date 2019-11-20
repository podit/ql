import numpy as np
from timeit import default_timer as timer

def do(q, runs, episodes, resolution, dataPoints, profileFlag, eDecayFlag,
        gamma, alpha, epsilon, decay, epsilonDecay, eDecayStart, eDecayEnd,
        eDecayRate, penalty, exponent, length, renderFlag):

    # Create aggregate array to store values for run length
    aggr_rewards = np.zeros(runs)
    aggr_stds = np.zeros(runs)
    aggr_ts_r = np.zeros((runs, int(dataPoints)))
    aggr_ts_r_min = np.zeros((runs, int(dataPoints)))
    aggr_ts_r_max = np.zeros((runs, int(dataPoints)))

    for r in range(runs):
        dp = 0
        timestep_reward = np.zeros(int(dataPoints))
        timestep_reward_min = np.zeros(int(dataPoints))
        timestep_reward_max = np.zeros(int(dataPoints))

        q.init_env(resolution)
        
        start_split = timer()
        if eDecayFlag: epsilon = epsilonDecay
        for episode in range(episodes):
            episode += 1
            
            q.lrn(epsilon, episode, penalty, exponent, length, alpha, gamma)

            if eDecayFlag:
                # Decay epsilon values during epsilon decay range
                if eDecayEnd >= episode >= eDecayStart:
                    epsilon -= eDecayRate
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
        avg_rwd, std_rwd = q.test_qtable() 
     
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

    return 
