import numpy as np
import utils

NUM_DESIRED_EPISODES = 500

id_tmp_dir = 'tmp/gym/7/'
try:
    # The first few times the results might not be written to file yet
    true_results = utils.load_results(id_tmp_dir)
    MAX_STEPS = 200

    all_results = np.array(true_results.apply(list))
    rewards = all_results[:, 1] 
    steps = all_results[:, 2] 
    num_episodes = len(steps)
    print("NUM EPISODES", num_episodes)
    steps = steps[:NUM_DESIRED_EPISODES]
    rewards = rewards[:NUM_DESIRED_EPISODES]
    print(steps)

    # all_success_rate = np.sum([step < MAX_STEPS - 1 for step in steps])/NUM_DESIRED_EPISODES
    # successful_episodes = [i for i in range(NUM_DESIRED_EPISODES) if steps[i] < MAX_STEPS - 1]
    all_success_rate = np.sum([step < MAX_STEPS - 1 for step in steps])/len(steps)
    successful_episodes = [i for i in range(len(steps)) if steps[i] < MAX_STEPS - 1]
    rewards = rewards[successful_episodes]
    steps = steps[successful_episodes]

    all_reward = np.mean(rewards)
    all_num_steps = np.mean(steps)
except IndexError:
    all_success_rate = 0
    all_reward = 0
    all_num_steps = 0

print("Success Rate Over All Environments:", all_success_rate * 100)
print("Average Reward Over All Environments:", all_reward)
print("Average Number of Steps Over All Environments:", all_num_steps)


    