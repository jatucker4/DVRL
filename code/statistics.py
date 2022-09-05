import numpy as np
import utils

id_tmp_dir = 'tmp/gym/10/'
try:
    # The first few times the results might not be written to file yet
    true_results = utils.load_results(id_tmp_dir)
    MAX_STEPS = 200

    all_results = np.array(true_results.apply(list))
    all_success_rate = all_results[:, 2] # Number of steps column
    denom = len(all_success_rate)
    all_success_rate = np.sum([steps < MAX_STEPS - 1 for steps in all_success_rate]) 
    all_success_rate /= denom
    all_num_steps = np.mean(all_results[:, 2])
    all_reward = np.mean(all_results[:, 1])  # Reward column
except IndexError:
    all_success_rate = 0
    all_reward = 0
    all_num_steps = 0

print("Success Rate Over All Environments:", all_success_rate * 100)
print("Average Reward Over All Environments:", all_reward)
print("Average Number of Steps Over All Environments:", all_num_steps)


    