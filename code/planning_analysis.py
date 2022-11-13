import numpy as np

# WARNING: This file assumes there are only 2 processes (at test time)!

planning_time_file = "planning_times.txt"
NUM_DESIRED_EPISODES = 250 
MAX_STEPS = 200

env_1_step = -1
env_2_step = -1
env_1_episode_time = 0
env_2_episode_time = 0
env_1_times = []
env_2_times = []
num_episodes_gotten_1 = 0
num_episodes_gotten_2 = 0
t = 0

with open(planning_time_file, 'r') as f:
    line = f.readline()
    while line != '': 

        if line == "Policy\n":
            line = f.readline() # Read the next line
            t = float(line)
       
        elif line == "Step\n": 
            line = f.readline() # Read the next line
            step = int(line)
            if env_1_step == -1 and env_2_step == -1: # Just started reading the file 
                env_1_step = step  # Assign this to be the first env 
                line = f.readline() # Read the next lines
                line = f.readline() 
                step = int(line)
                env_2_step = step  # Assign this to be the second env
                env_1_episode_time += t
                env_2_episode_time += t
            elif step == env_1_step + 1:
                env_1_step = step  # Step belongs to env 1
                env_1_episode_time += t
            elif step == env_2_step + 1:
                env_2_step = step  # Step belongs to env 2
                env_2_episode_time += t

        elif line == "Did not reach goal\n":
            line = f.readline() # Read the next line
            step = int(line)
            if step == env_1_step + 1:  # Env 1 did not reach goal
                print("ENV 1", step)
                num_episodes_gotten_1 += 1
                # Reset
                env_1_episode_time = 0
                env_1_step = 0
            elif step == env_2_step + 1:  # Env 2 did not reach goal
                print("ENV 2", step)
                num_episodes_gotten_2 += 1
                # Reset
                env_2_episode_time = 0
                env_2_step = 0
    
        elif line == "Reached Goal\n":
            line = f.readline() # Read the next line
            step = int(line)
            if step == env_1_step + 1:  # Reached goal belongs to env 1
                print("ENV 1", step)
                if num_episodes_gotten_1 < NUM_DESIRED_EPISODES:
                    env_1_times.append(env_1_episode_time/(env_1_step + 1))
                num_episodes_gotten_1 += 1
                # Reset
                env_1_episode_time = 0
                env_1_step = 0
            elif step == env_2_step + 1:  # Reached goal belongs to env 2 
                print("ENV 2", step)
                if num_episodes_gotten_2 < NUM_DESIRED_EPISODES:
                    env_2_times.append(env_2_episode_time/(env_2_step + 1))
                num_episodes_gotten_2 += 1
                # Reset
                env_2_episode_time = 0
                env_2_step = 0
    
            
        line = f.readline()
           

print(env_1_times, len(env_1_times))
print(env_2_times, len(env_2_times))

env_1_times.extend(env_2_times)
final_times = np.array(env_1_times)
print("\n\nAverage planning time over", len(final_times), "successful episodes:", np.mean(final_times))