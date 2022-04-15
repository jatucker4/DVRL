def main(_run,
         seed,
         opt,
         environment,
         rl_setting,
         log,
         algorithm,
         loss_function):
    """
    Entry point. Contains main training loop.
    """

    # Setup directory, vector of environments, actor_critic model, a 'rollouts' helper class
    # to compute target values and 'current_memory' which maintains the last action/observation/latent_state values
    # id_tmp_dir, envs, actor_critic, rollouts, current_memory = setup()
    
    tracked_rewards = {
        # Used to tracked how many screens weren't blanked out. Usually not needed
        'nr_observed_screens': collections.deque([0], maxlen=rl_setting['num_steps'] + 1),
        'episode_rewards': torch.zeros([rl_setting['num_processes'], 1]),
        'final_rewards': torch.zeros([rl_setting['num_processes'], 1]),
        'num_ended_episodes': 0
    }

    num_updates = int(float(loss_function['num_frames'])
                      // rl_setting['num_steps']
                      // rl_setting['num_processes'])

    # Count parameters
    num_parameters = 0


    # Initialise optimiser

    obs_loss_coef = algorithm['particle_filter']['obs_loss_coef']\
                    if algorithm['use_particle_filter']\
                    else algorithm['model']['obs_loss_coef']



    start = time.time()

    # Main training loop
    for j in range(num_updates):

        # Only predict observations sometimes: When predicted_times is a list of ints,
        # predict corresponding future observations, with 0 being the current reconstruction
        if log['save_reconstruction_interval'] > 0 and \
           float(obs_loss_coef) != 0 and \
           (j % log['save_reconstruction_interval'] == 0 or j == num_updates - 1):
            predicted_times = log['predicted_times']
        else:
            predicted_times = None

        # Main Loop over n_s steps for one gradient update
        tracked_values = collections.defaultdict(lambda: [])
        for step in range(rl_setting['num_steps']):
            
            old_observation = current_memory['current_obs']

            policy_return, current_memory, blank_mask, masks, reward = run_model(
                actor_critic=actor_critic,
                current_memory=current_memory,
                envs=envs,
                predicted_times=predicted_times)

            # Save in rollouts (for loss computation)
            rollouts.insert(step, reward, masks)

            # Track all bunch of stuff and also save intermediate images and stuff
            tracked_values = track_values(tracked_values, policy_return)

            if policy_return.predicted_obs_img is not None:
                save_images(policy_return, old_observation, id_tmp_dir, j, step)

            # Keep track of rewards
            final_rewards, avg_nr_observed, num_ended_episodes = track_rewards(
                tracked_rewards, reward, masks, blank_mask)

        # Compute bootstrapped value
        with torch.no_grad():
            policy_return = actor_critic(
                current_memory=current_memory,
                predicted_times=predicted_times,
                )

        next_value = policy_return.value_estimate

        # Compute targets (consisting of discounted rewards + bootstrapped value)
        rollouts.compute_returns(next_value, rl_setting['gamma'])

        # Compute losses:
        values = torch.stack(tuple(tracked_values['values']), dim=0)
        action_log_probs = torch.stack(tuple(tracked_values['action_log_probs']), dim=0)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(Variable(advantages.detach()) * action_log_probs).mean()

        # Average over batch and time
        avg_encoding_loss = torch.stack(tuple(tracked_values['encoding_loss'])).mean()
        dist_entropy = torch.stack(tuple(tracked_values['dist_entropy'])).mean()

        total_loss = (value_loss * loss_function['value_loss_coef']
                      + action_loss * loss_function['action_loss_coef']
                      - dist_entropy * loss_function['entropy_coef']
                      + avg_encoding_loss * loss_function['encoding_loss_coef'])


        # Only reset the computation graph every 'multiplier_backprop_length' iterations

        rollouts.after_update()


        # Logging = saving to database
        if j % log['log_interval'] == 0:
            end = time.time()
            utils.log_and_print(j, num_updates, end - start, id_tmp_dir, final_rewards, tracked_values,
                                num_ended_episodes, avg_nr_observed, avg_encoding_loss,
                                total_loss, value_loss, action_loss, dist_entropy,
                                rl_setting, algorithm, _run)
            utils.save_batches(current_memory, id_tmp_dir, j)

    # Save final model
    utils.save_model(id_tmp_dir, 'model_final', actor_critic, _run)