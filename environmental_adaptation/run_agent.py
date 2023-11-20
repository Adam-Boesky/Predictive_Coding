"""Run an agent with environmental adaptation."""
import os
import sys
import pickle
import numpy as np
import jax.numpy as jnp

sys.path.append('/n/home04/aboesky/ramanthan/Predictive_Coding')

from copy import deepcopy
from model import (init_params, PCNLog, update_acts_energy, update_weights_energy,
                   weight_clip, weight_noise, act_noise, energy, landscape, move,
                   landscape_continuous, move_continuous, get_environment_state, 
                   get_environment_stimuli, bandit, get_reaction_times, update_act_history)


def run_agent():
	"""
	This function runs the agent with 100,000 timesteps and an environment alternation rate
	of 10,000. It returns an array with the agent's reaction time after each environment change,
	as defined by the get_reaction_time function.

	TODO: fix this up to work for general agent network size, timescale, timesteps, etc. too tired rn
	"""
	### GET ENVIRONMENTAL PARAMETERS ###
	print('Initializing!!!')
	results_dir = sys.argv[-2]
	start_seed = sys.argv[-1]

	########################## RUN AGENT ##############################
	# Define rewards
	timesteps = 100_000
	sizes = [1, 30, 2]
	rewards = [0., 0.5]
	timescale = 10_000
	settle_steps = 10

	# Set initial reward and environment input
	initial_reward_input = 0.5

	initial_input = jnp.array([initial_reward_input])  # first neuron is for reward, second for environ


	hps = {

		'sizes' : sizes,

		'gamma' : 0.01, # Activity update rate
		'alpha' : 0.01, # Weight update rate

		'noise_beta'  : 0.01,  # Activity history update rate for noise scale
		'denom_constant' : 0.25, # Constant to add to denominator in energy term
		'noise_scale' : 0.01, # Activity noise scale

		'weight_noise_scale' : 0.01, # Weight noise scale

		'seed' : int(start_seed),

	}

	activities, weights, key = init_params(hps)
	activity_history = deepcopy(activities)

	log = PCNLog(hps)
	levers = np.zeros(timesteps)
	energies = np.zeros(timesteps)

	activities[0] = initial_input

	reward_history = jnp.array([0.])

	for t in range(timesteps):

		if t%10000==0:
			print(t)
			print(rewards)

		# Get environment stimulus
		#environment_stimulus = 0.  # TODO: change later

		# Get rewards from current environment
		current_env = get_environment_state(t, timescale)  # delete after we start updating stimulus. just use this! or we could introduce delay at some point! this is type of injury. cool
		rewards = get_environment_stimuli(current_env, [0., 0.5], [0.5, 0.])

		reward, levers[t] = bandit(activities[-1], rewards=rewards)

		for j in range(settle_steps):
			activities = update_acts_energy(activities, weights, activity_history, hps)
			activities, activity_history, key = act_noise(activities, activity_history, key, hps)

		weights = update_weights_energy(activities, weights, activity_history, hps)
		weights, key = weight_noise(weights, key, hps)
		weights = weight_clip(weights, cap=1.)

		activities[0] = jnp.array([reward])
		activity_history = update_act_history(activities, activity_history, hps)

		log.record(activities, weights)
		energies[t] = energy(activities, weights, activity_history, hps)


	###################### POST PROCESSING ##################################

	# Convert sensory activities (reward) log into array
	rewards_received = np.array([float(act[0]) for act in log.acts[0]])

	# Get reaction times
	reaction_times = get_reaction_times(rewards_received, timesteps, timescale, n=1000, pct=0.9)

	# Save data
	print('Storying data!!!')
	with open(os.path.join(results_dir, f'agent_{start_seed}.pkl'), 'wb') as f:
		pickle.dump(np.array(reaction_times), f)


if __name__ == '__main__':
	run_agent()
