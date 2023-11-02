"""Deploy a number of agents in the moving peak experiment"""
import os
import sys
import pickle
import numpy as np
import jax.numpy as jnp

sys.path.append('/n/home04/aboesky/ramanthan/Predictive_Coding')

from copy import deepcopy
from model import init_params, PCNLog, update_acts_energy, update_weights_energy, weight_clip, weight_noise, act_noise, energy, landscape, move


def run_agent():
	"""Run and record the life of an agent"""
	print('Initializing!!!')
	results_dir = sys.argv[-2]
	agent_i = sys.argv[-1]

	timesteps = 100_000
	sizes = [1, 30, 4]

	initial_input = jnp.array([0.5])

	hps = {

		'sizes' : sizes,

		'gamma' : 0.01, # Activity update rate
		'alpha' : 0.01, # Weight update rate

		'noise_beta'  : 0.01,  # Activity history update rate for noise scale
		'denom_constant' : 0.25, # Constant to add to denominator in energy term
		'noise_scale' : 0.01, # Activity noise scale

		'weight_noise_scale' : 0.01, # Weight noise scale

		'seed' : int(agent_i),

	}

	activities, weights, key = init_params(hps)
	activity_history = deepcopy(activities)

	log = PCNLog(hps)
	lever_history = []
	energies = []

	activities[0] = initial_input

	all_coordinates = np.zeros((timesteps+1, 2)).astype(int)

	# Get countours
	with open('/n/ramanathan_lab/aboesky/reward_contours/contours_1000.pkl', 'rb') as f:
		contours = pickle.load(f)

	# Liftime loop
	print('Running Liftetime!!!')
	ctour_indx = 0
	for t in range(timesteps):
		if t%10_000==0:
			print(t)

		# Every tenth timestep, advance the gaussian one step so the period is 10,000
		if t%10==0: 
			ctour_indx += 1
			if ctour_indx == len(contours):  # If on last index, restart
				ctour_indx = 0

		log.record(activities, weights)

		levers = landscape(contours[ctour_indx], all_coordinates[t])

		reward, all_coordinates[t+1], lever = move(all_coordinates[t], activities[-1], levers, contours[0].shape)
		lever_history.append(lever)

		activities = update_acts_energy(activities, weights, activity_history, hps)
		activities, activity_history, key = act_noise(activities, activity_history, key, hps)

		weights = update_weights_energy(activities, weights, activity_history, hps)
		weights, key = weight_noise(weights, key, hps)
		weights = weight_clip(weights, cap=1.)

		activities[0] = jnp.array([reward])
		energies.append(energy(activities, weights, activity_history, hps))

	# Save data
	print('Storying data!!!')
	with open(os.path.join(results_dir, f'agent_{agent_i}.pkl'), 'wb') as f:
		pickle.dump(np.array(all_coordinates), f)


if __name__=='__main__':
	run_agent()
