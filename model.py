"""Everything necessary to make a predictive coding agent."""
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from copy import deepcopy

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
relu = lambda x: jnp.maximum(0, x)


### Init #######################################################################
def init_params(hps) -> list:
	'''Returns arrays of initial activities and weights.
	Also returns a key for random generation.
	Inputs:
	Layer sizes and random seed.
	'''

	def init_weights(sizes: list, key) -> list:
		keys = random.split(key, num=len(sizes))
		return [jnp.array(random_layer_params(m, n, k)) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

	def random_layer_params(m: int, n: int, key):
		'''Returns a jax array of random numbers in (n, m) shape.
		This version is He initialization.
		'''
		scale = jnp.sqrt(2/m)
		return scale * random.normal(key, (n, m))

	activities = [jnp.zeros(s) for s in hps['sizes']]
	key = random.PRNGKey(hps['seed'])
	key, subkey = random.split(key)
	weights = init_weights(hps['sizes'], subkey)

	return activities, weights, key


### Usage #######################################################################
@jit
def ff(activities, weights):
	'''A one-step feedforward pass. Returns activities.
	Requires (l-1) calls to do full feedforward pass.
	'''
	new_acts = deepcopy(activities)

	for l in range(len(activities)-1):
		new_acts[l+1] = jnp.matmul(weights[l], relu(activities[l]))

	return new_acts


### Energy function and training ##########################################################################
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DEPRECATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# def energy(activities, weights, activity_history, hps):
# 	'''Calculates overall prediction loss.
# 	If beta==1 then noise is only based on the most recent activity.
# 	'''

# 	beta = hps['noise_beta']
# 	C = hps['denom_constant']

# 	# Update activity_history values
# 	# (1-beta) * previous_activity_history + beta * new_activity
# 	for l in range(len(activity_history)):
# 		activity_history[l] = (1-beta) * activity_history[l] + beta * activities[l]

# 	energy_sum = 0
# 	for l in range(len(activities)-1):
# 		energy_sum += jnp.sum((1/2) * ((activities[l+1] - jnp.matmul(weights[l], relu(activities[l]))) / (jnp.abs(activity_history[l+1]) + C)) ** 2)

# 	return energy_sum

def energy(activities, weights, activity_history, hps):
  '''Calculates overall prediction loss.
  If beta==1 then noise is only based on the most recent activity.

  This energy is for an architecture where the first input neuron is an "environment" neuron and only connected to the first neuron in the next layer.
  '''
  C = hps['denom_constant']

  # Set stop gradients at first layer of weights
  weights[0] = weights[0].at[5:,0].set(jax.lax.stop_gradient(0.))

  # Sum up energies over layers
  energy_sum = 0
  for l in range(len(activities)-1):
    energy_sum += jnp.sum((1/2) * ((activities[l+1] - jnp.matmul(weights[l], relu(activities[l])))/C )**2)#/ (activity_history[l+1] + C)) ** 2)
  return energy_sum


@jit
def update_act_history(activities, activity_history, hps):
  '''Update activity_history values
  (1-beta) * previous_activity_history + beta * new_activity
  Add relu so negative histories don't get incorporated
  '''
  beta = hps['beta']
  for l in range(len(activity_history)):
    activity_history[l] = (1-beta) * activity_history[l] + beta * relu(activities[l])
  return activity_history


@jit
def update_acts_energy(activities, weights, activity_history, hps):
	act_grads = grad(energy, argnums=0)(activities, weights, activity_history, hps)
	return [act - hps['gamma'] * d_act for act, d_act in zip(activities, act_grads)]


@jit
def update_weights_energy(activities, weights, activity_history,  hps):
	w_grads = grad(energy, argnums=1)(activities, weights, activity_history, hps)
	return [w - hps['alpha'] * d_w for w, d_w in zip(weights, w_grads)]


### Noise and clip functions ##########################################################################

@jit
def act_noise(activities, key, hps):
	''' Adds noise to each neuron.
	'''

	noise_scale = hps['noise_scale']

	new_activities = [[] for _ in activities]
	for l in range(len(activities)):

		key, subkey = random.split(key)
		noise = random.normal(subkey, activities[l].shape) * noise_scale
		new_activities[l] = activities[l] + noise

	return new_activities, key


@jit
def weight_noise(weights, key, hps):
	'''Adds some noise to the weights.
	'''
	new_weights = [[] for _ in weights]
	for l in range(len(weights)):
		key, subkey = random.split(key)
		noise = random.normal(subkey, weights[l].shape) * hps['weight_noise_scale']
		new_weights[l] = weights[l] + noise

	return new_weights, key


@jit
def weight_clip(weights, cap=2.):
	'''Makes sure weights don't go above some magnitude.
	'''
	new_weights = [[] for _ in weights]
	for l in range(len(weights)):
		new_weights[l] = jnp.clip(weights[l], -cap, cap)

	return new_weights


### Log ##########################################################################
class PCNLog():
	''' Records all current variables in the network.
	.acts[i][t][n] is a list of activities for the ith layer, timestep t, neuron n.
	'''
	def __init__(self, hps):

		self.acts = [[] for _ in range(len(hps['sizes']))]
		self.weights = [[] for _ in range(len(hps['sizes'])-1)]

	def record(self, activities, weights):
		[self.acts[i].append(acts) for i,acts in enumerate(activities)]
		[self.weights[i].append(weights) for i,weights in enumerate(weights)]

	def close(self):
		self.weights = [np.array(w) for w in self.weights]


# Adam: landscape code
def landscape_continuous(image, loc, adapt=False):
	'''Takes an image and returns values of new levers.
	Returns those levers as reward values for input to bandit.
	Also implements wraparound for the image as a "double cylinder".
	'''

	# If no adaptation, then subtract next value from current value.

	imshp = image.shape
	if adapt:
		nowloc = 0.
	else:
		nowloc = image[loc[0], loc[1]]

	# If top edge
	if loc[0]==0:
		up = image[imshp[0]-1, loc[1]] - nowloc
		down = image[loc[0]+1, loc[1]] - nowloc
	# If bottom edge
	elif loc[0]==imshp[0]-1:
		up = image[loc[0]-1, loc[1]] - nowloc
		down = image[0, loc[1]] - nowloc
	else:
		up = image[loc[0]-1, loc[1]] - nowloc
		down = image[loc[0]+1, loc[1]] - nowloc

	# If left edge
	if loc[1]==0:
		left = image[loc[0], imshp[1]-1] - nowloc
		right = image[loc[0], loc[1]+1] - nowloc
	# If right edge
	elif loc[1]==imshp[1]-1:
		left = image[loc[0], loc[1]-1] - nowloc
		right = image[loc[0], 0] - nowloc
	else:
		left = image[loc[0], loc[1]-1] - nowloc
		right = image[loc[0], loc[1]+1] - nowloc

	udlr = jnp.array([up, down, left, right])+1e-6
	return udlr


def move_continuous(loc, motors, udlr, imshape):
	'''Takes loc, motors, udlr, and imshape. Returns the reward of the new spot and the new location coordinates.
	Also implements wraparound for the image as a "double cylinder".
	'''

	lever_ind = jnp.argmax(motors)
	# Up
	if lever_ind==0:
		if loc[0]==0:
			new_loc = [imshape[0]-1, loc[1]]
		else:
			new_loc = [loc[0]-1, loc[1]]

	# Down
	elif lever_ind==1:
		if loc[0]==imshape[0]-1:
			new_loc = [0, loc[1]]
		else:
			new_loc = [loc[0]+1, loc[1]]

	# Left
	elif lever_ind==2:
		if loc[1]==0:
			new_loc = [loc[0], imshape[1]-1]
		else:
			new_loc = [loc[0], loc[1]-1]

	# Right
	elif lever_ind==3:
		if loc[1]==imshape[1]-1:
			new_loc = [loc[0], 0]
		else:
			new_loc = [loc[0], loc[1]+1]

	else:
		raise ValueError('Invalid lever input.')

	return udlr[lever_ind], new_loc, lever_ind


def landscape(image, loc):
	'''Takes an image and returns difference between current location and new levers UDLR.
	Returns those levers as reward values for input to bandit.
	'''
	imshp = image.shape

	# If top edge
	if loc[0]==0:
		up = 0.
		down = image[loc[0]+1, loc[1]] - image[loc[0], loc[1]]
	# If bottom edge
	elif loc[0]==imshp[0]-1:
		up = image[loc[0]-1, loc[1]] - image[loc[0], loc[1]]
		down = 0.
	else:
		up = image[loc[0]-1, loc[1]] - image[loc[0], loc[1]]
		down = image[loc[0]+1, loc[1]] - image[loc[0], loc[1]]

	# If left edge
	if loc[1]==0:
		left = 0.
		right = image[loc[0], loc[1]+1] - image[loc[0], loc[1]]
	# If right edge
	elif loc[1]==imshp[1]-1:
		left = image[loc[0], loc[1]-1] - image[loc[0], loc[1]]
		right = 0.
	else:
		left = image[loc[0], loc[1]-1] - image[loc[0], loc[1]]
		right = image[loc[0], loc[1]+1] - image[loc[0], loc[1]]

	udlr = jnp.array([up, down, left, right])+1e-6
	return udlr


def move(loc, motors, udlr, imshape):
	'''Takes loc, motors, udlr, and imshape. Returns the reward of the new spot and the new location coordinates.
	'''

	lever_ind = jnp.argmax(motors)
	# Up
	if lever_ind==0:
		if loc[0]==0:
			new_loc = [loc[0], loc[1]]
		else:
			new_loc = [loc[0]-1, loc[1]]

	# Down
	elif lever_ind==1:
		if loc[0]==imshape[0]-1:
			new_loc = [loc[0], loc[1]]
		else:
			new_loc = [loc[0]+1, loc[1]]

	# Left
	elif lever_ind==2:
		if loc[1]==0:
			new_loc = [loc[0], loc[1]]
		else:
			new_loc = [loc[0], loc[1]-1]

	# Right
	elif lever_ind==3:
		if loc[1]==imshape[1]-1:
			new_loc = [loc[0], loc[1]]
		else:
			new_loc = [loc[0], loc[1]+1]

	else:
		raise ValueError('Invalid lever input.')

	return udlr[lever_ind], new_loc, lever_ind


def landscape_rot(image, loc, orientation):
	'''Takes an image and returns difference between current location and new levers, ROTATION SCHEME.
	Returns those levers as reward values for input to bandit.
	Levers are: rotate CCW pi/2, go forward, rotate CW pi/2 deg.
	Orientation goes: 0:0, 1:pi/2, 2:pi, 3:3pi/2
	'''
	imshp = image.shape
	original = image[loc[0], loc[1]]

	if orientation==0:
		if loc[1]==imshp[1]-1:
			forward_reward = 0.
		else:
			forward_reward = image[loc[0], loc[1]+1] - original

	elif orientation==1:
		if loc[0]==0:
			forward_reward = 0.
		else:
			forward_reward = image[loc[0]-1, loc[1]] - original

	elif orientation==2:
		if loc[1]==0:
			forward_reward = 0.
		else:
			forward_reward = image[loc[0], loc[1]-1] - original

	elif orientation==3:
		if loc[0]==imshp[0]-1:
			forward_reward = 0.
		else:
			forward_reward = image[loc[0]+1, loc[1]] - original

	else:
		raise ValueError('Invalid orientation')

	lever_rewards = jnp.array([0., forward_reward, 0.]) + 1e-6

	return lever_rewards


def move_rot(loc, motors, lever_rewards, orientation, imshp):
	'''Takes loc, motors, lever rewards, and imshape. Returns the reward of the new spot, the new location coordinates, and the new orientation.
	'''

	lever_ind = jnp.argmax(motors)
	new_loc = deepcopy(loc)

	# No forward movement
	if lever_ind==0:
		# Turn CCW
		orientation = (orientation - 1) % 4
	elif lever_ind==2:
		# Turn CW
		orientation = (orientation + 1) % 4

	# Forward movement
	else:
		if orientation==0:
			if loc[1]!=imshp[1]-1:
				new_loc[1] += 1

		elif orientation==1:
			if loc[0]!=0:
				new_loc[0] -= 1

		elif orientation==2:
			if loc[1]!=0:
				new_loc[1] -= 1

		elif orientation==3:
			if loc[0]!=imshp[0]-1:
				new_loc[0] += 1

	return lever_rewards[lever_ind], new_loc, orientation, lever_ind


def bandit(motors, rewards=[0.1, 0.01, 1.]):
	lever_ind = jnp.argmax(motors)
	return rewards[lever_ind], lever_ind


### Jonah's functions ###
def get_environment_state(t, timescale):

	"""
	This function returns an environmental state, either 0 or 1, that switches
	according to the timescale specified.

	TODO: edit function to allow for n states that can take values specified by an array

	Arguments:
	t: current time
	timescale: time that passes between state switches
	"""

	# Start at 0, switch every timescale units of time
	if (t // timescale) % 2 == 0:
		return 0
	else:
		return 1


def get_environment_stimuli(environment_state, reward0, reward1):

	"""
	This function takes in the current environment state and outputs the
	associated bandit rewards, either reward0 for state=0 or reward1 for
	state=1.

	TODO: edit to allow for n rewards based on n states from edited
		get_environment_state function
	"""

	# State 1 is associated with reward1
	if environment_state:
		return reward1
	else:
		return reward0


## Define function to get rolling average over previous n timesteps
def get_rolling_average(array, n):

	# Define array to hold rolling average
	rolling_avg = np.copy(array)

	# Loop through array
	for i in range(len(array)-n+1):

		# Get average over up to n previous timesteps
		window_size = min(i, n)
		rolling_avg[i] = np.nanmean(array[i-window_size:i])

	return rolling_avg


## Define function to quantify reaction time
def get_reaction_times(sensory_input, timesteps, timescale, n, pct):

	"""
	Rough quantification of reaction time of agent to changing environment.

	Reaction time is defined as the number of timesteps from the change in environment
	until the agent is getting the reward for n steps pct% of the time. If the agent
	never switches to the other environment, the maximum time n is assigned.

	Return an array the reaction time for each environment change. Note that these
	will always be >=n, by definition.
	"""

	# Normalize rewards so that they are zero and one (fix for generality later)
	sensory_input = (sensory_input - np.min(sensory_input)) / (np.max(sensory_input) - np.min(sensory_input))

	# Define number of transitions
	num_transitions = timesteps // timescale

	# Separate levers for each segment of constant environment
	sensory_input_per_env = np.array(sensory_input).reshape(int(len(sensory_input) / timescale), timescale)

	# Initialize array to store reaction times
	reaction_times = np.zeros(num_transitions)

	# Loop through stationary environment phases
	for env in range(num_transitions):

		# Get rolling average of up to previous n timesteps for this environment phase
		rolling_input_per_env = get_rolling_average(sensory_input_per_env[env], n=n)

		# Get locations where the agent is has been getting the reward 90% of the time over n steps
		reward_locs = np.argwhere(rolling_input_per_env[n:] > pct)

		if len(reward_locs) != 0:
			# Get first time this occurs
			reaction_times[env] = np.min(reward_locs) + n
		else:
			# This never occurs, assign max time
			reaction_times[env] = n

	return reaction_times

