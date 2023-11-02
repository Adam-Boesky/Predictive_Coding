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
def energy(activities, weights, activity_history, hps):
	'''Calculates overall prediction loss.
	If beta==1 then noise is only based on the most recent activity.
	'''

	beta = hps['noise_beta']
	C = hps['denom_constant']

	# Update activity_history values
	# (1-beta) * previous_activity_history + beta * new_activity
	for l in range(len(activity_history)):
		activity_history[l] = (1-beta) * activity_history[l] + beta * activities[l]

	energy_sum = 0
	for l in range(len(activities)-1):
		energy_sum += jnp.sum((1/2) * ((activities[l+1] - jnp.matmul(weights[l], relu(activities[l]))) / (jnp.abs(activity_history[l+1]) + C)) ** 2)

	return energy_sum


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
def act_noise(activities, activity_history, key, hps):
	''' Adds noise to each neuron based on its past activity.
	'''

	noise_scale = hps['noise_scale']

	new_activities = [[] for _ in activities]
	for l in range(len(activities)):

		key, subkey = random.split(key)
		noise = random.normal(subkey, activities[l].shape) * noise_scale
		new_activities[l] = activities[l] + noise

	return new_activities, activity_history, key


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
