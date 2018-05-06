from __future__ import print_function, division, absolute_import

import numpy as np

"""
Missing things:
 - Function to generate equally-weighted posterior samples.
 - Proper saving off of outputs (the current output file is only set up for two parameters).
 - Ability to resume from previously generated outputs.
 - Calcualtion of uncertainty on evidence value.
 - Stopping criteria based on evidence uncertainty (at the moment just runs to the max number of iterations).
"""

class nested_sampler:
	""" A basic functioning nested sampling class. """

	def __init__(self, likelihood_func, prior_trans, ndim, n_live=400, maxiter=None, outfile_prefix=""):

		# The user-provided likelihood and prior transform functions.
		self.likelihood_func = likelihood_func
		self.prior_trans = prior_trans

		# The prefix with which all output files will be saved.
		self.outfile_prefix = outfile_prefix

		# The dimensionality of the parameter space
		self.ndim = ndim
		
		# The number of live points
		self.n_live = n_live

		# Set the maximum number of iterations the code will cycle through
		if maxiter is None:
			self.maxiter = 100000

		else:
			self.maxiter = maxiter

		# The number of function calls which have been performed, regardless of whether the point was accepted.
		self.n_calls = 0

		# The number of successful replaements whch have taken place.
		self.n_samples = 0

		# The fraction of the prior volume with likelihood higher than the current worst live point.
		self.X = 1.
		
		# The Bayesian evidence
		self.Z = 0.

		# A record of the value of X after each replacement.
		self.X_vals = np.zeros(self.maxiter)

		# A record of the log-likelihood value at each value of X
		self.loglike_vals = np.zeros(self.maxiter)

		# A record of the probability corresponding to each log-likelihood value.
		self.prob_vals = np.zeros(self.maxiter)

		# Draw initial live point positions at random from the parameter space.
		self.get_initial_points()



	def get_initial_points(self):
		""" Sets the initial state of the sampler by randomly distributing live points over the prior volume. """

		# Current positions of the live points in cube space (each axis runs from 0 to 1).
		self.live_cubes = np.zeros((self.n_live, self.ndim))

		# Current positions of the live points in the parameter space (prior volume).
		self.live_params = np.zeros((self.n_live, self.ndim))

		# Current log-likelihood value associated with each live point.
		self.live_loglike = np.zeros(self.n_live)

		# Set the initial values for each live point.
		for i in range(self.n_live):
			self.live_cubes[i,:] = np.random.rand(self.ndim)
			self.live_params[i,:] = self.prior_trans(np.copy(self.live_cubes[i,:]))
			self.live_loglike[i] = self.likelihood_func(self.live_params[i,:])


	def draw_new_point(self):
		""" Selects a new point from parameter space. The objective here is to draw at random from the region of the prior volume
		with likelihood greater than the current worst live point. This function is critical for the efficiency of the code. """

		maxp = np.max(self.live_cubes, axis=0)
		minp = np.min(self.live_cubes, axis=0)

		new_cube = (maxp - minp)*np.random.rand(self.ndim) + minp

		"""
		new_cube = np.random.rand(self.ndim)
		"""
 
		return new_cube


	def run(self):
		""" Run the sampler. This function uses the nested sampling procedure introduced by Skilling (2006). """

		f = open(self.outfile_prefix + "dead_points.txt", "w")
		f.write("parameter_1_value parameter_2_value weighting")
		print("\n Beginning nested sampling")
		print("-----------------------------------")

		while self.n_calls < self.maxiter:
			self.n_samples += 1

			worst_point = np.argmin(self.live_loglike)

			self.loglike_vals[self.n_samples] = self.live_loglike[worst_point]

			self.X_vals[self.n_samples] = np.exp(-(self.n_samples)/float(self.n_live))

			w_i = 0.5*(np.exp(-(self.n_samples-1.)/float(self.n_live)) - np.exp(-(self.n_samples+1.)/float(self.n_live)))

			self.Z += w_i*np.exp(self.live_loglike[worst_point])

			if self.Z == 0.:
				self.prob_vals[self.n_samples] = 0.

			else:
				self.prob_vals[self.n_samples] = w_i*np.exp(self.loglike_vals[self.n_samples])/self.Z

			f.write(str("%.5f" % self.live_params[worst_point][0]) + "\t" + str("%.5f" % self.live_params[worst_point][1]) + "\t" + str("%.5f" % self.prob_vals[self.n_samples]) + "\n")

			if self.n_samples % 50 == 0:
				print("Number of accepted samples:", self.n_samples)
				print("Number of likelihood calls:", self.n_calls)
				print("Log_e Bayesian Evidence:", np.round(np.log(self.Z), 5))
				print("-----------------------------------")

			while self.live_loglike[worst_point] >= self.likelihood_func(self.live_params[worst_point]):
				new_live_cube = self.draw_new_point()
				self.live_params[worst_point] = self.prior_trans(np.copy(new_live_cube))
				self.n_calls += 1

			self.live_cubes[worst_point,:] = np.copy(new_live_cube)
			self.live_loglike[worst_point] = self.likelihood_func(self.live_params[worst_point])

		print("Sampling is complete, best fitting parameters: ", self.live_params[np.argmax(self.live_loglike), :], "\n")

		f.close()



