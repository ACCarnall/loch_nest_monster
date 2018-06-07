from __future__ import print_function, division, absolute_import

import numpy as np

from .nballs import nballs


def calc_nballs_filling_factor(nlive, ndim, exp_factor, repeats=25, k=1):

	filling_factors = np.zeros(repeats)
	mean_oversamples = np.zeros(repeats)

	for i in range(repeats):
		live_points = np.random.rand(nlive, ndim)

		nsamples = 1000

		samples = np.random.rand(nsamples, ndim)

		bound = nballs(live_points, exp_factor, remove=False, k=k)

		in_n_balls = bound.in_n_balls(samples)

		filling_factors[i] =  1 - in_n_balls[in_n_balls == 0].shape[0]/nsamples

		mean_oversamples[i] = np.mean(in_n_balls)

	"""
	print("\nFilling factor:", np.mean(filling_factors),
		  "std: ", np.std(filling_factors))

	print("Mean oversample:", np.mean(mean_oversamples),
		  "std: ", np.std(mean_oversamples), "\n")
	"""
	return np.mean(filling_factors)

