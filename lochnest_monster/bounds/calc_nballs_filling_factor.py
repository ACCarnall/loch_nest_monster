from __future__ import print_function, division, absolute_import

import numpy as np

from .nballs import nballs


def calc_nballs_filling_factor(nlive, ndim, exp_factor):
	live_points = np.random.rand(nlive, ndim)

	nsamples = 10000

	samples = np.random.rand(nsamples, ndim)

	bound = nballs(live_points, exp_factor)

	in_n_balls = bound.in_n_balls(samples)

	filling_factor =  1 - in_n_balls[in_n_balls == 0].shape[0]/nsamples

	mean_oversample = np.mean(in_n_balls)

	print("Filling factor:", filling_factor)
	print("Mean oversample:", mean_oversample)


