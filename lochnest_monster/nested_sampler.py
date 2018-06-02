from __future__ import print_function, division, absolute_import

import numpy as np
import time
from scipy.misc import logsumexp

"""
Missing things:
 - Function to generate equally-weighted posterior samples.
 - Saving outputs.
 - Ability to resume from previously generated outputs.
 - Calcualtion of uncertainty on evidence value.
 - Efficient bounding options.
"""

class nested_sampler:
    """ A basic functioning nested sampling class. """

    def __init__(self, lnlike, prior_trans, ndim, n_live=400,
                 prefix="", tol_dlnz=0.01, method="box"):

        # The user-provided likelihood and prior transform functions.
        self.user_lnlike = lnlike
        self.user_prior_trans = prior_trans

        self.method = method
        self.tol_dlnz = tol_dlnz

        # The prefix with which all output files will be saved.
        self.prefix = prefix

        # The dimensionality of the parameter space.
        self.ndim = ndim

        # The number of live points.
        self.n_live = n_live

        # The number of function calls which have been performed,
        # regardless of whether the point was accepted.
        self.n_calls = 0

        # The number of successful replacements whch have been made.
        self.n_samples = 0

        # The natural ln of the Bayesian evidence.
        self.lnz = -np.inf

        # An approximate upper bound on the remaining evidence.
        self.dlnz = np.inf

        # A record of the ln-likelihood for each dead point.
        self.dead_lnlike = []

        # A record of the cube position for each dead point.
        self.dead_cubes = []

        # A record of the time taken for the last few likelihood calls.
        self.call_times = []

        # Randomly draw initial live point positions.
        self.get_initial_points()

    def prior_trans(self, input_cube):
        """ Wrapper on the user's prior transform function. """

        cube = np.copy(input_cube)

        return self.user_prior_trans(cube)

    def lnlike(self, input_param):
        """ Wrapper on the user's ln-likelihood function. """

        param = np.copy(input_param)

        return self.user_lnlike(param)

    def get_initial_points(self):
        """ Sets the initial state of the sampler by randomly
        distributing live points over the prior volume. """

        # Positions of live points in cube space (0 to 1).
        self.live_cubes = np.zeros((self.n_live, self.ndim))

        # Positions of live points in parameter space (or prior volume).
        self.live_params = np.zeros((self.n_live, self.ndim))

        # Current ln-likelihood value associated with each live point.
        self.live_lnlike = np.zeros(self.n_live)

        # Set the initial values for each live point.
        for i in range(self.n_live):
            self.live_cubes[i, :] = np.random.rand(self.ndim)
            self.live_params[i, :] = self.prior_trans(self.live_cubes[i, :])
            self.live_lnlike[i] = self.lnlike(self.live_params[i, :])

    def draw_new_point(self):
        """ Selects a new point from parameter space. The objective here
        is to draw at random from the region of the prior volume with
        likelihood greater than the current worst live point. This
        function is critical for the efficiency of the code. """

        if self.method == "box":
            maxp = np.max(self.live_cubes, axis=0)
            minp = np.min(self.live_cubes, axis=0)

            new_cube = (maxp - minp)*np.random.rand(self.ndim) + minp

        elif self.method == "uniform":
            new_cube = np.random.rand(self.ndim)

        return new_cube

    def _lnvolume(self, i):
        return -i/self.n_live

    def _lnweight(self):
        vol_low = self._lnvolume(self.n_samples+1)
        vol_high = self._lnvolume(self.n_samples-1)

        return logsumexp(a=[vol_high, vol_low], b=[0.5, -0.5])

    def run(self):
        """ Run the sampler. """

        print("\nSearching for Loch Nest Monster...\n")

        while self.dlnz > self.tol_dlnz:
            self.n_samples += 1

            # Calculate the ln weight and ln volume at this step.
            lnweight = self._lnweight()
            lnvolume = self._lnvolume(self.n_samples)

            # Index of the lowest likelihood live point.
            worst = self.live_lnlike.argmin()

            # Add the worst live point to the dead points array.
            self.dead_lnlike.append(self.live_lnlike[worst])
            self.dead_cubes.append(self.live_cubes[worst])

            # Add the lnz contribution from the worst live point to lnz.
            self.lnz = np.logaddexp(self.lnz, lnweight + self.dead_lnlike[-1])

            new_lnlike = -np.inf

            # Sample until we find a point with better lnlike.
            while self.live_lnlike[worst] >= new_lnlike:
                new_cube = self.draw_new_point()
                new_params = self.prior_trans(new_cube)
                time_start = time.time()
                new_lnlike = self.lnlike(new_params)
                self.call_times.append(time.time() - time_start)
                self.n_calls += 1

            self.live_cubes[worst, :] = np.copy(new_cube)
            self.live_params[worst, :] = np.copy(new_params)
            self.live_lnlike[worst] = np.copy(new_lnlike)

            # Estimate upper bound on evidence in the remaining volume
            self.dlnz = lnvolume + self.live_lnlike.max() - self.lnz

            # Print progress of the sampler
            if not self.n_samples % 500:
                self.print_progress()

        self.print_progress()

        print("\nSampling is complete!\n")

        self.calc_post_weights()
        
    def print_progress(self):
        print("{:<30}".format("Number of accepted samples:"),
              "{:>10}".format(self.n_samples))

        print("{:<30}".format("Number of likelihood calls:"),
              "{:>10}".format(self.n_calls))

        print("{:<30}".format("Sampling efficiency:"),
              "{:>10.4f}".format(self.n_samples/self.n_calls))

        print("{:<30}".format("Mean lnlike call time (ms):"),
              "{:>10.4f}".format(1000*np.mean(self.call_times)))

        print("{:<30}".format("Current lnZ:"),
              "{:>10.4f}".format(self.lnz))

        print("{:<30}".format("Estimated Remaining lnZ:"),
              "{:>10.4f}".format(self.dlnz))

        print("-----------------------------------------")

        self.call_times = []

    def calc_post_weights(self):
        i_vals = np.arange(self.n_samples+2)

        lnvolumes = -i_vals/self.n_live

        lnvolume_lims = np.array(zip(lnvolumes[:-2], lnvolumes[2:]))

        lnweights = logsumexp(a=lnvolume_lims, b=[0.5, -0.5], axis=1)

        lnweights += np.array(self.dead_lnlike)

        lnweights -= self.lnz

        self.weights = np.exp(lnweights)


