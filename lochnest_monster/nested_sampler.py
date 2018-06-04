from __future__ import print_function, division, absolute_import

import numpy as np
import time
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import corner
import sys

from . import bounds

"""
Missing things:
 - Function to generate equally-weighted posterior samples.
 - Saving outputs.
 - Ability to resume from previously generated outputs.
 - Calcualtion of uncertainty on evidence value.
 - Efficient bounding options.
"""

class nested_sampler(object):
    """ A basic functioning nested sampling class. """

    def __init__(self, lnlike, prior_trans, ndim, n_live=400,
                 prefix="", stop_frac=0.99, bound_type="unitcube",
                 visualise=False, verbose=True, exp_factor=1.25):

        # The user-provided likelihood and prior transform functions.
        self.user_lnlike = lnlike
        self.user_prior_trans = prior_trans

        self.bound_type = bound_type
        self.stop_frac = stop_frac
        self.verbose = verbose
        self.visualise = visualise
        self.exp_factor = exp_factor

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

        self.efficiency = 1.

        # The natural logarithm of the Bayesian evidence.
        self.lnz = -np.inf

        # An approximate upper bound on the remaining lnz.
        self.z_frac = 0.

        self.dead_lnlike = []
        self.dead_cubes = []
        self.dead_params = []
        self.lnweights = []

        self.call_times = []

        # Randomly draw initial live point positions.
        self.get_initial_points()

        self.bound = bounds.unitcube(self.live_cubes)

        self.proposed = []

        if self.visualise:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0., 1.)
            self.ax.set_ylim(0., 1.)
            self.plot_live = self.ax.scatter(self.live_cubes[:,0],
                                             self.live_cubes[:,1],
                                             s=2, color="red", zorder=9)

            self.plot_prop = self.ax.scatter(0., 0., s=3,
                                            color="blue", zorder=8)

            self.bound_plot = self.ax.plot(0., 0., color="black", zorder=10)[0]

            self.fig.canvas.draw()
            plt.pause(0.0001)
            plt.show(block=False)

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

    def update_bound(self):
        """ Update the bounding object to draw points within. """

        if not self.n_samples % 100:
            self.bound = getattr(bounds,
                                 self.bound_type)(self.live_cubes,
                                                  exp_factor=self.exp_factor)

    def draw_new_point(self):
        """ Selects a new point from the prior within the bound. """

        while True:
            new_cube = self.bound.draw_point()

            if new_cube.max() < 1 and new_cube.min() > 0:
                break

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

        while self.z_frac < self.stop_frac:
            self.n_samples += 1

            # Calculate the ln weight and ln volume at this step.
            lnweight = self._lnweight()
            lnvolume = self._lnvolume(self.n_samples)

            # Index of the lowest likelihood live point.
            worst = self.live_lnlike.argmin()

            # Add the worst live point to the dead points array.
            self.dead_lnlike.append(np.copy(self.live_lnlike[worst]))
            self.dead_cubes.append(np.copy(self.live_cubes[worst]))
            self.dead_params.append(np.copy(self.live_params[worst]))

            # Add the lnz contribution from the worst live point to lnz.
            self.lnz = np.logaddexp(self.lnz, lnweight + self.dead_lnlike[-1])

            self.lnweights.append(lnweight + self.dead_lnlike[-1])

            new_lnlike = -np.inf

            # Sample until we find a point with better lnlike.
            while self.live_lnlike[worst] >= new_lnlike:
                new_cube = self.draw_new_point()
                new_params = self.prior_trans(new_cube)

                time_start = time.time()
                new_lnlike = self.lnlike(new_params)
                self.call_times.append(time.time() - time_start)

                self.n_calls += 1

                self.proposed.append(np.copy(new_cube))

            self.live_cubes[worst, :] = np.copy(new_cube)
            self.live_params[worst, :] = np.copy(new_params)
            self.live_lnlike[worst] = np.copy(new_lnlike)

            # Estimate upper bound on evidence in the remaining volume
            max_wt_remain = lnvolume + self.live_lnlike.max() - self.lnz

            self.z_frac = np.max([0., 1. - np.exp(max_wt_remain)])

            self.efficiency = self.n_samples/self.n_calls

            # Print progress of the sampler
            if self.verbose and not self.n_samples % 100:
                self.print_progress()
                self.proposed = []

            self.update_bound()

        self.print_progress()

        if self.visualise:
            plt.close()
            plt.ioff()

        print("\nSampling is complete!\n")

        self._get_results()

    def _get_results(self):
        self.results = {}

        self.results["lnweights"] = np.array(self.lnweights) - self.lnz
        self.results["weights"] = np.exp(self.results["lnweights"])
        self.results["samples"] = np.zeros((len(self.dead_params), self.ndim))

        for i in range(len(self.dead_cubes)):
            self.results["samples"][i, :] = self.dead_params[i]

        choices = np.random.choice(np.arange(self.n_samples),
                                   self.n_samples,
                                   p=self.results["weights"])

        self.results["samples_eq"] = self.results["samples"][choices, :]

    def print_progress(self):
        print("{:<30}".format("Number of accepted samples:"),
              "{:>10}".format(self.n_samples))

        print("{:<30}".format("Number of likelihood calls:"),
              "{:>10}".format(self.n_calls))

        print("{:<30}".format("Sampling efficiency:"),
              "{:>10.4f}".format(self.efficiency))

        print("{:<30}".format("Mean lnlike call time (ms):"),
              "{:>10.4f}".format(1000*np.mean(self.call_times)))

        print("{:<30}".format("Current lnZ:"),
              "{:>10.4f}".format(self.lnz))

        print("{:<30}".format("Fraction of total Z"),
              "{:>10.4f}".format(self.z_frac))

        print("-----------------------------------------")

        self.call_times = []

        if self.visualise:
            self.progress_plotter()

    def progress_plotter(self):
        try:
            self.plot_live.set_offsets(np.c_[self.live_cubes[:,0],
                                             self.live_cubes[:,1]])

            prop_arr = np.zeros((len(self.proposed), self.ndim))
            for i in range(len(self.proposed)):
                prop_arr[i,:] = self.proposed[i]


            self.plot_prop.set_offsets(prop_arr)


            pos = self.bound.get_2d_coords()

            self.bound_plot.set_xdata(pos[:,0])
            self.bound_plot.set_ydata(pos[:,1])

            self.fig.canvas.draw()
            plt.pause(0.0001)
            #raw_input()

        except KeyboardInterrupt:
            sys.exit("killed.")        
