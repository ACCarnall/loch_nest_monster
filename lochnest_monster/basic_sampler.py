from __future__ import print_function, division, absolute_import

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.misc import logsumexp


class basic_sampler(object):
    """ Implement a basic nested sampling process by sampling randomly
    from the unit cube. This is very slow, it's better to use a bound!
    Also includes functions for printing progress and showing a live-
    updating plot of the sampling as it happens.

    Parameters
    ----------

    lnlike : function
        A function which takes an array of parameter values and returns
        the natural log of the likelihood at that point.

    prior_trans : function
        A function which transforms a sample from the unit cube to the
        prior volume you wish to sample.

    n_dim : int
        The number of free parameters you wish to fit.

    n_live : int
        The number of live points you wish to use.

    stop_frac : float
        The fraction of the evidence you wish to integrate up to. This
        defaults to 0.9.

    verbose: bool
        Print progress updates as sampling takes place.

    live_plot : bool
        Show a live-updating plot of the live points during sampling.

    """

    def __init__(self, lnlike, prior_trans, n_dim, n_live=400,
                 stop_frac=0.99, verbose=True, live_plot=False):

        self.user_lnlike = lnlike
        self.user_prior_trans = prior_trans
        self.n_dim = n_dim
        self.n_live = n_live
        self.stop_lnz_remain = np.log(1/stop_frac)
        self.verbose = verbose
        self.live_plot = live_plot

        self.n_calls = 0  # number of likelihood calls made
        self.n_samples = 0  # no of successful replacements made
        self.efficiency = 1. # no of replacements/no of calls
        self.lnz = -np.inf  # the natural logarithm of the evidence.
        self.lnz_remain = np.inf  # approx. upper bound on remaining lnZ

        self.dead_lnlike = []
        self.dead_cubes = []
        self.dead_params = []
        self.lnweights = []
        self.call_times = []
        self.proposed = []  # All proposed points (cleared periodically)

        self.get_initial_points() # randomly draw initial live points

        if self.live_plot:
            self.setup_live_plot()

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
        self.live_cubes = np.zeros((self.n_live, self.n_dim))

        # Positions of live points in parameter space (or prior volume).
        self.live_params = np.zeros((self.n_live, self.n_dim))

        # Current ln-likelihood value associated with each live point.
        self.live_lnlike = np.zeros(self.n_live)

        # Set the initial values for each live point.
        for i in range(self.n_live):
            self.live_cubes[i, :] = np.random.rand(self.n_dim)
            self.live_params[i, :] = self.prior_trans(self.live_cubes[i, :])
            self.live_lnlike[i] = self.lnlike(self.live_params[i, :])

    def _lnvolume(self, i):
        """ Get the expected remaining lnvolume after i samples. """
        return -i/self.n_live

    def _lnweight(self):
        """ Get the volume weighting applied to the ith lnlike. """

        vol_low = self._lnvolume(self.n_samples+1)
        vol_high = self._lnvolume(self.n_samples-1)

        return logsumexp(a=[vol_high, vol_low], b=[0.5, -0.5])

    def draw_new_point(self):
        """ Return a random point from the unit cube. """
        return np.random.rand(self.n_dim)

    def update_bound(self):
        """ The basic sampler doesn't have a bound. """
        return

    def run(self):
        """ Run the sampler. """

        print("\nSearching for Loch Nest Monster...\n")

        while self.lnz_remain > self.stop_lnz_remain:

            #print(np.logaddexp(self.lnz_remain, self.lnz) - self.lnz)
            #print(-self.stop_lnz_remain)

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
            extra_lnz = lnvolume + self.live_lnlike.max()
            self.lnz_remain = np.logaddexp(extra_lnz, self.lnz) - self.lnz

            self.efficiency = self.n_samples/self.n_calls

            # Print progress of the sampler
            if self.verbose and not self.n_samples % 50:
                self.print_progress()

            if self.live_plot and not self.n_samples % 50:
                self.update_live_plot()

            if not self.n_samples % 50:
                self.proposed = []

            self.update_bound()

        self.print_progress()

        if self.live_plot:
            plt.close()
            plt.ioff()

        print("\nSampling is complete!\n")

        self._get_results()

    def _get_results(self):
        """ Populate the results dictionary once sampling is done. """

        self.results = {}

        self.results["lnz"] = self.lnz
        self.results["ncalls"] = self.n_calls
        self.results["nsamples"] = self.n_samples
        self.results["efficiency"] = self.n_samples/self.n_calls
        self.results["lnweights"] = np.array(self.lnweights) - self.lnz
        self.results["weights"] = np.exp(self.results["lnweights"])
        self.results["samples"] = np.zeros((len(self.dead_params), self.n_dim))
        self.results["lnlike"] = self.dead_lnlike

        for i in range(len(self.dead_cubes)):
            self.results["samples"][i, :] = self.dead_params[i]

        choices = np.random.choice(np.arange(self.n_samples),
                                   self.n_samples,
                                   p=self.results["weights"])

        self.results["samples_eq"] = self.results["samples"][choices, :]

    def calc_lnz_error(self, n_draws=100):
        for i in range(n_draws):
            draws = np.random.rand(self.n_live, self.n_samples)
            draws_max = draws.max(axis=0)

            volumes = np.zeros(self.n_samples+1)
            volumes[0] = 1.

            for i in range(self.n_samples):
                volumes[i+1] = volumes[i]*draws_max[i]

            vol_elements = (volumes[:-1] - volumes[1:])

            lnz_sample = logsumexp(np.logaddexp(np.log(vol_elements), self.results["lnlike"]))

            print(lnz_sample)

    def print_progress(self):
        """ Print the current progress of the sampler. """

        print("{:<30}".format("Number of accepted samples:"),
              "{:>10}".format(self.n_samples))

        print("{:<30}".format("Number of likelihood calls:"),
              "{:>10}".format(self.n_calls))

        print("{:<30}".format("Run sampling efficiency:"),
              "{:>10.4f}".format(self.efficiency))

        print("{:<30}".format("Mean lnlike call time (ms):"),
              "{:>10.4f}".format(1000*np.mean(self.call_times)))

        print("{:<30}".format("Current lnZ:"),
              "{:>10.4f}".format(self.lnz))

        print("{:<30}".format("Estimated remaining lnZ:"),
              "{:>10.4f}".format(self.lnz_remain))

        print("{:<30}".format("Goal for remaining lnZ:"),
              "{:>10.4f}".format(self.stop_lnz_remain))

        print("{:<30}".format("Remaining lnvolume:"),
              "{:>10.4f}".format(self._lnvolume(self.n_samples)))

        print("-----------------------------------------")

        self.call_times = []

    def setup_live_plot(self):
        """ Set up live plotting of the sampler's progress. """

        plt.ion()
        self.fig = plt.figure(figsize=(9, 8))

        gs = mpl.gridspec.GridSpec(self.n_dim-1, self.n_dim-1,
                                   wspace=0., hspace=0.)

        self.axes = []
        self.plot_live = []
        self.plot_prop = []

        for i in range(self.n_dim-1):
            for j in range(1, self.n_dim):
                if i <=  j - 1:
                    ax = plt.subplot(gs[j-1,i])
                    ax.set_xlim(0., 1.)
                    ax.set_ylim(0., 1.)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    pl_live = ax.scatter(self.live_cubes[:,i],
                                         self.live_cubes[:,j],
                                         s=2, color="red", zorder=9)

                    pl_prop = ax.scatter(0., 0., s=3, color="blue", zorder=8)

                    self.axes.append(ax)
                    self.plot_live.append(pl_live)
                    self.plot_prop.append(pl_prop)

        self.fig.canvas.draw()
        plt.pause(0.01)
        plt.show(block=False)

    def update_live_plot(self):
        """ Update the live plot with current progress of sampler. """

        prop_arr = np.zeros((len(self.proposed), self.n_dim))
        for i in range(len(self.proposed)):
            prop_arr[i,:] = self.proposed[i]

        try:
            n = 0
            for i in range(self.n_dim):
                for j in range(self.n_dim):
                    if i < j:
                        live_cols = np.c_[self.live_cubes[:,i],
                                          self.live_cubes[:,j]]

                        prop_cols = np.c_[prop_arr[:,i],
                                          prop_arr[:,j]]

                        self.plot_live[n].set_offsets(live_cols)
                        self.plot_prop[n].set_offsets(prop_cols)
                        n += 1

            self.fig.canvas.draw()
            plt.pause(0.01)

        except KeyboardInterrupt:
            sys.exit("killed.")
