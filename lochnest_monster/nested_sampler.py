from __future__ import print_function, division, absolute_import

import numpy as np
import time
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import matplotlib as mpl
import corner
import sys

from . import bounds

"""
Missing things:
 - Saving outputs.
 - Ability to resume from previously generated outputs.
 - Calcualtion of uncertainty on evidence value/posterior.
 - multi-ellipsoid sampling.
"""

class nested_sampler(object):
    """ A basic functioning nested sampling class. """

    def __init__(self, lnlike, prior_trans, ndim, n_live=200,
                 prefix="", stop_frac=0.9, bound_type="nballs",
                 visualise=True, verbose=True, exp_factor=1.25,
                 fill_factor=0.9, k=4):

        self.user_lnlike = lnlike
        self.user_prior_trans = prior_trans
        self.bound_type = bound_type
        self.stop_frac = stop_frac
        self.verbose = verbose
        self.visualise = visualise
        self.exp_factor = exp_factor
        self.fill_factor = fill_factor
        self.prefix = prefix
        self.ndim = ndim
        self.n_live = n_live
        self.update_interval = int(0.1*self.n_live)
        self.k = k

        self.n_calls = 0  # number of likelihood calls made
        self.n_samples = 0  # no of successful replacements made
        self.efficiency = 1. # no of replacements/no of calls
        self.lnz = -np.inf  # the natural logarithm of the evidence.
        self.z_frac = 0.  # approximate upper bound on the remaining lnz

        self.dead_lnlike = []
        self.dead_cubes = []
        self.dead_params = []
        self.lnweights = []
        self.call_times = []

        self.get_initial_points()  # randomly draw initial live points

        # start with unit cube bound
        n_to_gen = int(10*self.update_interval/self.efficiency)
        self.bound = bounds.nballs(self.live_cubes, n_to_gen=n_to_gen, k=self.k)  

        self.proposed = []

        if self.bound_type == "nballs":
            self.exp_factor = 0.2
            ff = 0.

            while ff < self.fill_factor:
                ff = bounds.calc_nballs_filling_factor(self.n_live, self.ndim,
                                                       self.exp_factor, k=self.k)
                self.exp_factor += 0.05
            print("Using expansion factor of", self.exp_factor)

        if self.visualise:
            self.set_up_live_plot()

    def set_up_live_plot(self):
        """ Set up live plotting of the sampler's progress. """

        plt.ion()
        self.fig = plt.figure(figsize=(9, 8))

        gs = mpl.gridspec.GridSpec(self.ndim, self.ndim, wspace=0., hspace=0.)

        self.axes = []
        self.plot_l = []
        self.plot_p = []

        for i in range(self.ndim):
            for j in range(self.ndim):
                if i < j:
                    self.axes.append(plt.subplot(gs[j,i]))
                    self.axes[-1].set_xlim(0., 1.)
                    self.axes[-1].set_ylim(0., 1.)
                    self.axes[-1].set_xticks([])
                    self.axes[-1].set_yticks([])
                    self.plot_l.append(self.axes[-1].scatter(self.live_cubes[:,i],
                                                             self.live_cubes[:,j],
                                                             s=2, color="red",
                                                             zorder=9))

                    self.plot_p.append(self.axes[-1].scatter(0., 0., s=3,
                                                            color="blue",
                                                            zorder=8))
        
        #if self.ndim == 2:
        #    self.bound_plot = self.axes[0].plot(0., 0., color="black", zorder=10)[0]
        
        self.fig.canvas.draw()
        plt.pause(1.)
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

        if not self.n_samples % self.update_interval:
            n_to_gen = int(10*self.update_interval/self.efficiency)
            self.bound = bounds.nballs(self.live_cubes, n_to_gen=n_to_gen,
                                       k=self.k) 

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
            if self.verbose and not self.n_samples % self.update_interval:
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
        """ Print the current progress of the sampler. """

        #bound_vol = self.bound.calc_volume()

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

        print("{:<30}".format("Ln-volume remaining"),
              "{:>10.4f}".format(self._lnvolume(self.n_samples)))

        #print("{:<30}".format("Prior volume in bound"),
        #      "{:>10.4f}".format(bound_vol))

        print("-----------------------------------------")

        self.call_times = []

        if self.visualise:
            self.progress_plotter()

    def progress_plotter(self, dim0=0, dim1=1):

        prop_arr = np.zeros((len(self.proposed), self.ndim))
        for i in range(len(self.proposed)):
            prop_arr[i,:] = self.proposed[i]

        try:
            n = 0
            for i in range(self.ndim):
                for j in range(self.ndim):
                    if i < j:
                        self.plot_l[n].set_offsets(np.c_[self.live_cubes[:,i],
                                                         self.live_cubes[:,j]])
                        self.plot_p[n].set_offsets(np.c_[prop_arr[:,i],
                                                         prop_arr[:,j]])
                        n += 1
            """
            if self.ndim == 2:
                pos = self.bound.get_2d_coords()
                self.bound_plot.set_xdata(pos[:,0])
                self.bound_plot.set_ydata(pos[:,1])
            """
            self.fig.canvas.draw()
            plt.pause(1.)
            #raw_input()
            
        except KeyboardInterrupt:
            sys.exit("killed.")        
