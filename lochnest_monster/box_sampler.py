from __future__ import print_function, division, absolute_import

import numpy as np

from .bounds import box
from .basic_sampler import basic_sampler


class box_sampler(basic_sampler):
    """ Nested sampling implementing a single box method. This involves
    drawing samples from an N-dimensional box around the region occupied 
    by the live points which is expanded by some factor.

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

    expansion : float
        The factor by which the volume of the box is expanded. The
        default value is 1.25

    """

    def __init__(self, lnlike, prior_trans, n_dim, n_live=400, stop_frac=0.99,
                 verbose=True, live_plot=False, expansion=1.25):

        basic_sampler.__init__(self, lnlike, prior_trans, n_dim, n_live=n_live,
                               stop_frac=stop_frac, verbose=verbose,
                               live_plot=live_plot)

        self.expansion = expansion

        """ Update the bound every time the expected volume decreases by
        10 percent, this is the current default, can be varied """
        self.update_interval = int(0.1*self.n_live)

        self.update_bound()

    def update_bound(self):
        """ Update the bounding object to draw points within. """

        if not self.n_samples % self.update_interval:
            self.bound = box(self.live_cubes, expansion=self.expansion) 

    def draw_new_point(self):
        """ Select a new point from the prior within the bound. """

        return self.bound.draw_point()
