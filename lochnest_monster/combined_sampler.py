from __future__ import print_function, division, absolute_import

import numpy as np

from .bounds import combined
from .basic_sampler import basic_sampler


class combined_sampler(basic_sampler):
    """ Nested sampling implementing all the bounding methods I could 
    think of!

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

    volume_fill_frac : float
        The fraction of the volume within the mass of live points which 
        the balls aim to fill. Defaults to 0.99 (99 percent).

    k : int
        The kth nearest neighbour of each point is used to construct
        the balls, default is 5.
    """

    def __init__(self, lnlike, prior_trans, n_dim, n_live=400, stop_frac=0.99,
                 verbose=True, live_plot=False, use_box=True,
                 box_expansion=1.25, ell_expansion=1.25):

        basic_sampler.__init__(self, lnlike, prior_trans, n_dim, n_live=n_live,
                               stop_frac=stop_frac, verbose=verbose,
                               live_plot=live_plot)

        self.use_box = use_box
        self.box_expansion = box_expansion
        self.ell_expansion = ell_expansion

        # Update the bound every time the volume decreases by 10 percent
        self.update_interval = int(0.1*self.n_live)

        self.update_bound()

    def update_bound(self):
        """ Update the bounding object to draw points within. """

        if not self.n_samples % self.update_interval:
            n_to_sample = int(10*self.update_interval/self.efficiency) + 1

            self.bound = combined(self.live_cubes, use_box=self.use_box,
                                  box_expansion=self.box_expansion,
                                  ell_expansion=self.ell_expansion,
                                  n_to_sample=n_to_sample) 


    def draw_new_point(self):
        """ Select a new point from the prior within the bound. """

        return self.bound.draw_point()
