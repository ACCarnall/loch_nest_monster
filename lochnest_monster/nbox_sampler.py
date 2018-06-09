from __future__ import print_function, division, absolute_import

import numpy as np

from .bounds import nboxes, calc_nboxes_expansion, nboxes_fill_frac
from .basic_sampler import basic_sampler


class nbox_sampler(basic_sampler):
    """ Nested sampling implementing the nballs boundary method. This 
    uses a nearest-neighbours algorithm to draw spheres around each live
    point reaching some fraction of the way to its kth nearest neighbour
    then samples from within those spheres.

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
                 verbose=True, live_plot=False, volume_fill_frac=0.99, k=5):

        basic_sampler.__init__(self, lnlike, prior_trans, n_dim, n_live=n_live,
                               stop_frac=stop_frac, verbose=verbose,
                               live_plot=live_plot)

        self.k = k  # The kth nearest neighbour is used for nballs
        self.volume_fill_frac = volume_fill_frac  # Target filling fraction

        """ Update the bound every time the expected volume decreases by
        10 percent, this is the current default, can be varied """
        self.update_interval = int(0.1*self.n_live)

        """ Calculate the necessary expansion factor for the balls to
        meet the desired volume_fill_frac, this bit of code sucks """
        self.expansion = calc_nboxes_expansion()

        self.update_bound()

    def update_bound(self):
        """ Update the bounding object to draw points within. """

        if not self.n_samples % self.update_interval:
            sample_no = int(10*self.update_interval/self.efficiency)

            self.bound = nboxes(self.live_cubes, k=self.k,
                                expansion=self.expansion, sample_no=sample_no) 

    def draw_new_point(self):
        """ Select a new point from the prior within the bound. """

        while True:
            new_cube = self.bound.draw_point()

            if new_cube.max() < 1 and new_cube.min() > 0:
                break

        return new_cube
