from __future__ import print_function, division, absolute_import

import numpy as np


class box(object):

    def __init__(self, points, expansion=1.):

        self.points = points
        self.expansion = expansion
        self.n_dim = self.points.shape[1]

        self.lower = np.min(self.points, axis=0)
        self.upper = np.max(self.points, axis=0)

        self.centroid = (self.upper + self.lower)/2

        radius_factor = self.expansion**(1/self.n_dim)

        self.lower -= (self.centroid - self.lower)*(radius_factor-1)
        self.upper += (self.upper - self.centroid)*(radius_factor-1)

        self.lower[self.lower < 0] = 0.
        self.upper[self.upper > 1] = 1.

        self.widths = self.upper - self.lower

    def draw_point(self):
        """ Returns a single sample from within the box. """
        return self.widths*np.random.rand(self.n_dim) + self.lower
