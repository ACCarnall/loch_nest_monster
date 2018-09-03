from __future__ import print_function, division, absolute_import

import numpy as np


class box(object):

    def __init__(self, points, expansion=1.):

        self.points = points
        self.expansion = expansion
        self.n_dims = self.points.shape[1]

        self.lower = np.min(self.points, axis=0)
        self.upper = np.max(self.points, axis=0)

        self.centroid = (self.upper + self.lower)/2

        radius_factor = self.expansion**(1/self.n_dims)

        self.lower -= (self.centroid - self.lower)*(radius_factor-1)
        self.upper += (self.upper - self.centroid)*(radius_factor-1)

        self.lower[self.lower < 0] = 0.
        self.upper[self.upper > 1] = 1.

        self.widths = self.upper - self.lower

    def draw_point(self, n=1):
        """ Returns a single sample from within the box. """

        samples = self.widths*np.random.rand(n, self.n_dims) + self.lower

        return np.squeeze(samples)

    def in_box(self, point):
        """ Is the point in the box? """

        result = np.ones(point.shape).astype(bool)

        if result.ndim == 1:
            result = np.expand_dims(result, 0)

        for i in range(self.n_dims):
            result[:,i] = np.min([(point[:,i] > self.lower[i]),
                                  (point[:,i] < self.upper[i])], axis=0)

        return np.squeeze(result.min(axis=1))

