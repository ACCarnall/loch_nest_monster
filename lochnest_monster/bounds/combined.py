from __future__ import print_function, division, absolute_import

import numpy as np

from sklearn.neighbors import NearestNeighbors

from .box import box
from .nballs import nballs
from .ellipsoid import ellipsoid


class combined(object):

    def __init__(self, points, box_expansion=1.1, ell_expansion=1.1,
                 n_to_sample=10, use_box=True):


        self.n_to_sample = n_to_sample

        self.ellipsoid = ellipsoid(points, expansion=ell_expansion)
        self.nballs = nballs(points, use_box=use_box, box_expansion=box_expansion)

        self.generate_sample(n=self.n_to_sample)

    def generate_sample(self, n=1000):
        """ Generate sample_no samples ready to be accessed. """

        samples = self.ellipsoid.draw_point(n)
        samples = samples[self.nballs.in_n_balls(samples) > 0,:]

        self.samples = samples
        self.no_of_samples = self.samples.shape[0]
        self.sample_no = 0

    def draw_point(self):
        """ Returns a single sample from the  pre-computed array, gets
        more if it runs out. """

        sample = self.samples[self.sample_no, :]
        self.sample_no += 1

        if self.sample_no == self.no_of_samples:
            self.no_of_samples = 0
            while not self.no_of_samples:
                self.generate_sample(n=self.n_to_sample)

        return sample

    def in_composite(self, point):

        in_nballs = (self.nballs.in_n_balls(point) > 0)
        in_ellipsoid = self.ellipsoid.in_ellipsoid(point)

        return np.min(np.array([in_nboxes, in_ellipsoid]), axis=1)
