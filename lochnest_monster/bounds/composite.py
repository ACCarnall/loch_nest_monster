from __future__ import print_function, division, absolute_import

import numpy as np

from sklearn.neighbors import NearestNeighbors

from .box import box
from .nboxes import nboxes, calc_nboxes_expansion, nboxes_fill_frac
from .ellipsoid import ellipsoid


class composite(object):

    def __init__(self, points, box_expansion=1.1, ell_expansion=1.1,
                 nboxes_expansion=1., k=1, make_samples=10):

        self.make_samples = make_samples
        self.points = points
        self.k = k

        self.box = box(points, expansion=box_expansion)
        self.ellipsoid = ellipsoid(points, expansion=ell_expansion)
        self.nboxes = nboxes(points, k=self.k, remove=False,
                             expansion=nboxes_expansion)

        self.batch_generate_samples(n=self.make_samples)

    def batch_generate_samples(self, n=10):
        """ Generate sample_no samples ready to be accessed. """

        samples = self.ellipsoid.draw_point(n)
        samples = samples[self.box.in_box(samples),:]
        samples = samples[self.nboxes.in_n_boxes(samples) > 0,:]

        self.samples = samples
        self.no_of_samples = self.samples.shape[0]
        self.sample_no = 0

    def draw_point(self):
        """ Returns a single sample from the  pre-computed array, gets
        more if it runs out. """

        sample = self.samples[self.sample_no, :]
        self.sample_no += 1

        while self.sample_no >= self.no_of_samples-1:
            self.batch_generate_samples(n=self.make_samples)

        return sample

    def in_composite(self, point):

        in_box = self.box.in_box(point)
        in_nboxes = (self.nboxes.in_n_boxes(point) > 0)
        in_ellipsoid = self.ellipsoid.in_ellipsoid(point)

        return np.min(np.array([in_box, in_nboxes, in_ellipsoid]), axis=1)
