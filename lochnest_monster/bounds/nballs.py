from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.linalg import cholesky

import sklearn as skl
from sklearn.neighbors import NearestNeighbors


class nballs(object):

    def __init__(self, points, exp_factor=2.):

        self.points = points
        self.exp_factor = exp_factor

        self.ndim = self.points.shape[1]
        self.ball_nos = np.arange(self.points.shape[0])

        # Calculate the radii of each hyperball based on 
        self.radii = self.get_radii()
        self.radii *= self.exp_factor

        # Calculate the volumes of the hyperballs
        self.volumes = np.pi*self.radii**self.ndim
        self.volumes_norm = self.volumes/self.volumes.sum()

    def sample_sphere(self):
        """ Draw a random point from within a n-dimensional unit hyper-sphere. """

        n_gauss = np.random.randn(self.ndim)
        n_sphere = n_gauss/np.sqrt(np.sum(n_gauss**2))

        radius = np.random.rand()**(1./self.ndim)

        return n_sphere*radius

    def get_radii(self):
        """ Calculate the radii of the hyperballs (or the distance to
        the nearest neighbour for each live point). """
        nbrs = NearestNeighbors(n_neighbors=2,
                                algorithm='ball_tree').fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)

        return distances[:,1]

    def in_n_balls(self, samples):
        """ Figure out how many of the hyperballs a sample is in. """

        samples_in_n = np.zeros(samples.shape[0]).astype(int)

        if samples.ndim == 1:
            samples = np.expand_dims(samples, 0)

        if samples.ndim == 2:
            samples = np.expand_dims(samples, -1)  

        points = np.expand_dims(self.points.T, 0)
        radii = np.expand_dims(self.radii, 0)

        distances = np.sqrt(np.sum((samples - points)**2, axis=1))/radii

        samples_in_n = np.zeros_like(distances).astype(int)

        samples_in_n[distances < 1.] = 1

        return samples_in_n.sum(axis=1)

    def draw_point(self):
        """ Draw a sample uniformly from within the volume of the hyperballs. """

        while True:

            ball_no = np.random.choice(self.ball_nos, p=self.volumes_norm)

            sample = self.sample_sphere()
            sample *= self.radii[ball_no]
            sample += self.points[ball_no]

            in_n = self.in_n_balls(sample)

            if np.random.rand() < (1/in_n[0]):
                return sample


