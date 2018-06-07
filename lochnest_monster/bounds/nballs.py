from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.linalg import cholesky

import sklearn as skl
from sklearn.neighbors import NearestNeighbors


class nballs(object):

    def __init__(self, points, exp_factor=1.5, n_to_gen=10, remove=True, k=1):

        self.points = points
        self.exp_factor = exp_factor
        self.k = k

        self.ndim = self.points.shape[1]
        self.ball_nos = np.arange(self.points.shape[0])

        # Calculate the radii of each hyperball based on 
        self.radii = self.get_radii()
        self.radii *= self.exp_factor

        # Calculate the volumes of the hyperballs
        relative_volumes = self.radii**self.ndim
        vol_orig = relative_volumes.sum()
        self.volume_fracs = relative_volumes/vol_orig

        if remove:
            n_removed = 0
            while self.volume_fracs.max() > 0.1:
                self.radii[self.volume_fracs.argmax()] = 0.
                relative_volumes = self.radii**self.ndim
                new_vol = relative_volumes.sum()
                self.volume_fracs = relative_volumes/new_vol
                n_removed += 1

            if n_removed:
                print("Removed", n_removed, "balls totalling",
                      "{:4.2f}".format(1 - new_vol/vol_orig), "of volume.")

        self.batch_generate_samples(n=n_to_gen)

    def sample_sphere(self, n=1):
        """ Draw a random point from an n-dimensional unit sphere. """

        n_gauss = np.random.randn(n, self.ndim)
        n_sphere = n_gauss/np.expand_dims(np.sqrt(np.sum(n_gauss**2, axis=1)), 1)

        radius = np.expand_dims(np.random.rand(n)**(1/self.ndim), 1)

        return np.squeeze(n_sphere*radius)

    def get_radii(self):
        """ Calculate the radii of the hyperballs (or the distance to
        the nearest neighbour for each live point). """
        nbrs = NearestNeighbors(n_neighbors=self.k+1,
                                algorithm='ball_tree').fit(self.points)

        distances, indices = nbrs.kneighbors(self.points)

        return distances[:,-1]

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

    def generate_single_sample(self):
        """ Draw a sample uniformly from within the hyperballs. """

        while True:

            ball_no = np.random.choice(self.ball_nos, p=self.volume_fracs)

            sample = self.sample_sphere()
            sample *= self.radii[ball_no]
            sample += self.points[ball_no]

            in_n = self.in_n_balls(sample)

            if np.random.rand() < (1/in_n[0]):
                return sample

    def draw_point(self):
        sample = self.samples[self.sample_no, :]
        self.sample_no += 1

        if self.sample_no == self.no_of_samples-1:
            self.batch_generate_samples()

        return sample

    def batch_generate_samples(self, n=1000):
        """ Draw a sample uniformly from within the hyperballs. """

        ball_no = np.random.choice(self.ball_nos, size=n, p=self.volume_fracs)

        samples = self.sample_sphere(n=n)
        samples *= np.expand_dims(self.radii[ball_no], 1)
        samples += self.points[ball_no]

        in_n = self.in_n_balls(samples)
        mask = (np.random.rand(n) < (1/in_n))

        self.samples = samples[mask, :]
        self.no_of_samples = self.samples.shape[0]
        self.sample_no = 0

    def calc_volume(self, ntest=1000):
        """ Find the fraction of the unit cube the hyperballs fill. """

        test_points = np.random.rand(ntest, self.ndim)

        in_n = self.in_n_balls(test_points)

        return 1 - in_n[in_n==0].shape[0]/ntest



