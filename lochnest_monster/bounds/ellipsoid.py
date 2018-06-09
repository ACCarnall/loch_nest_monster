from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.linalg import cholesky

class ellipsoid(object):

    def __init__(self, points, expansion=1.):

        self.points = points
        self.n_dim = self.points.shape[1]
        self.expansion = expansion

        self.cov = np.cov(self.points.T)
        self.centroid = np.mean(self.points, axis=0)

        self.cov_inv = np.linalg.inv(self.cov)

        # Find the most distant point from the centroid.
        points_cent = self.points - self.centroid
        tdot = np.tensordot(points_cent, self.cov_inv, 1)
        magnitudes = np.tensordot(tdot, points_cent.T, 1)
        max_dist = np.sqrt(magnitudes.max())

        self.expand_factor = max_dist*self.expansion**(1/self.n_dim)

        self.sphere_tform = cholesky(self.cov, lower=True)

    def sample_sphere(self, n=1):
        """ Draw n random points from an N-dimensional unit sphere. """

        points_gauss = np.random.randn(n, self.n_dim)
        gauss_radii = np.sqrt(np.sum(points_gauss**2, axis=1))
        points_sphere = points_gauss/np.expand_dims(gauss_radii, 1)

        radius = np.expand_dims(np.random.rand(n)**(1/self.n_dim), 1)

        return np.squeeze(points_sphere*radius)

    def draw_point(self, n=1):
        """ Draw a random point uniformly from within the ellipse. """

        sphere_samples = self.sample_sphere(n=n)

        point = np.tensordot(self.sphere_tform, sphere_samples.T, 1).T
        point *= self.expand_factor
        point += self.centroid

        return point

    def in_ellipsoid(self, point):
        """ Is the point in the ellipse? """
        point_cent = point - self.centroid
        tdot = np.tensordot(point_cent, self.cov_inv, 1)
        magnitude = np.tensordot(tdot, point_cent.T, 1)
        radial_dist = np.sqrt(magnitude)*self.expansion**(1/self.n_dim)
        return radial_dist < self.expand_factor
