from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.linalg import cholesky

class ellipsoid(object):

    def __init__(self, points, exp_factor=1.):

        self.points = points
        self.exp_factor = exp_factor
        self.ndim = self.points.shape[1]
        self.cov = np.cov(self.points.T)
        self.centroid = np.mean(self.points, axis=0)
        #self.eigenvals, self.eigenvecs = np.linalg.eigh(self.cov)

        #self.eigenvals[0], self.eigenvals[1] = self.eigenvals[1], self.eigenvals[0]
        #self.eigenvecs[0], self.eigenvecs[1] = self.eigenvecs[1], self.eigenvecs[0]

        self.cov_inv = np.linalg.inv(self.cov)

        # Caclulate the expansion factor needed to enclose all points
        # and increase this by exp_factor.
        magnitudes = np.zeros(self.points.shape[0])
        points_cent = self.points - self.centroid

        for i in range(self.points.shape[0]):
            magnitudes[i] = np.dot(points_cent[i, :],
                                   np.dot(self.cov_inv, points_cent[i, :]))

        self.expansion = np.sqrt(magnitudes.max())
        self.expansion *= self.exp_factor**(1/self.ndim)

        self.sphere_tform = cholesky(self.cov, lower=True)

    def sample_sphere(self):
        """ Draw a random point from within a n-dimensional unit hypersphere. """

        n_gauss = np.random.randn(self.ndim)
        n_sphere = n_gauss/np.sqrt(np.sum(n_gauss**2))

        radius = np.random.rand()**(1./self.ndim)

        return n_sphere*radius

    def draw_point(self):
        """ Draw a random point uniformly from within the ellipse. """

        point = (np.dot(self.sphere_tform, self.sample_sphere()))
        point *= self.expansion
        point += self.centroid

        return point

    def get_2d_coords(self, dim0=0, dim1=1):
        """ Return a 2D array of the coordinates to make a 2D plot. """


        theta = np.expand_dims(np.arange(0., 2*np.pi+0.02, 0.01), 1)
        eval0 = self.eigenvals[dim0]
        eval1 = self.eigenvals[dim1]

        pos = (np.sqrt(eval0)*np.cos(theta)*self.eigenvecs[dim0]
               + np.sqrt(eval1)*np.sin(theta)*self.eigenvecs[dim1]) 

        pos *= self.expansion

        pos[:,0] += self.centroid[dim0]
        pos[:,1] += self.centroid[dim1]

        return pos

    def plot(self, dim0=0, dim1=1):
        """ Plot a 2D projection. """

        import matplotlib.pyplot as plt

        plt.figure()
        pos = self.get_2d_coords(dim0=dim0, dim1=dim1)
        plt.scatter(self.points[:,0], self.points[:,1])
        plt.plot(pos[:,0], pos[:,1], color="black", zorder=10)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.show()

