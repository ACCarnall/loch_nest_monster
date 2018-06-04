from __future__ import print_function, division, absolute_import

import numpy as np


class unitcube(object):

    def __init__(self, points, exp_factor=1.):

        self.points = points
        self.exp_factor = exp_factor

        self.ndim = self.points.shape[1]

        self.lower = np.zeros(self.ndim)
        self.upper = np.zeros(self.ndim) + 1.
        self.centroid = np.zeros(self.ndim) + 0.5

    def draw_point(self):
        return np.random.rand(self.ndim)

    def get_2d_coords(self, dim0=0, dim1=1):
        """ Return a 2D array of the coordinates to make a 2D plot. """


        pos = np.zeros((5, 2))
        pos[0,:] = [self.upper[dim0], self.upper[dim1]]
        pos[1,:] = [self.lower[dim0], self.upper[dim1]]
        pos[2,:] = [self.lower[dim0], self.lower[dim1]]
        pos[3,:] = [self.upper[dim0], self.lower[dim1]]
        pos[4,:] = [self.upper[dim0], self.upper[dim1]]

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
