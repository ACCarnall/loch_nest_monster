from __future__ import print_function, division, absolute_import

import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma

from .box import box

class nballs(object):
    """ A bounding object consisting of an N-dimensional ball 
    (hyperball) around each live point. """

    def __init__(self, points, n_to_sample=100, use_box=False,
                 box_expansion=1.):

        self.points = points
        self.use_box = use_box
        self.n_to_sample = n_to_sample

        self.n_dim = self.points.shape[1]
        self.ball_nos = np.arange(self.points.shape[0])
        self.radius = self.calculate_radius()

        if use_box:
            self.box = box(self.points, expansion=box_expansion)

        self.generate_sample(n=self.n_to_sample)

    def calculate_radius(self):
        """ Calculate the  distance to the kth nearest neighbour for 
        each live point. """

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
        distances, indices = nbrs.fit(self.points).kneighbors(self.points)

        avg_distance =  np.mean(distances[:,-1])

        radius_expansion = 2. # fudge factor 

        return radius_expansion*avg_distance

    def sample_sphere(self, n=1):
        """ Draw n random points from an N-dimensional unit sphere. """

        points_gauss = np.random.randn(n, self.n_dim)
        gauss_radii = np.sqrt(np.sum(points_gauss**2, axis=1))
        points_shell = points_gauss/np.expand_dims(gauss_radii, 1)

        radius = np.expand_dims(np.random.rand(n)**(1/self.n_dim), 1)

        return points_shell*radius

    def in_n_balls(self, samples):
        """ Calculate how many of the hyperballs the samples are in. """

        samples_in_n = np.zeros(samples.shape[0]).astype(int)

        if samples.ndim == 1:
            samples = np.expand_dims(samples, 0)

        if samples.ndim == 2:
            samples = np.expand_dims(samples, -1)  

        points = np.expand_dims(self.points.T, 0)
        distances = np.sqrt(np.sum((samples - points)**2, axis=1))/self.radius
        samples_in_n = np.zeros_like(distances).astype(int)
        samples_in_n[distances < 1.] = 1

        samples_in_n = samples_in_n.sum(axis=1)

        if self.use_box:
            samples_in_n[np.invert(self.box.in_box(samples))] = 0

        return samples_in_n

    def draw_point(self):
        """ Returns a single sample from within the hyperballs, samples
        come from a pre-computed array, gets more if it runs out. """

        sample = self.samples[self.sample_no, :]
        self.sample_no += 1

        if self.sample_no == self.no_of_samples:
            self.no_of_samples = 0
            while not self.no_of_samples:
                self.generate_sample(n=self.n_to_sample)

        return sample

    def generate_sample(self, n=1000):
        """ Draw a batch of samples uniformly from the hyperballs, this
        is quite quick, but slow enough to be worth vectorising. """

        ball_no = np.random.choice(self.ball_nos, size=n)

        samples = self.sample_sphere(n=n)
        samples *= self.radius
        samples += self.points[ball_no]

        if self.use_box:
            samples = samples[self.box.in_box(samples),:]

        in_n = self.in_n_balls(samples)
        mask = (np.random.rand(samples.shape[0]) < 1/in_n)

        if samples.ndim == 1:
            samples = np.expand_dims(samples, 0)

        self.samples = samples[mask, :]

        self.no_of_samples = self.samples.shape[0]
        self.sample_no = 0



def nballs_fill_frac(nlive, n_dim, n_repeats=100, n_samples=500):
    """ Calculate the volume fraction the hyperballs fill. """

    filling_factors = np.zeros(n_repeats)
    mean_oversamples = np.zeros(n_repeats)

    for i in range(n_repeats):
        live_points = np.random.rand(nlive, n_dim)

        samples = np.random.rand(n_samples, n_dim)

        bound = nballs(live_points)

        in_n_balls = bound.in_n_balls(samples)

        filling_factors[i] =  1 - in_n_balls[in_n_balls == 0].shape[0]/n_samples

        mean_oversamples[i] = np.mean(in_n_balls)

    """
    print("\nFilling factor:", np.mean(filling_factors),
          "std: ", np.std(filling_factors))

    print("Mean oversample:", np.mean(mean_oversamples),
          "std: ", np.std(mean_oversamples), "\n")
    """
    return np.mean(filling_factors)
