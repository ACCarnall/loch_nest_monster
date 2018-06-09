from __future__ import print_function, division, absolute_import

import numpy as np

from sklearn.neighbors import NearestNeighbors


class nballs(object):
    """ A bounding object consisting of an N-dimensional ball 
    (hyperball) around each live point. """

    def __init__(self, points, expansion=1., sample_no=10, remove=True, k=1):

        self.points = points
        self.expansion = expansion
        self.k = k

        self.n_dim = self.points.shape[1]
        self.ball_nos = np.arange(self.points.shape[0])

        # Calculate the radii of each hyperball
        self.radii = self.get_kth_neighbour_dist()
        self.radii *= self.expansion

        # Calculate the fraction of the total volume in each hyperball
        relative_volumes = self.radii**self.n_dim
        vol_orig = relative_volumes.sum()
        self.volume_fracs = relative_volumes/vol_orig

        """ Repeatedly remove hyperballs bigger than 10 percent of the
        total volume then recalculate the fractional volumes of the 
        remaining balls until no such >10 percent balls exist. """
        if remove:
            n_removed = 0
            while self.volume_fracs.max() > 0.1:
                self.radii[self.volume_fracs.argmax()] = 0.
                relative_volumes = self.radii**self.n_dim
                new_vol = relative_volumes.sum()
                self.volume_fracs = relative_volumes/new_vol
                n_removed += 1

            if n_removed:
                print("Removed", n_removed, "balls totalling",
                      "{:4.2f}".format(1 - new_vol/vol_orig), "of volume.")

        self.batch_generate_samples(n=sample_no)

    def sample_sphere(self, n=1):
        """ Draw n random points from an N-dimensional unit sphere. """

        points_gauss = np.random.randn(n, self.n_dim)
        gauss_radii = np.sqrt(np.sum(points_gauss**2, axis=1))
        points_sphere = points_gauss/np.expand_dims(gauss_radii, 1)

        radius = np.expand_dims(np.random.rand(n)**(1/self.n_dim), 1)

        return np.squeeze(points_sphere*radius)

    def get_kth_neighbour_dist(self):
        """ Calculate the  distance to the kth nearest neighbour for 
        each live point. """

        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='ball_tree')

        distances, indices = nbrs.kneighbors(self.points).fit(self.points)

        return distances[:,-1]

    def in_n_balls(self, samples):
        """ Calculate how many of the hyperballs the samples are in. """

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
        """ Returns a single sample from within the hyperballs, samples
        come from a pre-computed array, gets more if it runs out. """

        sample = self.samples[self.sample_no, :]
        self.sample_no += 1

        if self.sample_no == self.no_of_samples-1:
            self.batch_generate_samples()

        return sample

    def batch_generate_samples(self, n=1000):
        """ Draw a batch of samples uniformly from the hyperballs, this
        is quite quick, but slow enough to be worth vectorising. """

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

        test_points = np.random.rand(ntest, self.n_dim)

        in_n = self.in_n_balls(test_points)

        return 1 - in_n[in_n==0].shape[0]/ntest


def nballs_fill_frac(nlive, n_dim, expansion, k=1, repeats=25):
    """ Calculate the volume fraction the hyperballs fill. """

    filling_factors = np.zeros(repeats)
    mean_oversamples = np.zeros(repeats)

    for i in range(repeats):
        live_points = np.random.rand(nlive, n_dim)

        nsamples = 1000

        samples = np.random.rand(nsamples, n_dim)

        bound = nballs(live_points, expansion, remove=False, k=k)

        in_n_balls = bound.in_n_balls(samples)

        filling_factors[i] =  1 - in_n_balls[in_n_balls == 0].shape[0]/nsamples

        mean_oversamples[i] = np.mean(in_n_balls)

    """
    print("\nFilling factor:", np.mean(filling_factors),
          "std: ", np.std(filling_factors))

    print("Mean oversample:", np.mean(mean_oversamples),
          "std: ", np.std(mean_oversamples), "\n")
    """
    return np.mean(filling_factors)
