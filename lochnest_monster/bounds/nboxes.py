from __future__ import print_function, division, absolute_import

import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class nboxes(object):
    """ A bounding object consisting of an N-dimensional ball 
    (hyperball) around each live point. """

    def __init__(self, points, expansion=1., sample_no=10, remove=True, k=1):

        self.points = points
        self.expansion = expansion
        self.k = k

        self.n_dim = self.points.shape[1]
        self.box_nos = np.arange(self.points.shape[0])

        # Calculate the half-width of each cube along each dimension
        self.box_dims = self.get_kth_neighbour_dims()
        self.box_dims *= self.expansion

        # Calculate the fraction of the total volume in each hyperball
        relative_volumes = np.prod(2*self.box_dims, axis=1)

        vol_orig = relative_volumes.sum()
        self.volume_fracs = relative_volumes/vol_orig

        """ Repeatedly remove hypercubes bigger than 10 percent of the
        total volume then recalculate the fractional volumes of the 
        remaining balls until no such >10 percent balls exist. """
        if remove:
            n_removed = 0
            while self.volume_fracs.max() > 0.1:
                self.box_dims[self.volume_fracs.argmax(),:] = 0.
                relative_volumes = np.prod(2*self.box_dims, axis=1)
                new_vol = relative_volumes.sum()
                self.volume_fracs = relative_volumes/new_vol
                n_removed += 1

            if n_removed:
                print("Removed", n_removed, "boxes totalling",
                      "{:4.2f}".format(1 - new_vol/vol_orig), "of volume.")
            
        self.batch_generate_samples(n=sample_no)

    def get_kth_neighbour_dims(self):
        """ Calculate the  distance to the kth nearest neighbour for 
        each live point. """

        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='ball_tree')

        distances, indices = nbrs.fit(self.points).kneighbors(self.points)

        dims = np.abs(self.points - self.points[indices[:,-1], :])/2

        return dims

    def in_n_boxes(self, samples):
        """ Calculate how many of the hyperballs the samples are in. """

        if samples.ndim == 1:
            samples = np.expand_dims(samples, 0)

        if samples.ndim == 2:
            samples = np.expand_dims(samples, -1)  

        points = np.expand_dims(self.points.T, 0)

        box_dims = np.expand_dims(self.box_dims.T, 0)

        dist_ratios = np.abs(samples - points)/box_dims

        max_axis_dist = dist_ratios.max(axis=1)

        samples_in_n = (max_axis_dist < 1.).sum(axis=1)

        return samples_in_n

    def generate_single_sample(self):
        """ Draw a sample uniformly from within the hyperballs. """

        while True:

            ball_no = np.random.choice(self.box_nos, p=self.volume_fracs)

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

        cube_no = np.random.choice(self.box_nos, size=n, p=self.volume_fracs)

        samples = np.random.rand(n, self.n_dim)
        samples *= 2*self.box_dims[cube_no,:]
        samples += self.points[cube_no,:] - self.box_dims[cube_no,:]

        in_n = self.in_n_boxes(samples)
        mask = (np.random.rand(n) < (1/in_n))

        self.samples = samples[mask, :]
        self.no_of_samples = self.samples.shape[0]
        self.sample_no = 0

    def calc_volume(self, ntest=1000):
        """ Find the fraction of the unit cube the hyperballs fill. """

        test_points = np.random.rand(ntest, self.n_dim)

        in_n = self.in_n_balls(test_points)

        return 1 - in_n[in_n==0].shape[0]/ntest

    def plot(self):

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        self.fig = plt.figure(figsize=(9, 8))

        gs = mpl.gridspec.GridSpec(self.n_dim-1, self.n_dim-1,
                                   wspace=0., hspace=0.)

        self.axes = []

        for i in range(self.n_dim-1):
            for j in range(1, self.n_dim):
                if i <=  j - 1:
                    print(i, j)
                    ax = plt.subplot(gs[j-1,i])
                    ax.set_xlim(0., 1.)
                    ax.set_ylim(0., 1.)
                    #ax.set_xticks([])
                    #ax.set_yticks([])
                    ax.scatter(self.points[:,i], self.points[:,j], zorder=9, color="blue", s=3)
                    #ax.scatter(self.samples[:,i], self.samples[:,j], zorder=10, color="green", s=2)

                    for k in range(self.points.shape[0]):

                        ax.fill_between([self.points[k,i] - self.box_dims[k,i], self.points[k,i] + self.box_dims[k,i]],
                                         [self.points[k,j] - self.box_dims[k,j], self.points[k,j] - self.box_dims[k,j]],
                                         [self.points[k,j] + self.box_dims[k,j], self.points[k,j] + self.box_dims[k,j]],
                                         alpha=0.5, color="red")

                    self.axes.append(ax)
        
        plt.show()


def nboxes_fill_frac(nlive, n_dim, expansion, k=5, repeats=200, plot=False):
    """ Calculate the volume fraction the hyperballs fill. """

    filling_factors = np.zeros(repeats)
    mean_oversamples = np.zeros(repeats)

    for i in range(repeats):
        live_points = np.random.rand(nlive, n_dim)
        bound = nboxes(live_points, expansion, k=k, remove=False)

        nsamples = 500
        samples = np.random.rand(nsamples, n_dim)
        in_n_boxes = bound.in_n_boxes(samples)

        filling_factors[i] =  1 - in_n_boxes[in_n_boxes == 0].shape[0]/nsamples

        mean_oversamples[i] = np.mean(in_n_boxes)

    if plot:

        nsamples = 5000
        samples = np.random.rand(nsamples, n_dim)
        bound = nboxes(live_points, expansion, k=k)
        in_n_boxes = bound.in_n_boxes(samples)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(in_n_boxes, bins=100)
        plt.show()

    return np.mean(filling_factors)

def objective(expansion, nlive, n_dim, k=5, obj_fill_frac=0.99):
    fill_frac = nboxes_fill_frac(nlive, n_dim, expansion[0], k=k)
    print(expansion, fill_frac)
    return np.abs(fill_frac - obj_fill_frac)


def calc_nboxes_expansion(fill_frac, nlive, n_dim, k=5):
    """ Calculate the volume fraction the hyperballs fill. """

    minim = minimize(objective, 1*k, args=(nlive, n_dim, k, fill_frac), tol=0.01, method="Nelder-Mead")
    return minim["x"]
    