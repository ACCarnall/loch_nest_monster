from __future__ import print_function, division, absolute_import

import numpy as np

"""
Missing things:
 - Function to generate equally-weighted posterior samples.
 - Saving outputs.
 - Ability to resume from previously generated outputs.
 - Calcualtion of uncertainty on evidence value.
"""


class nested_sampler:
    """ A basic functioning nested sampling class. """

    def __init__(self, loglike, prior_trans, ndim, n_live=400,
                 maxiter=None, prefix="", tol_dlogz=0.01):

        # The user-provided likelihood and prior transform functions.
        self.user_loglike = loglike
        self.user_prior_trans = prior_trans

        self.tol_dlogz = tol_dlogz

        # The prefix with which all output files will be saved.
        self.prefix = prefix

        # The dimensionality of the parameter space.
        self.ndim = ndim

        # The number of live points.
        self.n_live = n_live

        # Set the maximum number of iterations the code will run to.
        if maxiter is None:
            self.maxiter = 100000

        else:
            self.maxiter = maxiter

        # The number of function calls which have been performed,
        # regardless of whether the point was accepted.
        self.n_calls = 0

        # The number of successful replaements whch have taken place.
        self.n_samples = 0

        # The natural log of the Bayesian evidence.
        self.lnz = -np.inf

        # A record of the log-likelihood value at each value of X.
        self.dead_loglike = np.zeros(self.maxiter)
        self.post_wts = np.zeros(self.maxiter)

        # Randomly draw initial live point positions.
        self.get_initial_points()

    def prior_trans(self, input_cube):
        """ Wrapper on the user's prior transform function. """
        cube = np.copy(input_cube)

        return self.user_prior_trans(cube)

    def loglike(self, input_param):
        """ Wrapper on the user's log-likelihood function. """

        param = np.copy(input_param)

        return self.user_loglike(param)

    def get_initial_points(self):
        """ Sets the initial state of the sampler by randomly
        distributing live points over the prior volume. """

        # Positions of live points in cube space (0 to 1).
        self.live_cubes = np.zeros((self.n_live, self.ndim))

        # Positions of live points in parameter space (or prior volume).
        self.live_params = np.zeros((self.n_live, self.ndim))

        # Current log-likelihood value associated with each live point.
        self.live_loglike = np.zeros(self.n_live)

        # Set the initial values for each live point.
        for i in range(self.n_live):
            self.live_cubes[i, :] = np.random.rand(self.ndim)
            self.live_params[i, :] = self.prior_trans(self.live_cubes[i, :])
            self.live_loglike[i] = self.loglike(self.live_params[i, :])

    def draw_new_point(self, mode="box"):
        """ Selects a new point from parameter space. The objective here
        is to draw at random from the region of the prior volume with
        likelihood greater than the current worst live point. This
        function is critical for the efficiency of the code. """

        if mode == "box":
            maxp = np.max(self.live_cubes, axis=0)
            minp = np.min(self.live_cubes, axis=0)

            new_cube = (maxp - minp)*np.random.rand(self.ndim) + minp

        elif mode == "uniform":
            new_cube = np.random.rand(self.ndim)

        return new_cube

    def _logvol_i(self, i):
        return -i/float(self.n_live)

    def _get_random_t(self, i):
        return np.random.rand(self.ndim).max()

    def _logwt_i(self, i):
        vol_iplus1 = np.exp(self._logvol_i(self.n_samples+1))
        vol_isub1 = np.exp(self._logvol_i(self.n_samples-1))

        return np.log(vol_isub1 - vol_iplus1) + np.log(0.5)

    def run(self):
        """ Run the sampler. This function uses the nested samplin
        procedure introduced by Skilling (2006). """
        """
        f = open(self.prefix + "dead_points.txt", "w")
        f.write("parameter_1_value parameter_2_value relative_logwt")
        print("\n Beginning nested sampling")
        print("-----------------------------------")
        """
        dlogz = np.inf

        while dlogz > self.tol_dlogz:
            self.n_samples += 1

            # Calculate the log weight and log volume at this step.
            logwt = self._logwt_i(self.n_samples)
            logvol = self._logvol_i(self.n_samples)

            # Index of the lowest likelihood live point.
            worst = self.live_loglike.argmin()

            # Add the worst live point to the dead points array.
            self.dead_loglike[self.n_samples] = self.live_loglike[worst]
            self.post_wts[self.n_samples] = logwt + self.live_loglike[worst]

            # Add the lnz contribution from the worst live point to lnz.
            self.lnz = np.logaddexp(self.lnz, self.post_wts[self.n_samples])

            # f.write(str("%.5f" % self.live_params[worst][0]) + "\t"
            #        + str("%.5f" % self.live_params[worst][1]) + "\t"
            #        + str("%.5f" % (logwt + self.live_loglike[worst])) + "\n")

            # Sample until we find a point with better loglike.
            worst_param = self.live_params[worst]

            while self.live_loglike[worst] >= self.loglike(worst_param):
                new_live_cube = self.draw_new_point()
                self.live_params[worst] = self.prior_trans(new_live_cube)
                self.n_calls += 1

            # Update the worst live point to the new value.
            self.live_cubes[worst, :] = np.copy(new_live_cube)
            self.live_loglike[worst] = self.loglike(self.live_params[worst])

            # Estimate upper bound on evidence in the remaining volume
            dlogz = logvol + self.live_loglike.max() - self.lnz

            # Print progress of the sampler
            if not self.n_samples % 50:
                print("Number of accepted samples:", self.n_samples)
                print("Number of likelihood calls:", self.n_calls)
                print("Sampling efficiency:", self.n_samples/self.n_calls)
                print("lnZ:", np.round(self.lnz, 5))
                print("dlnZ:", np.round(dlogz, 5))
                print("-----------------------------------")

        # Extract only the used slice from the arrays.
        self.dead_loglike = self.dead_loglike[1:self.n_samples+1]
        self.post_wts = self.post_wts[1:self.n_samples+1]

        # Normalise the weights.
        self.post_wts -= self.lnz

        print("Sampling is complete, best fitting parameters:",
              self.live_params[np.argmax(self.live_loglike), :], "\n")

        # f.close()
