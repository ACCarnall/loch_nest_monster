import numpy as np 
import matplotlib.pyplot as plt
import corner

import lochnest_monster as nesty


def lnprob(param):
    y_model = make_fake_data_linear(x, param)
    return -0.5*np.sum(((y - y_model)/y_err)**2)


def prior_transform(cube):
    return 20.*cube-10.


def make_fake_data_linear(x, param, sigma=None):
    m = param[0]
    c = param[1]

    y = m*x + c
    if sigma:
        #y += sigma*np.random.randn(x.shape[0])
        y_err = np.zeros_like(x) + sigma

        return y, y_err

    return y


# Make some fake straight line data to fit
x = np.arange(0., 20., 2.)
true_param = [1.5, 5.]  # Gradient, intercept
y, y_err = make_fake_data_linear(x, true_param, 1.0)

# Make a plot of the fake data
plt.figure()
plt.errorbar(x, y, yerr=y_err, lw=1.0, linestyle=" ",
             capsize=3, capthick=1, color="black")

plt.scatter(x, y, color="blue", s=25, zorder=4, linewidth=1,
            facecolor="blue", edgecolor="black")

plt.show()

# Set up the sampler and sample the posterior
sampler = nesty.ellipsoid_sampler(lnprob, prior_transform, len(true_param),
                                  verbose=True, live_plot=False, n_live=400)

# Try out the nball_sampler and box_sampler,
# also try setting live_plot to True.

sampler.run()

# Make a corner plot of the results
corner.corner(sampler.results["samples_eq"])
plt.savefig("example_corner.pdf", bbox_inches="tight")
