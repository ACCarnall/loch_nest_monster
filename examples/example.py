import numpy as np 
import matplotlib.pyplot as plt
import corner

import lochnest_monster as nesty


def make_fake_data_linear(x, param, sigma):

    m = param[0]
    c = param[1]

    y = m*x + c
    y += sigma*np.random.randn(x.shape[0])
    y_err = np.zeros_like(x) + sigma

    return y, y_err


x = np.arange(0., 20., 2.)

true_param = [1.5, 5.]  # Gradient, intercept.

y, y_err = make_fake_data_linear(x, true_param, 1.0)

plt.figure()

plt.errorbar(x, y, yerr=y_err, lw=1.0, linestyle=" ",
             capsize=3, capthick=1, color="black")

plt.scatter(x, y, color="blue", s=25, zorder=4, linewidth=1,
            facecolor="blue", edgecolor="black")

plt.show()


def lnprob(param):
    y_model = param[0]*x + param[1]
    return -0.5*np.sum(((y - y_model)/y_err)**2)

def prior_transform(cube):
    return 20.*cube-10.

sampler = nesty.nested_sampler(lnprob, prior_transform, 2)

sampler.run()