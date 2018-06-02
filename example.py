import numpy as np 
import matplotlib.pyplot as plt
import emcee
import corner
import os
import pymultinest as pmn

import lochnest_monster as nesty

class Fit:

	def __init__(self):

		self.true_m = 2.5
		self.true_c = 50.
		self.true_sigma = 5.
		self.ndim = 2

		self.make_fake_data()


	def make_fake_data(self):
		self.x = np.arange(1., 101., 5.)

		self.y = self.true_m*self.x + self.true_c

		for i in range(self.x.shape[0]):

			self.y[i] += self.true_sigma*np.random.randn()


	def prior(self, cube, *args):
		cube[0] *= 2*self.true_c
		cube[1] *= 2*self.true_m

		return cube


	def lnprob(self, param, *args):
		return -0.5*np.sum(np.log(2*np.pi*self.true_sigma**2)) - 0.5*np.sum(((self.y - self.model(param))/self.true_sigma)**2.)


	def model(self, param, *args):
		return param[0] + param[1]*self.x 


	def fit_emcee(self):

		sampler = emcee.EnsembleSampler(200, self.ndim, self.lnprob)

		p0 = np.zeros((200, self.ndim))

		p0[:,0] = self.true_m + 0.01*np.random.randn(200)
		p0[:,1] = self.true_c + 0.01*np.random.randn(200)

		sampler.run_mcmc(p0, 400)

		self.posterior = np.zeros((200*200, self.ndim+1))

		self.posterior[:,:-1] = sampler.chain[:, 200:, :].reshape((-1, self.ndim))

		self.post_models = np.zeros((self.x.shape[0], self.posterior.shape[0]))

		for i in range(self.posterior.shape[0]):
			self.post_models[:,i] = self.model(self.posterior[i,:-1])


	def fit_nest(self):

		self.sampler = nesty.nested_sampler(self.lnprob, self.prior, self.ndim, n_live=1000)

		self.sampler.run()

		#sampler.plot_contour()


	def fit_pmn(self):
		pmn.run(self.lnprob, self.prior, self.ndim, importance_nested_sampling = False, verbose = True, sampling_efficiency = "parameter", n_live_points = 400, outputfiles_basename="pmn/test-", const_efficiency_mode = True)

		a = pmn.Analyzer(n_params = self.ndim, outputfiles_basename="pmn/test-")

		s = a.get_stats()

		self.posterior = np.loadtxt("pmn/test-.txt")[:,2:]
		self.posterior_weights = np.loadtxt("pmn/test-.txt", usecols=(0))

		mode_evidences = []
		for i in range(len(s["modes"])):
			mode_evidences.append(s["modes"][i]["local log-evidence"])

		f = open("pmn/test-resume.dat")

		f.readline()
		self.niter = int(f.readline().split()[1])
		print " "
		print "Best fit parameters: ", s["modes"][np.argmax(mode_evidences)]["maximum"]
		print "Global Z: ", np.exp(s['nested sampling global log-evidence'])
		print "No of iterations: ", self.niter
		print " "
		raw_input()

		#self.post_models = np.zeros((self.x.shape[0], self.posterior.shape[0]))

		#for i in range(self.posterior.shape[0]):
		#	self.post_models[:,i] = self.model(self.posterior[i,:-1])



	def plot_corner(self):

		corner.corner(self.posterior, weights=self.posterior_weights, smooth=1, smooth1d=0.5, truths=[self.true_m, self.true_c], quantiles=[0.16, 0.5, 0.84], labels=["$m$", "$c$"], show_titles=True)
		
		plt.show()


	def plot_data(self):
		plt.figure()
		plt.errorbar(self.x, self.y, yerr=self.true_sigma, lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=3, color="black")
		plt.scatter(self.x, self.y, color="blue", s=25, zorder=4, linewidth=1, facecolor="blue", edgecolor="black")
		plt.show()


	def plot_fit(self):
		plt.figure()
		plt.errorbar(self.x, self.y, yerr=self.true_sigma, lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=3, color="black")
		plt.scatter(self.x, self.y, color="blue", s=25, zorder=4, linewidth=1, facecolor="blue", edgecolor="black")
		plt.plot(self.x, np.median(self.post_models, axis=1), color="darkorange", zorder=8)
		plt.fill_between(self.x, np.percentile(self.post_models, 16, axis=1), np.percentile(self.post_models, 84, axis=1), alpha=0.5, color="navajowhite")
		plt.show()




testfit = Fit()

#testfit.plot_data()

#testfit.fit_pmn()

#testfit.plot_corner()

#os.system("rm pmn/*")

testfit.fit_nest()

#testfit.sampler.plot_contour()

"""
testfit.fit_emcee()
testfit.plot_corner()
testfit.plot_fit()
"""



