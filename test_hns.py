import numpy as np
import ray

import raynest.model
from raynest.parameter import LivePoint
from scipy.stats import norm, beta, uniform, expon, invgamma
from scipy.special import gammaln, logsumexp
from lal import MSUN_SI, C_SI, G_SI, PC_SI
from itertools import cycle, product
from math import lgamma
import sys
import matplotlib.pyplot as plt

from waveform import orbital_frequency, amplitude, chirp, distance

def astrophysical_distribution(m, mu, var):
    return norm(mu,np.sqrt(var)).pdf(m)

def sample_astrophysical_distribution(mu, var, n0):
    return np.abs(norm(mu,np.sqrt(var)).rvs(size = n0))

def volume_distribution(r, z, rho0, r0, z0):
    return rho0*np.exp(-r/r0)*np.exp(-np.abs(z)/z0)

def sample_volume_distribution(r0, z0, n0):
    return distance(expon(scale=r0).rvs(size=n0),expon(scale=z0).rvs(size=n0))

def dirichlet_sample_approximation(G0, alpha = 1, tol=0.01):
    """
    Generate samples from the base distribution
    This is a set of thetas so that the observed distribution is
    G = \sum w_i theta_i
    
    G0 should be an iterable
    """
    betas = []
    pis = []
    betas.append(beta(1, alpha).rvs())
    pis.append(betas[0])

    while 1-sum(pis) > tol:
        s = np.sum([np.log1p(-b) for b in betas])
        new_beta = beta(1, alpha).rvs()
        betas.append(new_beta)
        pis.append(new_beta * np.exp(s))
        
    D   = len(G0)
    n   = len(pis)
    
    # generate a number of samples equal to the number of observed categories (events)
    thetas = np.column_stack([G0[i].rvs(size=n) for i in range(D)])

    return pis, thetas

def snr(h,d,T,srate,asd=1e-18):
    dt = 1.0/srate
    return np.sqrt((4./T)*np.sum(h*d/(dt*asd)**2))

def generate_injection_parameters(events):
    return [LivePoint(['M','r','z','phi'], d=np.array((events[i,0],events[i,1],events[i,2],events[i,3]))) for i in range(events.shape[0])]

def generate_signal_parameters(x, inj = None):
    # this is the individual PE job
    if inj is not None:
        return [LivePoint(['M','D','phi'], d=np.array((inj[i]['M'],
                                                       distance(inj[i]['r0'],inj[i]['z0']),
                                                       inj[i]['phi']))) for i in range(len(inj))]

    else:
        p = [LivePoint(['M','D','phi'], d=np.array((x[i,0],distance(x[i,1],x[i,2]),x[i,3]))) for i in range(x.shape[0])]
        return p

def default_prior_density(p):
    
    if p['M'] < 0. or p['M'] > 3.0:
        return -np.inf
    if p['phi'] < 0.0 or p['phi'] > 2*np.pi:
        return -np.inf
    logP = 2.*np.log(distance(p['r'],p['z']))
    return logP

def default_likelihood(p, data, sigma = 1e-18):
    signal = chirp(p, t, t0 = 30000)
    r = (data-signal)/sigma
    return np.sum(-0.5*r**2), signal

def evolve_signal_parameters(p, data, n = 100):
    
    acc    = 0
    p0     = p.copy()
    logP0  = p0.logP + p0.logL
    d      = len(p0.values)
    signal = np.zeros(len(data))
    default_variance = np.array((0.01, 10, 5, 0.1))
    pt = np.column_stack([np.random.normal(0,d,size=n) for d in default_variance])

    for trials in range(n):
#        print('trials',trials)
        p0.values += pt[trials,:]
        p0.logP = default_prior_density(p0)
#        print(p0.logP,p0.values)
        p0.logL, sig = default_likelihood(p0, data)
#        print(p0.logL,p0.values)
        logP = p0.logP + p0.logL
#        print(logP,logP0,p0.values)
        if logP-logP0 > np.log(np.random.uniform(0,1)):
            logP0 = logP
            p = p0.copy()
            signal = sig
            acc += 1
        else:
            p0 = p.copy()

    return p, logP, signal

def sine_gaussian(x,t):
    e = (t-x['t0'])/x['tau']
    return x['A']*np.exp(-e**2)*np.cos(2*np.pi*x['f']*t + x['phi'])

def generate_noise(t, sigma = 1.0):
    return norm(0.0,sigma).rvs(size = len(t))

def generate_signals(t, pars, t0 = 30000):
    return np.array([generate_signal(t, x, t0 = t0) for x in pars])

def generate_signal(t, x, t0 = 30000):
    return chirp(x, t, t0 = t0)

#@ray.remote(num_cpus = 4)
def compute_logL(theta, time, data, n, injection = None):

    logL          = np.zeros(n, dtype=np.float64)
    signals_array = np.zeros((len(theta), len(time)), dtype=np.float64)
    post          = []
    for j,t in enumerate(theta):
        signals_array[j,:] = generate_signal(time, LivePoint(['M','r','z','phi'],d=t))
    
    # gibbs sample the signals
    for i in range(n):
        for j,t in enumerate(theta):
            p, logP, sig = evolve_signal_parameters(LivePoint(['M','r','z','phi'],d=t), data, n = 10)
            signals_array[j,:] = sig
            
            if i == n-1:
                post.append(p)
            # for each sample, generate the signals
        #    signals_array = generate_signals(time, x, t0 = 30000)

            # compute the optimal snr
            signal_to_noise_ratio  = np.array([snr(s,s,T,srate) for s in signals_array])
            detections, = np.where(signal_to_noise_ratio > 0)
        #        detections = set(detections)  #Set is more efficient, but doesn't reorder your elements if that is desireable
        #    mask = np.array([True for _ in range(len(signals_array))])
            mask    = np.array([(d in detections) for d in range(len(signal_to_noise_ratio))])
        #
            sigma_h = np.sum(np.square(signals_array[~mask,:].sum(axis=0)))
            s       = np.sqrt(sigma**2+sigma_h)
        #            s = sigma
        #
        if np.all(mask) == False:
            logL[i] = -0.5*np.sum((data/s)**2)
        else:
            # compute the residuals
            r = (data - signals_array[mask,:].sum(axis=0))/s
            # and the likelihood
            logL[i] = -0.5*np.sum(r**2)

    return post, logsumexp(logL)/n

class HModel(raynest.model.Model):

    def __init__(self, time, data, mc_n = 1, injection = None):
        self.time       = time
        self.data       = data
        self.injection  = injection
        self.names      = ['concentration_parameter', 'N', 'mu_m', 'sigma_m', 'r0', 'z0']
        self.bounds     = [[0,10], [1,10], [mu-0.05,mu+0.05], [sigma-0.05, sigma+0.05], [r0-1000, r0+1000], [z0-100, z0+100]]
        self.nbins      = 8
        self.bins       = [np.linspace(self.bounds[2][0],self.bounds[2][1],self.nbins),
                           np.linspace(self.bounds[4][0],self.bounds[4][1],self.nbins),
                           np.linspace(self.bounds[5][0],self.bounds[5][1],self.nbins),
                           np.linspace(0.0,2*np.pi,self.nbins)]
        self.mc_n       = mc_n

    def log_likelihood(self, p, sigma = 1e-18):
        # sample the DP to get the masses from the discrete mass distribution (the universe realisation of the mass function)
        G0            = (norm(p['mu_m'], p['sigma_m']), expon(scale=p['r0']), expon(scale=p['z0']), uniform(0.0,2*np.pi))
        wi, theta     = dirichlet_sample_approximation(G0, alpha = p['concentration_parameter'], tol = 0.001)
        post, logL    = compute_logL(theta, self.time, self.data, self.mc_n, injection = self.injection)
        # marginalise (averaging)
        
        Arr = np.row_stack([p.values for p in post])

        # histogram the parameters
        q, edges = np.histogramdd(Arr, bins = self.bins, weights = wi)

        bincenters = np.array([(e[1:]+e[:-1])/2. for e in edges])
        dx         = np.prod([(np.abs(e[1]-e[0])) for e in edges])
        
        X = product(*bincenters)
        A = np.array([x for x in X])
        a = p['concentration_parameter']
        prior_counts     = a-1
        counts           = np.ravel(q)
        posterior_counts = prior_counts + counts
            
        predictions   = np.prod([G0[i].pdf(A[:,i]) for i in range(theta.shape[1])],axis=0)*dx
        predictions  /= np.sum(predictions)
        normalisation = np.sum([lgamma(e+1) for e in posterior_counts])-lgamma(np.sum(posterior_counts+1))

        logL_dp = np.dot(posterior_counts,np.log(predictions))-normalisation
            
        return logL+logL_dp

    def log_prior(self,p):
        logP = super(HModel,self).log_prior(p)
        if np.isfinite(logP):
            logP += -np.log(p['sigma_m'])
            logP += -np.log(p['N'])
            logP += invgamma.logpdf(p['concentration_parameter'],1)
        return logP
    
    def run(self):
        pass

if __name__ == "__main__":
#    for i in range(100):
#    print("====> i = ",i)# 66 = loud
    np.random.seed(66)
    # generate the noise
    T = 20000
    srate = 0.2/8.
    sigma_noise = 1e-18
    t      = np.linspace(0,T,int(T*srate))
    noise  = generate_noise(t, sigma = sigma_noise)
    # generate the population, assume q=1, data from https://arxiv.org/pdf/1606.05292.pdf
    mu     = 0.68*2
    sigma  = 0.13*np.sqrt(2)
    n0     = 1
    r0     = 3471
    z0     = 274
    alpha  = 1
    
    print("length = ", int(T*srate), len(t))
        
    if n0 > 0:
        inj_hyper_params = {'concentration_parameter':alpha, 'mu_m':mu, 'sigma_m':sigma, 'r0':r0, 'z0':z0, 'N':n0}
        G0 = (norm(inj_hyper_params['mu_m'], inj_hyper_params['sigma_m']),
              expon(scale=inj_hyper_params['r0']), expon(scale=inj_hyper_params['z0']), uniform(0.0,2*np.pi))
              
        w, events  = dirichlet_sample_approximation(G0, alpha = inj_hyper_params['concentration_parameter'], tol=1e-6)

        inj_params = generate_injection_parameters(events)
        # generate the signals
        signals= generate_signals(t, inj_params, t0 = 30000)
    # get the data
        data = noise+signals.sum(axis=0)
        print("snrs = ", [snr(s,s,T,srate) for s in signals])
    else:
        data = noise
    
    print("noise likelihood = ", -0.5*np.sum((noise/sigma_noise)**2))
    print("data likelihood = ", -0.5*np.sum((data/sigma_noise)**2))
    
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    ax.plot(t, data, 'k', lw=0.7, label='data')
    ax.plot(t, noise, 'r', lw=0.5, label='noise', alpha=0.5)
    try:
        for s in signals:
            ax.plot(t,s, lw=0.7)
            ax.plot(t, signals.sum(axis=0), lw=1.5, color='purple', alpha=0.5)
    except:
        pass
    plt.legend()
    plt.savefig('test/data.pdf', bbox_inches='tight')
    plt.close(fig)

    M = HModel(t, data, injection = None)
    
    if sys.argv[1] == '1':
        work=raynest.raynest(M, verbose=2, nnest=1, nensemble=3, nlive=100, maxmcmc=20, nslice=0, nhamiltonian=0, resume=0, periodic_checkpoint_interval=1800, output='test')
        work.run()
        posterior_samples = work.posterior_samples
    else:
        import h5py
        with h5py.File('test/raynest.h5','r') as f:
            logZ = f['combined']['logZ'][()]
            posterior_samples = f['combined']['posterior_samples'][:]
        print(len(posterior_samples))

    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    m   = np.linspace(0.0,3.0,100)

    for s in posterior_samples:
        ax.plot(m, astrophysical_distribution(m,s['mu_m'],s['sigma_m']), 'turquoise', lw=0.7, alpha=0.5)
    for inj in inj_params:
        ax.vlines(inj['M'],0.05,0.9,colors='r',alpha=0.5,lw=0.5)

    ax.plot(m, astrophysical_distribution(m,mu,sigma), 'k', lw=0.7, label='injection')
    plt.legend()
    plt.savefig('test/mass_dist_reconstruction.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(2)
    ax  = fig.add_subplot(111)
    r   = np.linspace(1.0,5000.0,100)
    z   = np.linspace(1.0,500.0,100)
    X,Y = np.meshgrid(r,z)
    C   = ax.contourf(X,Y, volume_distribution(X, Y, 1, inj_hyper_params['r0'], inj_hyper_params['z0']), 100)
    ax.set_xlabel('r[pc]')
    ax.set_ylabel('z[pc]')
    plt.colorbar(C)
    plt.savefig('test/volume_reconstruction.pdf', bbox_inches='tight')
    plt.close(fig)
    
    import corner
    
    fig = corner.corner(np.array([posterior_samples[n] for n in inj_hyper_params.keys()]).T,
                  labels=[k for k in inj_hyper_params.keys()],
                  quantiles=[0.05, 0.5, 0.95], truths = [inj_hyper_params[n] for n in inj_hyper_params.keys()],
                  show_titles=True, title_kwargs={"fontsize": 12}, smooth2d=1.0)
    plt.savefig('test/hyper_injection.pdf', bbox_inches='tight')
    plt.close(fig)
#
#    for n in M.names:
#        fig = plt.figure()
#        ax  = fig.add_subplot(111)
#        ax.hist(M.t[n], bins=100, density=True)
#        if n == 'mu':
#            for m in events:
#                ax.axvlines(m)
#        plt.savefig('test/{0}.pdf'.format(n), bbox_inches='tight')
#        plt.close(fig)
        
