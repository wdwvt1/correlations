#!/usr/bin/env/python
#file created 2/26/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"


"""
This code is meant to create tables with OTUs that have simple time series 
relationships. 

All signals take the form of:
y_shift + alpha*signal_func(phi(theta+omega)) + noise

freq = [1, 2, 3]
amp = [1, .5, .25]
phase = [0, .25*pi, .5*pi]
noise = [0, .25, .5]
adj = [[subsample_otu_evenly, .5], [subsample_otu_zero, .5, .3], 
    [subsample_otu_zero, .5, .75]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
otus = vstack([generate_otu_from_pt_in_R5(q[i], sin, 10) for i in range(243)])

plt.plot(arange(50), otus[0], arange(50), otus[3], arange(50), otus[6])
plt.show()

This will show the 3 otus which differ only in the amount of noise they recieve. 
To plot specific otus you have to pull out their generating index. The 
generating index goes from 0-242. The generating index 0 corresponds to 
parameters [subsample_otu_zero, .5, .75], 0, 0 ,1 ,1 (in ascending order). The
generating index 1 just moves one to the right in the adj paramater. Following
this thinking, we would have to choose inds = [0,81,162] to get the three
otus which differ only in their frequency (because 3**4 is 81). 
"""

from numpy import (array, where, sin, cos, pi, hstack, linspace, arange,
    searchsorted, vstack, e)
from numpy.random import shuffle
from scipy.signal import square, sawtooth 
from scipy.stats.distributions import uniform

def add_noise(noise_func_and_params, data):
    '''Add noise at each index to given data based on func and params.
    Inputs:
     noise_func_and_params - list where first entry is a 
     scipy.stats.distribution function, and the other parameters are passed to 
     that function and given in the required order.
     data - 1d arr, otu data from generate_signal_data.'''
    noise = noise_func_and_params[0].rvs(*noise_func_and_params[1:],
        size=data.size)
    return data+noise

def signal(alpha, phi, omega, signal_func, sampling_freq=100 ,lb=0, ub=2*pi):
    '''Make a signal. 
    Inputs:
     alpha - float, amplitude factor. Set to 1 if you do not intend to modify
      the amplitude of the signal at all. Set to 0 to get uniform 0 signal. 
     phi - float, phase_shift factor. Set to 1 if you do not intend to modify 
      the frequency at all. Set to 0 to get uniform signal_func(t_0) signal.
     omega - float, phase_offset.
     signal_func - function used to generate the signal. 
     sampling_freq - int, number of points to generate between lb, ub.
     lb, ub - float, bonds of the signal.'''
    timepoints = linspace(lb, ub, sampling_freq)
    return alpha*signal_func(phi*(timepoints+omega))

def make_pop_growth_func(K, N_0, r):
    '''Create function for calculating logistic population growth given params.
    Inputs:
     K - numeric, carrying capacity of the population.
     N_0 - numeric, population at time 0. 
     r - Malthusian paramter, population growth rate.
    Example usage with signal function:
    _f = make_pop_growth_func(1000,50,.01)
    signal(1,1,0,_f)
    '''
    return lambda t: K/(1.+((K/N_0 - 1.0)*e**(-r*t)))

def superimpose_signals(signal_calls, group_y_shift, noise_func_and_params):
    '''Superimpose signals on top of one another.
    Inputs:
     signal_calls - 
     noise_func_and_params - list where first entry is a 
     scipy.stats.distribution function, and the other parameters are passed to 
     that function and given in the required order.'''
    signals = [signal(*i) for i in signal_calls]
    # add the y_shift to the whole group
    sup_sig = group_y_shift+reduce(lambda x,y: x+y, signals)
    noisy_sig = add_noise(noise_func_and_params, sup_sig)
    #return where(noisy_sig > 0, noisy_sig, 0.0)
    return noisy_sig

def make_otu(superimposed_signals):
    '''makes otu'''
    otu = hstack(superimposed_signals)
    return where(otu > 0, otu, 0.).astype(int)

def subsample_otu_random(otu, fraction):
    '''Subsample an otu by randomly taking fraction of otu points.'''
    inds = arange(len(otu))
    shuffle(inds)
    inds = array(sorted(inds[:int(len(otu)*fraction)]))
    return otu[inds]

def subsample_otu_choose(otu, indices):
    '''Return given indices of otu.'''
    return otu[indices]

def subsample_otu_evenly(otu, fraction):
    '''Subsample the otu evenly.
    Notes:
     If you wish to subsample a given (discrete) signal at an arbitrary depth 
     you can either guarantee that you get the given depth or you can guarantee
     that all your subsampled points are equally far apart. Consider trying to 
     get 40 points at even distance from 100 points. Since there is no integer
     multiple of 40 that is very close to 100 you are stuck spacing the points 
     out unevenly. If you had wanted to pick 33 points then you could start at
     0 and take every other 3rd, but the same doesnt apply for numbers that are
     not divisors of len(otu). This script ensures that you get as even sampling
     as you can while preserving the right number of points.'''
    inds = searchsorted(arange(len(otu)),
        linspace(0,len(otu)-1, int(len(otu)*fraction)), side='left')
    return otu[inds]


def cube_d5_indices(d1, d2, d3, d4, d5):
    '''Return indices of 5dhypercube wtih each dim split into len(dn) parts.
    Imagine that you are going to split each dimension into 3 parts. We can
    visualize this with the following diagram where each vertical group of 
    points is a dimension and each point is a location on that dimension. This
    function will return all distinct paths through the grid below where each 
    path goes through one location in each dimension.
    .    .    .    .    .
    .    .    .    .    .
    .    .    .    .    .
    d1   d2   d3   d4   d5

    The below represents the point (2,1,2,3,3) in R5
    3  .    .    .    x    x
    2  x    .    x    .    .
    1  .    x    .    .    .
       d1   d2   d3   d4   d5
    numpy.indices does this, but the notation is much harder for me to follow.
    could also make this with the binary numbers up to 243?
    '''
    vals = []
    for pt_d1 in d1:
        for pt_d2 in d2:
            for pt_d3 in d3:
                for pt_d4 in d4:
                    for pt_d5 in d5:
                        vals.append([pt_d1,pt_d2,pt_d3,pt_d4,pt_d5])
    return vals

def subsample_otu_zero(otu, ss_fraction, zero_fraction):
    '''Evenly subsample an OTU and randomly set zero_fraction entries = 0.'''
    ss_otu = subsample_otu_evenly(otu, ss_fraction)
    inds = arange(len(ss_otu))
    shuffle(inds)
    ss_otu[inds[:int(len(inds)*zero_fraction)]] = 0
    return ss_otu

def generate_otu_from_pt_in_R5(pt, wave_f, y_shift=None):
    '''Generate an OTU sequence from a pt in R5.'''
    freq, amp, phase_offset, noise, sampling_params = pt
    # noise is from a uniform distribution where amp*noise controls noise level
    noise_func_and_params = [uniform, -noise*amp, 2*noise*amp]
    # if y_pos is none we will randomly select the y_shift for the signal
    # between 50 percent of amplitude and 150 percent of amplitude
    if y_shift==None:
        y_shift = uniform.rvs(.5*amp, 1.5*amp)
    # make the base otu + y_shift
    base_otu = y_shift + signal(amp, freq, phase_offset, wave_f, 
        sampling_freq=100, lb=0, ub=2*pi)
    # add noise
    noisy_otu = add_noise(noise_func_and_params, base_otu)
    # subsample the otu according to the sampling_f and params
    sampling_f = sampling_params[0]
    return sampling_f(noisy_otu, *sampling_params[1:])

def random_inds(n, k):
    '''Return k indices randomly from arange(n) in ascending order.'''
    tmp = arange(n) 
    shuffle(tmp)
    inds = tmp[:k]
    inds.sort()
    return inds

################################################################################
# Functions unutilized but left in case of future need. Untested. 
################################################################################

def compose_two(g, f):
    '''Compose two functions.
    Adapted directly from http://joshbohde.com/blog/functional-python.
    '''
    return lambda *args: g(f(*args))

def signal_composition(signal_funcs):
    '''Compose >2 functions
    Adapted directly from: http://joshbohde.com/blog/functional-python
    '''
    return reduce(compose_two, signal_funcs)


