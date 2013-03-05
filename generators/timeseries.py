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


example of generating a signal

# generate the first part of the OTU signal as the sum of a square and sawtooth
# wave. 
gen1 = [[8, 2, 0, square],
       [8, 2, .5*pi, sawtooth]]
# generate the second part of the otu signal as a single sin wave. 
gen2 = [[8, 2, 0, sin]]

# noise function and params as well as the general y_shift
nfap = [uniform, -3, 6]
gys = 20

# do signal generation
s1 = superimpose_signals(gen1, gys, nfap)
s2 = superimpose_signals(gen2, gys, nfap)

#sum to make otu
o1 = make_otu([s1,s2])
"""

from numpy import (array, where, sin, cos, pi, hstack, linspace, arange,
    searchsorted)
from numpy.random import shuffle
from scipy.signal import square, sawtooth 
from scipy.stats.distributions import uniform

def generate_signal_data(alpha, phi, omega, signal_func, timepoints):
    '''Make signal data with func and given offset, shift, and amplitude.'''
    return alpha*signal_func(phi*(timepoints+omega))

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
     alpha - float, amplitude factor.
     phi - float, phase_shift.
     omega - float, phase_offset.
     signal_func - function used to generate the signal. 
     sampling_freq - int, number of points to generate between lb, ub.
     lb, ub - float, bonds of the signal.'''
    timepoints = linspace(lb, ub, sampling_freq)
    return generate_signal_data(alpha, phi, omega, signal_func, timepoints)

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
    return inds, otu[inds]

def subsample_otu_choose(otu, indices):
    '''Return given indices of otu.'''
    return otu[array(indices)]

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
    return inds, otu[inds]

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


