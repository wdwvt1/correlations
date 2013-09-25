#!/usr/bin/env/python
#file created 9/14/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"

'''
Utility code for basic operations.
'''

from numpy import array, where, arange, argsort
from numpy.random import shuffle
from copy import copy

def coercive_zero_inflation(data, zero_fraction, exact=False):
    '''Cause zero_fraction of data to be 0s.
    Inputs:
     data - 2d array of numeric values.
     zero_fraction - float in (0.0,1.0), fraction of values of data to turn to 0
     exact - boolean, if True, will calculate the fraction of zeros that data 
     already has and add only enough more to raise overall fraction to 
     zero_fraction.'''
    assert 1>zero_fraction>0, 'zero_fraction must be in (0,1.0)' 
    tmp = copy(data)
    if exact:
        zero_fraction = zero_fraction - (tmp==0).sum()/float(tmp.size)
    if zero_fraction < 0:
        raise ValueError('zero fraction of data higher than passed fraction.')
    if exact: 
        r,c = tmp.nonzero()
    else:
        r,c = where(tmp==tmp)
    inds = arange(len(r))
    shuffle(inds)
    ub = round(zero_fraction*tmp.size)
    tmp[r[inds[:ub]],c[inds[:ub]]] = 0
    return tmp

def subtraction_zero_inflation(data, zero_fraction):
    '''Subtract x from data such that data has ~ zero_fraction 0s.
    Can't guarantee exactness because data may have a bunch of repeated values, 
    e.g. data=ones((N,M)).
    Inputs:
     data - 2d array of numeric values.
     zero_fraction - float in (0.0,1.0), fraction of values of data to turn to 0
    '''
    # python zero indexes so we have to subtract one to find the median value
    tmp = data.flatten()
    x_median = tmp[argsort(tmp)[round(tmp.size*zero_fraction) - 1]]
    return where(data - x_median > 0, data - x_median, 0), x_median
    

