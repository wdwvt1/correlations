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
Code for verifying that the tables being created and operations being used are 
returning mathematically correct data. Essentially math test code.
'''

from numpy import array, where, arange, argsort, bincount
import matplotlib.pyplot as plt

def plot_sample_abundance(data):
    '''Plot an OTU, sample abundance curve for OTU presence in samples.'''
    bc = bincount(data.astype(bool).sum(1), minlength=data.shape[1])
    plt.bar(arange(len(bc)), bc, width=1.0)
    plt.xlim(0,len(bc))
    plt.show()

