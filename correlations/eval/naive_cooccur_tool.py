#!/usr/bin/evn python

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__version__ = "1.7.0-dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"
__status__ = "Development"

from qiime.stats import (assign_correlation_pval)
from qiime.otu_significance import (CORRELATION_TEST_CHOICES)
from numpy import array, zeros

"""
This library contains code for evaluating co-occurrence using a naive approach. 
"""

def naive_cc_tool(bt, corr_method, pval_assignment_method, cval_fp, pval_fp):
    '''Calculate co-occurence using naive approach.

    Inputs:
     bt - biom table with OTUs to be correlated.
     corr_method - str, correlation statistics to use, one of pearson, 
     spearmans_rho, or kendalls_tau.
     pval_assignment_method - str, one of parametric_t_distribution, 
     fisher_z_transform, bootstrapped, kendall.
    '''
    data = array([bt.data(i, axis='observation') for i in bt.observation_ids])
    r,c = data.shape
    ccs = zeros((r,r))
    ps = zeros((r,r))
    test_fn = CORRELATION_TEST_CHOICES[corr_method]
    for o1 in range(r):
        for o2 in range(o1+1,r):
            cc = test_fn(data[o1], data[o2])
            ccs[o1][o2] = cc
            # assign correlation pvalues
            if pval_assignment_method == 'None':
                ps[o1][o2] = 1.0
            else:
                pval = assign_correlation_pval(cc, len(data[o1]), 
                    pval_assignment_method, permutations=1000, 
                    perm_test_fn=test_fn, v1=data[o1], v2=data[o2])
                ps[o1][o2] = pval
    # write values
    header = '#OTU ID\t'+'\t'.join(bt.observation_ids)
    clines = [header]+[bt.observation_ids[i]+'\t'+'\t'.join(map(str,ccs[i])) \
        for i in range(r)]
    plines = [header]+[bt.observation_ids[i]+'\t'+'\t'.join(map(str,ps[i])) \
        for i in range(r)]
    o = open(cval_fp, 'w')
    o.writelines('\n'.join(clines))
    o.close()
    o = open(pval_fp, 'w')
    o.writelines('\n'.join(plines))
    o.close()

