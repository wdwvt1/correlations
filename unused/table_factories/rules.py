#!/usr/bin/env/python
#file created 2/13/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"

from numpy import array, where, apply_along_axis, sum
from scipy.stats.distributions import uniform, lognorm
from generators.rules import model1_otu


# Table 1:

# this produces a roughly 60 percent sparse data set that looks good for use 
# in rule based approaches. 
raw_vals = expon.rvs(-70,80,size=10000).astype(int)
abunds = where(raw_vals > 0, raw_vals, 0)


class mod_beta:
    """Distribution funcs that can be called like scipy.stats.distributions 1's.
    """
    def __init__(self):
        # don't need to do anything, just class so we can have the rvs method
        pass
    def rvs(self, loc, scale, size):
        '''Draw from exponential random variable with vals < 0 -> 0.'''
        # produces roughyl 60 percent sparsity with the following call
        # expon.rvs(-70,80,size=10000)
        raw_vals = expon.rvs(loc,scale,size).astype(int)
        abunds = where(raw_vals > 0, raw_vals, 0)
        return abunds

class mod_normal:
    """Distribution funcs that can be called like scipy.stats.distributions 1's.
    """
    def __init__(self):
        pass
    def rvs(self, loc, scale, size):
        pass








# test 2/25/2013
# create 200 otu X 50 sample table. use 4 otu's per rule 

# generate background otus which will be used to induce 
w = lognorm.rvs(2,1,size=10000).astype(int).reshape(200,50)
# generate model 1 rules
model1_rules = []
for i in range(10):
    rule_vals = uniform.rvs(-100,400,size=8)
    # sort rules so that left is lb <= ub
    rule_vals = where(rule_vals > 0, rule_vals, 0).reshape(4,2)
    rule_vals.sort()
    model1_rules.append(map(list, rule_vals))
# define weights and new otu distribution function
weights = array([1.0, 1.0, 1.0, 1.0, .2])
df_and_params = [lognorm, 2, 1]
# create new otus
table_index = 0
new_otus = []
table_map = []
for rules in model1_rules:
    tmp_table_map = [table_index]
    inducer_arr = w[table_index:table_index+4]
    notu = model1_otu(inducer_arr, df_and_params, weights, rules)
    new_otus.append(notu)
    table_index+=4
    tmp_table_map.append(table_index-1)
    table_map.append(tmp_table_map)




