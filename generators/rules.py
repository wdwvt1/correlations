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


"""Code for generating OTU tables from the rule model. 

The rule models central concept is that the presence of a given set of OTUs 
('inducers') will determine the presence of a another OTU according to some
defined procedure. There are currently two models implemented. Both share 
an initial step. 

Shared procedure:
X random OTU vectors are created (via independent draws of a given distribution
or via some other procedure) and a set of rules is specified where for any 
given sample, the value of the induced OTU for that sample is (at least 
partially) dependent on the values of the X inducing OTUs.

Divergenet procedure
Model 1: this model has the presence or absence of the induced OTU controlled by
a set of rules on the inducing OTUs. The more rules or constraints that the 
inducing OTUs satisfy, the more likely the presence of the induced OTU. The 
basic function logic is to draw a the induced OTU vector from whatever the 
passed distribution is, and then multiply it by a binary vector of 1's and 0's. 
This binary vector is created by calculating the number of rules or conditions 
the inducing OTUs satisfy at each sample and then using the passed weights 
to calculate how likely the binary vector is to have a 0 at a given sample based
on how many inducing OTUs satisfied conditions for this sample. This aims to be
a simple approximation of mutualism or competition where certain species 
populations are controlled by other species in a direct way. 

example

Model 2: this model has the value for the induced OTU as a function of the 
inducing OTUs plus some amount of noise. This model has a similar internal logic
to Model 1, but instead of the binary weighting vector + random draw for 
determining the induced OTU abundance, its a direct function of the abundances
of the inducers. Any inducing OTUs which add to the value of the induced OTU
are added first before subtraction occurs. This is because the presence of some
OTU might cause the overall decline of the induced OTU (for example a simple 
predator prey model where each predator reduces the total population that would
otherwise exist by 10X predator population). 

example

r = [[0.4, 'self'], [2.0, 'self'], [0.5, 'self']]
model2_otu(ia, r)
ia = 
array([[74, 32, 32, 95, 21, 30, 74, 28, 33, 49],
       [61, 44, 59, 21, 40, 30, 45, 56, 90, 85],
       [75, 51, 39, 83, 92, 48, 18, 56,  3, 75]])

"""



def model1_eval_rule(lb, ub, val):
    """Evaluate if a rule is satisfied.
    All rules are of the form lb <= val < ub unless lb=ub=0, then returns True
    if val=0."""
    if lb == ub == val == 0: #user wants val to be 0, implies absence of OTU
        return True
    elif lb <= val < ub:
        return True
    else:
        return False

def model1_eval_rules(rules, vals):
    """Evaluate a list of rules for number of rules satisified."""
    return sum([model1_eval_rule(lb=r[0], ub=r[1], val=v) for r,v in 
        zip(rules, vals)])

def model1_otu(inducer_arr, df_and_params, weights, rules):
    """Creates an OTU vector according to model 1 and given params.
    Inputs:
     inducer_arr - 2d array or OTU vectors which are inducing vectors for the 
     created otu.
     df_and_params - list with [0]=scipy.stats.distribution func, and following 
     entries params for the df in proper order. 
     weight_arr - arr, prob(out_vector[i]!=0 | x rules satisfied). weight arr is 
     len(rules)+1 for model 1. The reason for the +1 is that if no rules are 
     specified, a base probability for occurrence of the OTU is still needed.
     rules - list of len(2) lists with l[0]=lower bound, l[1] upper bound for
     rule evaluation.
     """
    num_samples = inducer_arr.shape[1]
    # random draw from distribution according to parameters
    unweighted_draw = df_and_params[0].rvs(*df_and_params[1:],size=num_samples)
    # find out number of conditions satisfied and create uniform draw matrix.
    cs = [model1_eval_rules(rules, otu_vals) for otu_vals in inducer_arr.T]
    # draw from uniform distribution and weight according to weights matrix.
    res = where(uniform.rvs(0,1,size=num_samples) > 1-weights.take(cs), 1., 0)
    # multiply element wise to produce output vector
    return (res*unweighted_draw).astype(int)

def model2_eval_rules(rules, vals):
    """Calculate value of new OTU based on rules and vals.
    Inputs:
     rules - list of len(2) lists with l[0] float (amount to multiply by when 
     creating new vector) and l[1] str ('add' or 'sub' where add means 
     multiply l[0] by inducing OTU abundance and 'sub' means multiply l[0] by
     inducing OTU abundance and subtract from total after all 'add' additions
     have been made).
    """
    add_total = sum([val*r[0] for val, r in zip(vals,rules) if r[1]=='add'])
    sub_total = sum([val*r[0] for val, r in zip(vals,rules) if r[1]=='sub'])
    notu_val = add_total-sub_total
    return notu_val if notu_val > 0 else 0

def model2_otu(inducer_arr, rules):
    """Creates an OTU vector according to model 2 and given params.
    Inputs:
     inducer_arr - 2d array or OTU vectors which are inducing vectors for the 
     created otu.
     rules - list of len(2) lists with l[0] float (amount to multiply by when 
     creating new vector) and l[1] str ('self' or 'sum' where self means 
     multiply l[0] times inducing OTU abundance and 'sum' means multiply l[0] by
     induced OTU abundance after all 'self' additions have been made).
    """
    # create rule function which assigns to each inducer array input an output 
    raw_otu_vals = array([model2_eval_rules(rules,i) for i in inducer_arr.T])
    # add noise
    return uniform.rvs(.9,.2,size=raw_otu_vals.shape[0])*raw_otu_vals


