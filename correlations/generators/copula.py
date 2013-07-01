#!/usr/env/bin/python
# File created 2/13/2013
__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"


from numpy.random import multivariate_normal, uniform
from numpy import array, zeros, eye, tril, linspace, log, where
from scipy.stats import kendalltau, spearmanr
from scipy.stats.distributions import uniform, lognorm, beta, norm
from numpy.linalg import cholesky, LinAlgError


def copula(num_samples, rho_mat, mu_mat, methods):
    """Copula procedure to generate an OTU table with corrs close to rho_mat.
    Inputs:
     num_samples - int, number of samples. 
     rho_mat - 2d arr, symmetric positive definite matrix which specifies the 
     correlation or covariation between the otu's in the table. 
     mu_mat - 1d arr w/ len(num_otus), mean of otu for multivariate random call.
     methods - list of lists w/ len(num_otus), each list has a variable number 
     of elements. the first element in each list is the 
     scipy.stats.distributions function like lognorm or beta. this is the 
     function that we draw values from for the actual otu. the remaining entries
     are the parameters for that function in order that the function requires 
     them.
    """
    num_otus = len(mu_mat)
    # draw from multivariate normal distribution with specified parameters.
    # transpose so that it remains otuXsample matrix.
    Z = multivariate_normal(mean=mu_mat, cov=rho_mat, size=num_samples).T
    # using the inverse cdf of the normal distribution find where each sample 
    # value for each otu falls in the normal cdf.
    U = norm.cdf(Z)
    # make the otu table using the methods and cdf values. ppf_args[0] is the 
    # distribution function (eg. lognorm) whose ppf function we will use
    # to transform the cdf vals into the new distribution. ppf_args[1:] is the 
    # params of the function like a, b, size, loc etc. 
    otu_table = array([ppf_args[0].ppf(otu_cdf_vals, *ppf_args[1:], 
        size=num_otus) for ppf_args, otu_cdf_vals in zip(methods, U)])
    return where(otu_table > 0, otu_table, 0)

def scipy_corr(otu_table, corr_method):
    """Performs scipy.stats.corr_method between all rows."""
    res = eye(otu_table.shape[0]) #rowXrow comparison table
    for row in range(otu_table.shape[0]):
        for col in range(row+1, otu_table.shape[0]):
            res[row][col] = corr_method(otu_table[row], otu_table[col])[0] #p,t
            # otu_table[col] is a row, confusing variable name
    return res

def make_symmetric(arr, trace_1=False):
    """Make a square triangular matrix symmetric, set trace to 1 if True."""
    res = arr+arr.T
    if trace_1:
        for i in range(arr.shape[0]):
            res[i][i] = 1.
    else:
        for i in range(arr.shape[0]):
            res[i][i] = arr[i][i] # if arr not hollow, res[i][i]=2*arr[i][i]
    return res

def generate_rho_matrix(distribution, params, num_otus, iters):
    """Make a rho matrix according to the given distribution and parameters.
    For physically meaningful correlations you must create a positive-definite
    rho matrix. If it is not positive definite it implies a correlation that 
    can't exist. To test for if the matrix is PD the only measure I have found
    is the Cholesky decomposition. It is O(n**3), but it appears to be the only
    necessary and sufficient test for positive definiteness. 

    A positive definite matrix can be created with matrix multiplication A*A.T
    but this will alter the main diagonal away from 1 (called unit diagonal in 
    the literature). 

    If a uniform distribution is passed, you can guarantee that the random 
    rho matrix will be PD if the the bounds of the distribution are
    +- 1/(num_otus-1) according to:
    http://stats.stackexchange.com/questions/13368/off-diagonal-range-for-guaranteed-positive-definiteness
    
    The code will attempt to draw from the distribution you specified iter 
    number of times to create a positive definite matrix. If all iters trials 
    fail it will return an error. 
    """
    # draw from the distribution, reshape to be num_otusXnum_otus array
    for i in range(iters):
        rho = distribution.rvs(*params,
            size=num_otus**2).reshape(num_otus,num_otus)
        sym_rho = make_symmetric(rho, trace_1=True)
        try:
            _ = cholesky(sym_rho)
            print '%s iters were required to draw pos-def rho matrix.' % (i+1)
            return sym_rho
        except LinAlgError:
            pass
    raise ValueError('A symmetric, positive definite matrix could not '+\
        'be computed with the given inputs and random draw.')

