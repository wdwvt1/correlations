#!/usr/env/bin/python

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"



'''This code is utilized to test the stability and distributional properties of
the copula procedure.'''

##########
#Stability
##########
'''This section tests how stable the copula procedure is as a function of input
matrix size, and distribution of correlation coefficients.


The factors I believe will affect stability are the number of otus, the number
of samples (the length of the otu vectors), the structure and distributional 
properties of the Rho (correlation) matrix, and the distributional properties of
the otu vectors. 

Test from 
'''
# plotting code
import matplotlib.pyplot as plt
from matplotlib.pylab import matshow

from generators.copula import (copula, scipy_corr, make_symmetric,
    generate_rho_matrix)
from numpy import zeros, empty, arange, mean, log10, array
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt

def stability_test(num_otus, num_samples, rho_mat_args, sample_draw_args, 
    iters):
    '''Test stability at the given coordinates.
    Input ex: 
    kt,sp,rhos = stability_test(10,25,[uniform, -1,2], [lognorm, 2,0], 50)
    the rho_mat_args is a function, then the non size inputs for that function.
    in this case we have uniform distribution function and are passing -1,1 as 
    the bounds. size will always be passed separately. the lognorm call is the 
    same
    Note -- scipy uniform works strangely. [-1,2] -> a=-1, b=1. 
    '''
    # make the rho (correlation) matrix
    rhos = generate_rho_matrix(rho_mat_args[0], rho_mat_args[1:], 
        num_otus=num_otus, iters=10)
    # make the methods list for use with copula function. add num samples as 
    # the required size for the draw
    sample_draw_args.append(num_samples)
    methods = [sample_draw_args]*num_otus
    # make array of means from which mvariate normal distribution is drawn
    mus = zeros(num_otus) # always draw at 0. 
    # prepare results containers. store results in otuXotuXiters matrix
    # judge based on kendalls tau, spearman
    kt_res = empty((iters,num_otus,num_otus)) #depth of matrix is iters
    sp_res = empty((iters,num_otus,num_otus)) #depth of matrix is iters
    for i in range(iters):
        otu_table = copula(num_samples=num_samples, rho_mat=rhos, mu_mat=mus,
            methods=methods)
        kt_res[i] = scipy_corr(otu_table, kendalltau)
        sp_res[i] = scipy_corr(otu_table, spearmanr)
    return kt_res, sp_res, rhos


def plot_mean_deviation(res, incov, title_string):
    '''Plots the mean deviations between the input rho matrix and the output
    rho matrix in a variety of ways.
    Inputs:
     res - num_otusXnum_otusXiters 3d array. the result of stability test.
     incov - num_otusXnum_otus 2d array. input rho matrix. 
     title_string - suptitle of the figure, str.
    '''
    
    mean_diffs = res-incov
    mean_percentage_diffs = (res-incov)/incov

    mean_diff_boxplot_vals = []
    mean_percentage_diff_boxplot_vals = []

    for i in range(incov.shape[0]):
        for j in range(i+1, incov.shape[0]):
            mean_diff_boxplot_vals.append(mean_diffs[:,i,j])
            mean_percentage_diff_boxplot_vals.append(mean_percentage_diffs[:,i,j])

    mean_diff_hist_vals = []
    [mean_diff_hist_vals.extend(i) for i in mean_diff_boxplot_vals]
    mean_percentage_diff_hist_vals = []
    [mean_percentage_diff_hist_vals.extend(i) for i in 
        mean_percentage_diff_boxplot_vals]

    print ('Mean difference in-out cov: %s\n'+\
          'Mean fraction difference in-out cov: %s') % \
          (mean(mean_diff_hist_vals),mean(mean_percentage_diff_hist_vals))

    fig = plt.figure()
    fig.suptitle(title_string)
    sb1 = fig.add_subplot(221)
    sb1.boxplot(mean_diff_boxplot_vals, 
        positions=arange(len(mean_diff_boxplot_vals)))
    sb1.set_title('mean deviation')
    sb1.grid()
    sb1.minorticks_on()
    sb1.tick_params(axis='x', labelsize=5)
    sb1.set_xlabel('covariance matrix index')

    sb2 = fig.add_subplot(222)
    sb2.boxplot(mean_percentage_diff_boxplot_vals, 
        positions=arange(len(mean_percentage_diff_boxplot_vals)))
    sb2.set_title('mean fraction of input deviation')
    sb2.grid()
    sb2.minorticks_on()
    sb2.tick_params(axis='x', labelsize=5)
    sb2.set_xlabel('covariance matrix index')

    sb3 = fig.add_subplot(223)
    sb3.hist(mean_diff_hist_vals,bins=100,cumulative=False)
    sb3.set_title('mean deviation')
    sb3.grid()
    sb3.minorticks_on()
    sb3.set_xlabel('difference between in/out covariance values: out-in')
    
    log10_abs_vals = log10(abs(array(mean_percentage_diff_hist_vals)))
    sb4 = fig.add_subplot(224)
    sb4.hist(log10_abs_vals,bins=100,cumulative=False)
    sb4.set_title('mean percentage deviation')
    sb4.grid()
    sb4.minorticks_on()
    sb4.set_xlabel('log10, abs, fraction difference covariance values: '+\
        '((out-in)/in)')
    
    plt.show()




# code for testing actual procedure

# num_samples = 100
# num_otus = 25
# rho_mat = tril(uniform(-.1,.1,size=num_otus**2).reshape(num_otus,num_otus)) #take lower portion
# rho_mat = make_symmetric(rho_mat, trace_1=True)
# mu_mat = zeros(num_otus)
# methods_lol = [[spsd.lognorm,2,0,1]]*num_otus
# otu_table = cupola(num_samples=num_samples, rho_mat=rho_mat, mu_mat=mu_mat,
#     methods=methods_lol)
# corrs = make_symmetric(scipy_corr(otu_table, kendalltau))

# # code for plotting

# d = abs(corrs-rho_mat)
# matshow(d,cmap=plt.cm.jet)
# xmin,xmax = plt.xlim()
# ymin,ymax = plt.ylim()
# bars = linspace(xmin,xmax,rho_mat.shape[0]+1)
# plt.hlines(bars,ymin,ymax)
# plt.vlines(bars,xmin,xmax)
# plt.colorbar()
# plt.show()


# d = corrs
# matshow(d,cmap=plt.cm.jet)
# xmin,xmax = plt.xlim()
# ymin,ymax = plt.ylim()
# bars = linspace(xmin,xmax,rho_mat.shape[0]+1)
# plt.hlines(bars,ymin,ymax)
# plt.vlines(bars,xmin,xmax)
# plt.colorbar()
# plt.show()

