#!/usr/bin/env python
from __future__ import division 

__author__ = "Will Van Treuren, Sophie Weiss"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren, Sophie Weiss"]
__license__ = "GPL"
__version__ = ".9-dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"
__status__ = "Development"

from numpy import linspace, array, logical_xor
import matplotlib.pyplot as plt
from correlations.eval.result_eval import interacting_edges
from biom.parse import parse_biom_table
from correlations.eval.parse import SparCCResults

def roc_edge_count(start, stop, lhs_dim, rhs_dim, obs_edges, true_edge_type):
    '''Similar functionality to interacting_edges, but specifically for ROCs.

    This code counts the number of edges that are true positives (TP), truly 
    correlated (IT), and are positive (whether true positive or false positive, 
    TT). 

    Inputs:
     start - int, the inclusive lower bound of the numerical part of the OTUs 
     that are correlated (e.g. 7 for o7). 
     stop - int, the exclusive upper bound of the numerical part of the OTUs 
     that are correalted (e.g. 7 for o7).
     lhs_dim - int, number of OTUs on the LHS of the rule. 
     obs_edges - list of tuples, observed OTU edges (e.g. [('o1','o19'),...]
     true_edge_type - str, one of ['any','lhs_lhs','lhs_rhs','rhs_rhs']. 
     determines what counts as true edges, i.e. LHS otu to LHS otu, LHS to
     RHS (or RHS to LHS) otu, RHS to RHS otu, or any.
    '''
    if start/(lhs_dim+rhs_dim) != start//(lhs_dim+rhs_dim):
        print('Start index is not evenly divisible by the total dimension.'+\
            ' Normalizing to prevent errors.')
        tmp_start = 0 #start - start 
        tmp_stop = stop - start
        def _f(x):
            return (int(x[0][1:])-start, int(x[1][1:])-start)
    else:
        # evenly divisible means we don't need to normalize
        def _f(x):
            return (int(x[0][1:]), int(x[1][1:]))
        tmp_start = start
        tmp_stop = stop
    TP = 0
    int_edges = map(_f, obs_edges)
    for edge in int_edges:
        o1, o2 = edge
        if tmp_start <= o1 < tmp_stop and tmp_start <= o2 < tmp_stop:
            # calculate integer parts of o1,o2
            i1 = o1//(lhs_dim+rhs_dim)
            i2 = o2//(lhs_dim+rhs_dim)
            if i1 == i2: #same relationship
                r1 = o1%(lhs_dim+rhs_dim)
                r2 = o2%(lhs_dim+rhs_dim)
                if true_edge_type=='lhs_lhs': #both otus in LHS 
                    if r1 < lhs_dim and r2 < lhs_dim: 
                        TP+=1
                elif true_edge_type=='rhs_rhs': #both otus in RHS 
                    if r1 >= lhs_dim and r2 >= lhs_dim: 
                        TP+=1
                elif true_edge_type=='lhs_rhs': #one otu in LHS, one in RHS
                    if logical_xor(r1 < lhs_dim, r2 < lhs_dim):
                        TP+=1
                elif true_edge_type=='any': #either of the above is allowed
                    if any([r1 < lhs_dim and r2 < lhs_dim,
                            r1 >= lhs_dim and r2 >= lhs_dim,
                            logical_xor(r1 < lhs_dim, r2 < lhs_dim)]):
                        TP+=1
    # calculate the number of truly correlated things
    groups = (stop-start)/float(lhs_dim+rhs_dim) #should be int, but jic.
    if true_edge_type=='lhs_lhs':
        IT = groups*lhs_dim*(lhs_dim-1)/2.
    if true_edge_type=='rhs_rhs':
        IT = groups*rhs_dim*(rhs_dim-1)/2.
    if true_edge_type=='lhs_rhs':
        IT = groups*lhs_dim*rhs_dim
    if true_edge_type=='any':
        IT = groups*(lhs_dim+rhs_dim)*(lhs_dim+rhs_dim-1)/2.
    TT = len(obs_edges)
    return TP, IT, TT

def roc(TP, IT, TT, E):
    '''Calculates the data necessary for making ROC curves
    
    Constructs the following table, and from it sensitivity and specificty.
                 Is True
    Tests True   Yes    No
           Yes   TP     FP    TT
            No   FN     TN    TF
                 IT     IF    2*E

    TP, FP, FN, TN = true pos, false pos, false neg, true neg
    IT, IF, TT, TF = is true, is false, tests true, tests false
    E = total number of items tested
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    '''
    FN = IT - TP 
    FP = TT - TP
    TN = E - (TP + FN + FP)
    sens = TP/float(TP+FN)
    spec = TN/float(TN+FP) 
    return sens, spec

def plot_roc(specificity_pts, sensitivity_pts, pvals, lbl, c, pval_labels=False,
    ms=4):
    '''Plot a ROC curve.

    Inputs:
     specificty_pts - list/arr of numeric values produced by roc.
     sensitivity_pts - list/arr of numeric values produced by roc.
     pvals - list/arr of numeric values used in calculations. must be in the 
     same order as the sensitivity and specificity pts that were calculated from
     them. 
     lbl - str, the lbl you want for the plotted data series. 
     c - str, matplotlib color for points and line for this data series. 
     pvals_labels - boolean, if true will plot the pvals above their 
     corresponding points.
     ms - int, size of the pts.
    '''
    plt.plot(1-specificity_pts, sensitivity_pts, color=c, markerfacecolor=c, 
        markersize=ms,  marker='s', linewidth=1.0, label=lbl, alpha = .5)
    if pval_labels:
        [plt.text(1- specificity_pts[i], sensitivity_pts[i], \
            map(str, pvals)[i]) for i in range(len(pvals))]

def finish_roc_plot(title):
    '''Finalize ROC curves to make them more attractive/informative.'''
    plt.title(title)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.grid(True)
    plt.plot([0,1],[0,1],'k-', label='Baseline', alpha = .5)
    plt.legend(loc=4)
    plt.xlim(-.02,1.02)
    plt.ylim(-.02,1.02)
    plt.show()



