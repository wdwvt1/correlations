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

from numpy import linspace, array
import matplotlib.pyplot as plt
from correlations.eval.result_eval import interacting_edges
from biom.parse import parse_biom_table
from correlations.eval.parse import SparCCResults

def roc_counter(start, stop, dim, obs_edges):
    '''Count the TP and FP based on observed edges and knowledge of true edges.

    Inputs:
     start - int, the inclusive lower bound of the numerical part of the OTUs 
     that are correlated (e.g. 7 for o7). 
     stop - int, the exclusive upper bound of the numerical part of the OTUs 
     that are correalted (e.g. 7 for o7).
     dim - int, number of OTUs on the LHS of the rule. 
     obs_edges - list of tuples, observed OTU edges (e.g. [('o1','o19'),...]
    '''
    (total_detected, cis_edges, cis_cps, cis_mes, trans_edges, trans_cps,
        trans_mes) = interacting_edges(start, stop, dim, obs_edges, 
        interactions=['none']*len(obs_edges)) #interactions is hackish, but we don't
    # need if for this function
    TP = total_detected
    TT = len(obs_edges)
    IT = (stop-start)/(dim+1.)
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

def plot_roc(specificity_pts, sensitivity_pts, pvals):
    '''plots a ROC curve and saves if save is a path'''
    fig = plt.figure()
    plt.plot(1-specificity_pts, sensitivity_pts, 'g-', 
        label='ROC Points', alpha = .5)
    plt.plot(1-specificity_pts, sensitivity_pts, 'gs', markersize=10,alpha = .5)
    plt.plot(linspace(0,1,1000),linspace(0,1,1000), 'k-', markersize=1, 
        label='Baseline', alpha = .5)
    [plt.text(1- specificity_pts[i], sensitivity_pts[i], map(str, pvals)[i]) for i in range(len(pvals))]
    plt.legend()
    plt.grid()
    plt.title('ROC Curve')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.show()



