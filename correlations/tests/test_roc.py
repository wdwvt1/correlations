#!/usr/bin/env python

from __future__ import division 

__author__ = "Sophie Weiss"
__copyright__ = "Copyright 2013, Sophie Weiss"
__credits__ = ["Will Van Treuren, Sophie Weiss"]
__license__ = "GPL"
__version__ = ".9-dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"
__status__ = "Development"

from correlations.eval.roc import (roc_counter, roc)
from cogent.util.unit_test import (TestCase, main)
from numpy import array

class ROCTests(TestCase):
    '''Top level class for testing ROC calculations.'''
    
    def setUp(self):
        '''Create variables needed for all tests.'''
        pass # nothing needed by all functions

    def test_roc_counter(self):
        '''Test that TP, TT, IT are calculated correctly.'''
        # test basic 1d relationship of the form oX -> f(oX,oY)
        start = 0
        stop = 8
        dim = 1
        obs_edges = [('o0', 'o7'), ('o1', 'o13'), ('o0', 'o3'), ('o1', 'o0'), 
            ('o0', 'o1'), ('o1', 'o3'), ('o2', 'o3')]
        exp_TP, exp_IT, exp_TT = 3, 4, 7
        exp_TP, exp_IT, exp_TT = roc_counter(start, stop, dim, obs_edges)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)
        # test again with more complex relationship
        start = 60
        stop = 100
        dim = 4
        




        true_positive_edges = [\
        ('o0', 'o1'), ('o2', 'o3'), ('o4', 'o5'),
        ('o6', 'o7'), ('o8', 'o9'), ('o10', 'o11'), 
        ('o12', 'o13'), ('o14', 'o15'), ('o16', 'o17'), 
        ('o18', 'o19'), ('o1', 'o0'), ('o3', 'o2'), 
        ('o5', 'o4'), ('o7', 'o6'), ('o9', 'o8'), 
        ('o11', 'o10'), ('o13', 'o12'), ('o15', 'o14'), 
        ('o17', 'o16'), ('o19', 'o18')]
        edges = cr.edges

        true_stop = 20
        total_otus = 100

        totP = len(true_positive_edges)/2.
        exp_totP = 10
        unique_combinations = ((total_otus)*(total_otus-1))/2.
        totN = unique_combinations
        exp_unique_combinations = 4950
        exp_edges = [\
        ('o8', 'o14'), ('o8', 'o13'), ('o14', 'o9'),
        ('o44', 'o61'), ('o13', 'o9'), ('o14', 'o13'),
        ('o11', 'o14'), ('o11', 'o13'), ('o11', 'o9'),
        ('o12', 'o19'), ('o11', 'o8'), ('o19', 'o18'),
        ('o8', 'o9'), ('o83', 'o63')] 
        TP=0.
        FP=0.
        for x in edges:
            if x in true_positive_edges:
                TP+=1
            else:
                FP+=1

        FN = totP - TP
        TN = totN - FP
        sensitivity = TP/float(TP+FN)
        specificity = TN/float(TN+FP) 
        exp_TP = 2.0
        exp_FN = 8.0
        exp_FP = 12.0
        exp_TN = 4938.0
        exp_sensitivity = 0.2
        exp_specificity = 0.997575757576


        self.assertEqual(exp_totP, totP)
        self.assertEqual(exp_unique_combinations, unique_combinations)
        self.assertEqual(exp_edges, cr.edges)
        self.assertEqual(exp_TP, TP)
        self.assertEqual(exp_FN, FN)
        self.assertEqual(exp_FP, FP)
        self.assertEqual(exp_TN, TN)
        self.assertEqual(exp_sensitivity, sensitivity)
        self.assertEqual(exp_specificity, specificity)

if __name__ == '__main__':
    main()




CONET_LINES = [\
'OTU1\tOTU2\tinteractionType\tmethodname_score\tp-value\tq-value\tsignificance\n',
'o8\to14\tcopresence\t[sim_brownian=0.99, dist_kullbackleibler=0.079, correl_pearson=0.99, correl_spearman=0.99, dist_bray=0.09]\t0\t0\t100\n',
'o8\to13\tcopresence\t[dist_kullbackleibler=0.079, dist_bray=0.13, sim_brownian=0.98, correl_spearman=0.999, correl_pearson=0.96]\t0\t0\t100\n',
'o14\to9\tcopresence\t[dist_kullbackleibler=0.02, dist_bray=0.05, sim_brownian=0.997, correl_spearman=0.99, correl_pearson=0.996]\t0\t0\t100\n',
'o44\to61\tcopresence\t[dist_kullbackleibler=1.87, dist_bray=0.5, sim_brownian=0.63, correl_spearman=0.45, correl_pearson=0.75]\t1.75E-5\t0.003675\t2.4347\n',
'o13\to9\tcopresence\t[dist_kullbackleibler=0.035, dist_bray=0.068, sim_brownian=0.995, correl_spearman=-0.998, correl_pearson=0.995]\t0\t0\t100\n',
'o14\to13\tcopresence\t[dist_kullbackleibler=0.11, dist_bray=0.117, sim_brownian=0.986, correl_spearman=0.988, correl_pearson=0.985]\t0\t0\t100\n',
'o11\to14\tcopresence\t[dist_kullbackleibler=0.035, dist_bray=0.105, sim_brownian=0.988, correl_spearman=0.998, correl_pearson=0.987]\t0\t0\t100\n',
'o11\to13\tcopresence\t[dist_kullbackleibler=0.001, dist_bray=0.013, sim_brownian=0.9998, correl_spearman=0.9998, correl_pearson=0.9998]\t0\t0\t100\n',
'o11\to9\tcopresence\t[dist_kullbackleibler=0.022, dist_bray=0.055, sim_brownian=0.997, correl_spearman=-0.998, correl_pearson=0.997]\t0\t0\t100\n',
'o12\to19\tcopresence\t[dist_kullbackleibler=0.126, dist_bray=0.09, sim_brownian=0.817, correl_spearman=0.948, correl_pearson=0.641]\t3.79E-6\t8.91E-4\t3.05\n',
'o11\to8\tcopresence\t[dist_kullbackleibler=0.0738, dist_bray=0.122, sim_brownian=0.982, correl_spearman=0.999, correl_pearson=0.966]\t0\t0\t100\n',
'o19\to18\tcopresence\t[dist_kullbackleibler=0.0146, dist_bray=0.022, sim_brownian=0.97, correl_spearman=-0.902, correl_pearson=0.992]\t4.37E-12\t1.59E-9\t8.798\n',
'o8\to9\tcopresence\t[dist_kullbackleibler=0.058, dist_bray=0.102, sim_brownian=0.989, correl_spearman=0.998, correl_pearson=0.98]\t0\t0\t100\n',
'o83\to63\tcopresence\t[dist_kullbackleibler=0.035, dist_bray=0.695, sim_brownian=0.514, correl_spearman=0.2807, correl_pearson=0.313]\t1.16E-5\t0.00256885\t2.59\n']













