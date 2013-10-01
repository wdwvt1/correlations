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

from correlations.eval.roc import (roc_edge_count, roc)
from cogent.util.unit_test import (TestCase, main)
from numpy import array

class ROCTests(TestCase):
    '''Top level class for testing ROC calculations.'''
    
    def setUp(self):
        '''Create variables needed for all tests.'''
        pass # nothing needed by all functions

    def test_roc_ege_count(self):
        '''Test that TP, TT, IT are calculated correctly.'''
        # test basic 1d relationship of the form oX -> f(oX,oY)
        start = 0
        stop = 8
        lhs_dim = 1
        rhs_dim = 1
        obs_edges = [('o0', 'o7'), ('o1', 'o13'), ('o0', 'o3'), ('o1', 'o0'), 
            ('o0', 'o1'), ('o1', 'o3'), ('o2', 'o3')]
        true_edge_type = 'any'
        exp_TP, exp_IT, exp_TT = 3, 4, 7
        obs_TP, obs_IT, obs_TT = roc_edge_count(start, stop, lhs_dim, rhs_dim,
            obs_edges, true_edge_type)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)
        # test again with more complex relationship
        start = 60
        stop = 100
        lhs_dim = 4
        rhs_dim = 1
        obs_edges = [('o1','o65'), ('o64','o60'), ('o65','o64'), ('o71','o72'),
            ('o80','o86'), ('o110','o111')]
        true_edge_type = 'lhs_lhs'
        exp_TP, exp_IT, exp_TT = 1, 48, 6
        obs_TP, obs_IT, obs_TT = roc_edge_count(start, stop, lhs_dim, rhs_dim,
            obs_edges, true_edge_type)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)
        # test cases that might give it trouble
        start = 10
        stop = 40
        lhs_dim = 2
        rhs_dim = 1
        obs_edges = [('o10','o11'), ('o11','o12'), ('o10','o12'), ('o13','o12'),
            ('o11','o13'), ('o13','o10')]
        true_edge_type = 'lhs_rhs'
        exp_TP, exp_IT, exp_TT = 2, 20, 6
        obs_TP, obs_IT, obs_TT = roc_edge_count(start, stop, lhs_dim, rhs_dim,
            obs_edges, true_edge_type)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)
        true_edge_type = 'rhs_rhs'
        exp_TP, exp_IT, exp_TT = 0, 0, 6
        obs_TP, obs_IT, obs_TT = roc_edge_count(start, stop, lhs_dim, rhs_dim,
            obs_edges, true_edge_type)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)
        true_edge_type = 'lhs_lhs'
        exp_TP, exp_IT, exp_TT = 1, 10, 6
        obs_TP, obs_IT, obs_TT = roc_edge_count(start, stop, lhs_dim, rhs_dim,
            obs_edges, true_edge_type)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)
        # test another strange case
        start = 13
        stop = 113
        lhs_dim = 8
        rhs_dim = 2
        obs_edges = [('o10','o11'), ('o11','o12'), ('o10','o12'), ('o13','o12'),
            ('o11','o13'), ('o13','o10'), ('o13', 'o20'), ('o14','o13'), 
            ('o21','o22'),('o32','o31')]
        true_edge_type = 'rhs_rhs'
        exp_TP, exp_IT, exp_TT = 2, 10, 10
        obs_TP, obs_IT, obs_TT = roc_edge_count(start, stop, lhs_dim, rhs_dim,
            obs_edges, true_edge_type)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)
        true_edge_type = 'lhs_rhs'
        exp_TP, exp_IT, exp_TT = 0, 160, 10
        obs_TP, obs_IT, obs_TT = roc_edge_count(start, stop, lhs_dim, rhs_dim,
            obs_edges, true_edge_type)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)
        true_edge_type = 'any'
        exp_TP, exp_IT, exp_TT = 4, 450, 10
        obs_TP, obs_IT, obs_TT = roc_edge_count(start, stop, lhs_dim, rhs_dim,
            obs_edges, true_edge_type)
        self.assertEqual(exp_TP, obs_TP)
        self.assertEqual(exp_IT, obs_IT)
        self.assertEqual(exp_TT, obs_TT)

    def test_roc(self):
        '''Test that the values needed for a ROC curve are calculated correctly.
        '''
        TP, IT, TT, E = 5., 10., 20., 100.
        TN = E - (IT-TP+TT)
        exp_sens, exp_spec = TP/IT, TN/(TT-TP+TN)
        obs_sens, obs_spec = roc(TP, IT, TT, E)
        self.assertEqual(exp_sens, obs_sens)
        self.assertEqual(exp_spec, obs_spec)


if __name__ == '__main__':
    main()











