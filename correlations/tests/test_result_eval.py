#!/usr/bin/env python
# file created 7/1/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren, Sophie Weiss"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"

'''
Test functions used for evaluating the results of the tools on teh syntehtic 
data.
'''

from cogent.util.unit_test import TestCase, main
from correlations.eval.result_eval import (interacting_edges)
from biom.parse import parse_biom_table
from biom.table import table_factory
from numpy import array


class ResultEvaluationFunctions(TestCase):
    '''Top level class for testing result evaluation functions.'''
    
    def setUp(self):
        '''No variables needed by all tests.'''
        pass

    def test_interacting_edges(self):
        '''Test that interacting edges are calculated correctly.'''    
        edges = [\
            ('o0','o2'),
            ('o1','o2'),
            ('o0','o1'),
            ('o3','o2'),
            ('o3','o5'),
            ('o11','o25'),
            ('o0','o6'),
            ('o47','o50'),
            ('o6','o7')]
        interactions = [\
            'mutualExclusion',
            'mutualExclusion',
            'mutualExclusion',
            'mutualExclusion',
            'copresence',
            'copresence',
            'copresence',
            'copresence',
            'copresence']

        # test with 1d relationships
        start = 0
        stop = 10
        dim = 1

        exp_total_detected = 3
        exp_cis_edges = 0
        exp_cis_cps = 0
        exp_cis_mes = 0
        exp_trans_edges = 3
        exp_trans_cps = 1
        exp_trans_mes = 2

        self.assertEqual((exp_total_detected, exp_cis_edges, exp_cis_cps, 
            exp_cis_mes, exp_trans_edges, exp_trans_cps, exp_trans_mes),
            interacting_edges(start, stop, dim, edges, interactions))

        # test with 1d relationships, offset from last 
        start = 1
        stop = 5
        dim = 1

        exp_total_detected = 1
        exp_cis_edges = 0
        exp_cis_cps = 0
        exp_cis_mes = 0
        exp_trans_edges = 1
        exp_trans_cps = 0
        exp_trans_mes = 1

        self.assertEqual((exp_total_detected, exp_cis_edges, exp_cis_cps, 
            exp_cis_mes, exp_trans_edges, exp_trans_cps, exp_trans_mes),
            interacting_edges(start, stop, dim, edges, interactions))
        
        # test case with 2d relationships. inverted direction in some cases.
        start = 0
        stop = 6
        dim = 2    
        
        exp_total_detected = 4
        exp_cis_edges = 1
        exp_cis_cps = 0
        exp_cis_mes = 1
        exp_trans_edges = 3
        exp_trans_cps = 1
        exp_trans_mes = 2

        self.assertEqual((exp_total_detected, exp_cis_edges, exp_cis_cps, 
            exp_cis_mes, exp_trans_edges, exp_trans_cps, exp_trans_mes),
            interacting_edges(start, stop, dim, edges, interactions))
        
        # adjust start to make sure it handles non zero starts correctly
        start = 3
        stop = 9
        dim = 2

        exp_total_detected = 2
        exp_cis_edges = 1
        exp_cis_cps = 1
        exp_cis_mes = 0
        exp_trans_edges = 1
        exp_trans_cps = 1
        exp_trans_mes = 0

        self.assertEqual((exp_total_detected, exp_cis_edges, exp_cis_cps, 
            exp_cis_mes, exp_trans_edges, exp_trans_cps, exp_trans_mes),
            interacting_edges(start, stop, dim, edges, interactions))

        # test with 4d relationships
        start = 0
        stop = 50
        dim = 5

        exp_total_detected = 6
        exp_cis_edges = 5
        exp_cis_cps = 1
        exp_cis_mes = 4
        exp_trans_edges = 1
        exp_trans_cps = 1
        exp_trans_mes = 0

        self.assertEqual((exp_total_detected, exp_cis_edges, exp_cis_cps, 
            exp_cis_mes, exp_trans_edges, exp_trans_cps, exp_trans_mes),
            interacting_edges(start, stop, dim, edges, interactions))



if __name__ == '__main__':
    main()