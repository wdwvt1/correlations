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
Test functions used for evaluating the results of the tools on the syntehtic 
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


class Timeseries_definitions_test(TestCase):

    def setUp(self):
        '''sets up variables needed by many tests'''
        self.freq = [1, 2, 3]
        self.amp = [100, 50, 25]
        self.phase = [0, .25*pi, .5*pi]
        self.noise = [0, .25, .5]
        self.adj = [[subsample_otu_evenly, .5], [subsample_otu_zero, .5, .3], 
            [subsample_otu_zero, .5, .75]]
        self.q = cube_d5_indices(freq, amp, phase, noise, adj)

    def test_timeseries_indices(self):
        '''test that the parser works correctly for combinations actually used in timeseries tables round 2
        freq 1 otus: 0-80, freq 2 otus: 81-161, freq 3 otus: 162-242
        amp 1 otus: 0-26, 81-107, 162-188; amp .5 otus: 27-53, 108-134, 189-215; amp .25 otus: 54-80, 135-161, 216-242
        phase 0 otus: 0-8, 81-89, 162-170, 27-35, 108-116, 189-197, 54-62, 135-143, 216-224
        phase .25pi otus: 9-17, 90-98, 171-179, 36-44, 117-125, 198-206, 63-71, 144-152, 225-233
        phase .5pi otus: 18-26, 99-107, 180-188, 45-53, 126-134, 207-215, 72-80, 153-161, 234-242 
        noise 0 otus: 0-2, 9-11, 36-34, 81-83, 90-92, 117-119, 162-164, 171-173, 198-200, 63-65, 144-146, 225-227, 27-29, 108-110, 189-191, 54-56, 135-137, 216-218,
        18-20, 99-101, 180-182, 45-47, 126-128, 207-209, 72-74, 153-155, 234-236
        noise .25 otus: 3-5, 12-14, 36-38, 84-86, 93-95, 120-122, 165-167, 174-176, 201-203, 66-68, 147-149, 228-230, 30-32, 111-113, 193-195, 57-59, 138-140, 219-221,
        21-23, 102-104, 183-185, 48-50, 129-131, 210-212, 75-77, 156-158, 237-239
        noise .5 otus: 6-8, 15-17, 39-41, 87-89, 96-98, 123-125, 168-170, 177-179, 204-206, 69-71, 150-152, 231-233, 33-35, 114-116, 196-198, 60-62, 141-143, 222-224, 
        24-26, 105-107, 186-188, 51-53, 132-134, 213-215, 78-80, 159-161, 240-242
        adj(1) otus: 0, 3, 6, 9, 12, 15 ... 237, 240
        adj(2) otus: 1, 4, 7, 10, ... 238, 241
        adj(3) otus: 2, 5, 8, 11, ... 239, 242 
        '''
        exp_indices = array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
        [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161],
        [162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242], 
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188], 
        [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215], 
        [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242], 
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 28, 29, 30, 31, 32, 33, 34, 35, 54, 55, 56, 57, 58, 59, 60, 61, 62, 81, 82, 83, 84, 85, 86, 87, 88, 89, 108, 109, 110, 111, 112, 113, 114, 115, 116, 135, 136, 137, 138, 139, 140, 141, 142, 143, 162, 163, 164, 165, 166, 167, 168, 169, 170, 189, 190, 191, 192, 193, 194, 195, 196, 197, 216, 217, 218, 219, 220, 221, 222, 223, 224], 
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 42, 43, 44, 63, 64, 65, 66, 67, 68, 69, 70, 71, 90, 91, 92, 93, 94, 95, 96, 97, 98, 117, 118, 119, 120, 121, 122, 123, 124, 125, 144, 145, 146, 147, 148, 149, 150, 151, 152, 171, 172, 173, 174, 175, 176, 177, 178, 179, 198, 199, 200, 201, 202, 203, 204, 205, 206, 225, 226, 227, 228, 229, 230, 231, 232, 233], 
        [18, 19, 20, 21, 22, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50, 51, 52, 53, 72, 73, 74, 75, 76, 77, 78, 79, 80, 99, 100, 101, 102, 103, 104, 105, 106, 107, 126, 127, 128, 129, 130, 131, 132, 133, 134, 153, 154, 155, 156, 157, 158, 159, 160, 161, 180, 181, 182, 183, 184, 185, 186, 187, 188, 207, 208, 209, 210, 211, 212, 213, 214, 215, 234, 235, 236, 237, 238, 239, 240, 241, 242], 
        [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 45, 46, 47, 54, 55, 56, 63, 64, 65, 72, 73, 74, 81, 82, 83, 90, 91, 92, 99, 100, 101, 108, 109, 110, 117, 118, 119, 126, 127, 128, 135, 136, 137, 144, 145, 146, 153, 154, 155, 162, 163, 164, 171, 172, 173, 180, 181, 182, 189, 190, 191, 198, 199, 200, 207, 208, 209, 216, 217, 218, 225, 226, 227, 234, 235, 236], 
        [3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 39, 40, 41, 48, 49, 50, 57, 58, 59, 66, 67, 68, 75, 76, 77, 84, 85, 86, 93, 94, 95, 102, 103, 104, 111, 112, 113, 120, 121, 122, 129, 130, 131, 138, 139, 140, 147, 148, 149, 156, 157, 158, 165, 166, 167, 174, 175, 176, 183, 184, 185, 192, 193, 194, 201, 202, 203, 210, 211, 212, 219, 220, 221, 228, 229, 230, 237, 238, 239], 
        [6, 7, 8, 15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44, 51, 52, 53, 60, 61, 62, 69, 70, 71, 78, 79, 80, 87, 88, 89, 96, 97, 98, 105, 106, 107, 114, 115, 116, 123, 124, 125, 132, 133, 134, 141, 142, 143, 150, 151, 152, 159, 160, 161, 168, 169, 170, 177, 178, 179, 186, 187, 188, 195, 196, 197, 204, 205, 206, 213, 214, 215, 222, 223, 224, 231, 232, 233, 240, 241, 242], 
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240], 
        [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241], 
        [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242]])
        indices = timeseries_indices(self.freq, self.amp, self.phase, self.noise, self.adj, self.q)
        self.assertEqual(exp_indices, indices)

    def test_null_sig_node_locs_timeseries(self):
        '''Return locations of sig OTUs in list of lists of indices for each of the
        parameter values for each of the d5 dimensions (here the dimensions are 
        freq, amp, phase, noise, adj, each with three parameter values following above)'''
        sig_nodes = ['o1','o2', 'o240', 'o280']
        locs = null_sig_node_locs_timeseries(indices, sig_nodes)
        exp_locs = [0, 0, 2, 3, 3, 5, 6, 6, 8, 9, 9, 11, 12, 13, 14]
        self.assertEqual(exp_locs, locs)

    def test_null_edge_directionality_timeseries(self):
        exp_edge_dir_mat = array([ 
        [ 2. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 2. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 2. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 1. , 1. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 2. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0.],
        [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.]])
        indices = timeseries_indices(self.freq, self.amp, self.phase, self.noise, self.adj, self.q)
        num_nodes = [81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81]
        otu1 = ['o1', 'o1', 'o1']
        otu2 = ['o3', 'o1', 'o240']
        edge_dir_mat = null_edge_directionality_timeseries(otu1, otu2, num_nodes, indices)
        self.assertEqual(exp_edge_dir_mat, edge_dir_mat)



if __name__ == '__main__':
    main()