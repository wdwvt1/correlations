#!/usr/bin/env python
# file created 7/1/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Sophie Weiss, Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"

'''
Tests parsers.
'''

from cogent.util.unit_test import TestCase, main
from correlations.eval.parse import (CorrelationCalcs, CoNetResults, RMTResults,
    SparCCResults, LSAResults, triu_from_flattened) 
from biom.parse import parse_biom_table
from biom.table import table_factory
from numpy import array


# lists of lines are the input for each parser. here we define some lol's so 
# that we don't put them in setup since thats called before every function and
# is generally unnecessary. 

CONET_LINES = [\
'OTU1\tOTU2\tinteractionType\tmethodname_score\tp-value\tq-value\tsignificance\n',
'o3\to10\tcopresence\t[sim_brownian=0.59, dist_kullbackleibler=5.11, correl_pearson=0.6908, correl_spearman=0.1592, dist_bray=0.588]\t9.299E-4\t0.00295\t2.53\n',
'o1\to2\tmutualExclusion\t[dist_kullbackleibler=7.69, dist_bray=0.367, sim_brownian=0.03, correl_spearman=-0.1592, correl_pearson=-0.13]\t4.299E-3\t2.95\t3.13\n',
'o7\to10\tcopresence\t[dist_bray=0.81, sim_brownian=0.93, correl_spearman=0.6792, dist_kullbackleibler=0.69, correl_pearson=0.93]\t.0000462\t.0145\t6.6\n',
'o10\to25\tmutualExclusion\t[correl_spearman=-0.45, dist_kullbackleibler=0.969, dist_bray=0.2000334, sim_brownian=0.00193, correl_pearson=-0.43]\t.424\t.6\t.145\n',
'o3\to7\tcopresence\t[correl_pearson=0.13, correl_spearman=0.09113, dist_kullbackleibler=0.367, dist_bray=10.81, sim_brownian=0.0462]\t.362\t5.0145\t20.6\n']

RMT_LINES = [\
'OTU1\tOTU2\tType of interaction\tScore\tSignificance\n',
'o0\to188\tCorrelated\t0.466793678938798\t>=0.420\n',
'o0\to266\tCorrelated\t0.425935393560306\t>=0.420\n',
'o0\to458\tCorrelated\t0.45478037837313\t>=0.420\n',
'o1\to145\tCorrelated\t0.503077695845381\t>=0.420\n',
'o1\to353\tCorrelated\t-0.457869532664975\t>=0.420\n',
'o1\to453\tCorrelated\t0.470297558216443\t>=0.420\n']

SPARCC_PVAL_LINES = [\
'#OTU ID\to0\to1\to2\to3\to4\to5\to6\to7\to8\to9\to10',
'o0\t1\t0.724\t0.196\t0.032\t0.86\t0.506\t0.926\t0.126\t0.45\t0.772\t0.678',
'o1\t0.724\t1\t0.654\t0.368\t0.806\t0.644\t0.102\t0.918\t0.176\t0.35\t0.492',
'o2\t0.196\t0.654\t1\t0.194\t0.652\t0.298\t0.896\t0.856\t0.57\t0.592\t0.738',
'o3\t0.032\t0.368\t0.194\t1\t0.616\t0.764\t0.494\t0.574\t0.146\t0.418\t0.114',
'o4\t0.86\t0.806\t0.652\t0.616\t1\t0.106\t0.83\t0.552\t0.082\t0.618\t0.39',
'o5\t0.506\t0.644\t0.298\t0.764\t0.106\t1\t0.07\t0.608\t0.142\t0.006\t0.208',
'o6\t0.926\t0.102\t0.896\t0.494\t0.83\t0.07\t1\t0.636\t0.46\t0.78\t0.99',
'o7\t0.126\t0.918\t0.856\t0.574\t0.552\t0.608\t0.636\t1\t0.626\t0.206\t0.334',
'o8\t0.45\t0.176\t0.57\t0.146\t0.082\t0.142\t0.46\t0.626\t1\t0.432\t0.83',
'o9\t0.772\t0.35\t0.592\t0.418\t0.618\t0.006\t0.78\t0.206\t0.432\t1\t0.028',
'o10\t0.678\t0.492\t0.738\t0.114\t0.39\t0.208\t0.99\t0.334\t0.83\t0.028\t1']

SPARCC_CVAL_LINES = [\
'#OTU ID\to0\to1\to2\to3\to4\to5\to6\to7\to8\to9\to10',
'o0\t1\t0.042809762\t0.188155902\t0.325252076\t-0.024798583\t-0.097447861\t-0.013592412\t-0.2149632\t0.104918053\t-0.042625977\t-0.052934052',
'o1\t0.042809762\t1\t0.07306177\t-0.1336303\t0.036931222\t0.071362562\t-0.206405692\t-0.020485855\t-0.191335103\t-0.130767713\t0.100322858',
'o2\t0.188155902\t0.07306177\t1\t-0.180974904\t0.063132164\t-0.151249484\t0.01695292\t-0.027655694\t-0.080379352\t-0.080476622\t-0.045902642',
'o3\t0.325252076\t-0.1336303\t-0.180974904\t1\t0.081234874\t0.048957148\t0.110376013\t0.084476499\t0.213748379\t-0.119698252\t-0.224474732',
'o4\t-0.024798583\t0.036931222\t0.063132164\t0.081234874\t1\t0.220598434\t0.035346683\t0.08494071\t-0.246088506\t-0.067847301\t-0.115678103',
'o5\t-0.097447861\t0.071362562\t-0.151249484\t0.048957148\t0.220598434\t1\t-0.249954887\t0.072032574\t-0.223972682\t-0.375335961\t-0.185103934',
'o6\t-0.013592412\t-0.206405692\t0.01695292\t0.110376013\t0.035346683\t-0.249954887\t1\t-0.067077296\t-0.114213068\t-0.036058815\t0.002439413',
'o7\t-0.2149632\t-0.020485855\t-0.027655694\t0.084476499\t0.08494071\t0.072032574\t-0.067077296\t1\t0.072317055\t0.188817447\t0.135742427',
'o8\t0.104918053\t-0.191335103\t-0.080379352\t0.213748379\t-0.246088506\t-0.223972682\t-0.114213068\t0.072317055\t1\t0.107634175\t-0.033746075',
'o9\t-0.042625977\t-0.130767713\t-0.080476622\t-0.119698252\t-0.067847301\t-0.375335961\t-0.036058815\t0.188817447\t0.107634175\t1\t0.285243205',
'o10\t-0.052934052\t0.100322858\t-0.045902642\t-0.224474732\t-0.115678103\t-0.185103934\t0.002439413\t0.135742427\t-0.033746075\t0.285243205\t1']

LSA_LINES = [\
'X\tY\tLS\tlowCI\tupCI\tXs\tYs\tLen\tDelay\tP\tPCC\tPpcc\tSPCC\tPspcc\tDspcc\tSCC\tPscc\tSSCC\tPsscc\tDsscc\tQ\tQpcc\tQspcc\tQscc\tQsscc\tXi\tYi',
'o0\to0\t1\t1\t1\t1\t1\t50\t0\t0\t1\t0\t1\t0\t0\t1\t0\t1\t0\t0\t0\t0\t0\t0\t0\t1\t1',
'o0\to1\t0.238642\t0.238642\t0.238642\t23\t23\t18\t0\t0.358318\t-0.063483\t0.661401\t-0.063483\t0.661401\t0\t0.038752\t0.788822\t0.038752\t0.788822\t0\t1\t0.991798\t0.991798\t0.996933\t0.996933\t1\t2',
'o0\to2\t-0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3',
'o0\to3\t0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3',
'o1\to0\t0.292531\t0.292531\t0.292531\t15\t15\t32\t0\t0.153532\t0.031291\t0.829209\t0.031291\t0.829209\t0\t0.145162\t0.313482\t0.145162\t0.313482\t0\t1\t0.991965\t0.991965\t0.992018\t0.992018\t1\t4',
'o1\to1\t-0.204842\t-0.204842\t-0.204842\t1\t1\t27\t0\t0.558544\t-0.081485\t0.573741\t-0.081485\t0.573741\t0\t-0.139496\t0.3329\t-0.139496\t0.3329\t0\t1\t0.991798\t0.991798\t0.993358\t0.993358\t1\t5',
'o1\to2\t-0.142261\t-0.142261\t-0.142261\t1\t1\t50\t0\t0.931414\t-0.24658\t0.084303\t-0.24658\t0.084303\t0\t-0.144106\t0.317046\t-0.144106\t0.317046\t0\t1\t0.991798\t0.991798\t0.993064\t0.993064\t1\t6',
'o1\to3\t0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3',
'o2\to0\t-0.184914\t-0.184914\t-0.184914\t18\t18\t21\t0\t0.691456\t0.18021\t0.21045\t0.18021\t0.21045\t0\t0.000144\t0.99947\t0.000144\t0.99947\t0\t1\t0.991798\t0.991798\t1\t1\t1\t7',
'o2\to1\t-0.220149\t-0.220149\t-0.220149\t12\t12\t36\t0\t0.460606\t-0.182264\t0.205209\t-0.182264\t0.205209\t0\t-0.13383\t0.353059\t-0.13383\t0.353059\t0\t1\t0.991798\t0.991798\t0.993358\t0.993358\t1\t8',
'o2\to2\t0.249305\t0.249305\t0.249305\t1\t1\t22\t0\t0.31018\t-0.003987\t0.978079\t-0.003987\t0.978079\t0\t0.09916\t0.492111\t0.09916\t0.492111\t0\t1\t0.993039\t0.993039\t0.994862\t0.994862\t1\t9',
'o2\to3\t-0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3',
'o3\to0\t-0.141335\t-0.141335\t-0.141335\t29\t29\t19\t0\t0.936635\t-0.066955\t0.644091\t-0.066955\t0.644091\t0\t0.048643\t0.736606\t0.048643\t0.736606\t0\t1\t0.991798\t0.991798\t0.996672\t0.996672\t1\t10',
'o3\to1\t0.142968\t0.142968\t0.142968\t21\t21\t25\t0\t0.931414\t-0.000984\t0.994591\t-0.000984\t0.994591\t0\t-0.026074\t0.857087\t-0.026074\t0.857087\t0\t1\t0.993453\t0.993453\t0.998875\t0.998875\t1\t11',
'o3\to2\t0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3',
'o3\to3\t0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3']







# class CorrelationCalcs(TestCase):
#     '''Test base class correlation object.'''

#     def setUp(self):
#         '''Create variables needed for testing base class of CorrelationCalcs.
#         '''



class CoNetParserTests(TestCase):
    '''Top level class for testing CoNet Parser.'''
    
    def setUp(self):
        '''Create variables needed for the CoNet tests.'''
        self.CoNetResultsObj = CoNetResults(CONET_LINES)

    def test_self_properties(self):
        '''Test that properties like edges, otus, data, etc. are correct.'''

        exp_otu1 = ['o3', 'o1', 'o7', 'o10', 'o3']
        exp_otu2 = ['o10', 'o2', 'o10', 'o25', 'o7']
        exp_sig_otus = list(set(exp_otu1).union(set(exp_otu2)))
        exp_edges = zip(exp_otu1, exp_otu2)
        exp_interactions = array(['copresence', 'mutualExclusion', 'copresence',
            'mutualExclusion', 'copresence'])
        exp_pvals = array([9.299E-4, 4.299E-3, .0000462, .424, .362])
        exp_qvals = array([0.00295, 2.95, .0145, .6, 5.0145])
        exp_sigs = array([2.53, 3.13, 6.6,.145, 20.6])
        exp_scores = array([[  6.90800000e-01,   1.59200000e-01,   5.88000000e-01,
          5.11000000e+00,   5.90000000e-01],
       [ -1.30000000e-01,  -1.59200000e-01,   3.67000000e-01,
          7.69000000e+00,   3.00000000e-02],
       [  9.30000000e-01,   6.79200000e-01,   8.10000000e-01,
          6.90000000e-01,   9.30000000e-01],
       [ -4.30000000e-01,  -4.50000000e-01,   2.00033400e-01,
          9.69000000e-01,   1.93000000e-03],
       [  1.30000000e-01,   9.11300000e-02,   1.08100000e+01,
          3.67000000e-01,   4.62000000e-02]])
        exp_methods = ['correl_pearson', 'correl_spearman', 'dist_bray', 
            'dist_kullbackleibler', 'sim_brownian']


        self.assertEqual(exp_otu1, self.CoNetResultsObj.otu1)
        self.assertEqual(exp_otu2, self.CoNetResultsObj.otu2)
        self.assertEqual(set(exp_sig_otus), set(self.CoNetResultsObj.sig_otus))
        self.assertEqual(exp_edges, self.CoNetResultsObj.edges)
        self.assertEqual(exp_interactions, self.CoNetResultsObj.interactions)
        self.assertFloatEqual(exp_pvals, self.CoNetResultsObj.pvals)
        self.assertFloatEqual(exp_qvals, self.CoNetResultsObj.qvals)
        self.assertFloatEqual(exp_sigs, self.CoNetResultsObj.sigs)
        self.assertTrue((exp_scores == self.CoNetResultsObj.scores).all())
        self.assertEqual(exp_methods, self.CoNetResultsObj.methods)

class RMTParserTests(TestCase):
    '''Top level class for testing RMT Parser.'''

    def setUp(self):
        '''Create variables needed for the RMT tests.'''
        self.RMTResultsObj = RMTResults(RMT_LINES)

    def test_self_properties(self):
        '''Test that properties like edges, otus, data, etc. are correct.'''

        exp_otu1 = ['o0', 'o0', 'o0', 'o1', 'o1', 'o1']
        exp_otu2 = ['o188', 'o266', 'o458', 'o145', 'o353', 'o453']
        exp_sig_otus = list(set(exp_otu1).union(set(exp_otu2)))
        exp_edges = zip(exp_otu1, exp_otu2)
        exp_interactions = ['copresence', 'copresence', 'copresence', 
            'copresence', 'mutualExclusion', 'copresence']
        exp_scores = [0.466793678938798,0.425935393560306, 0.45478037837313,
            0.503077695845381, -0.457869532664975, 0.470297558216443]
        exp_sigs = [0.420, 0.420, 0.420, 0.420, 0.420, 0.420]

        self.assertEqual(exp_otu1, self.RMTResultsObj.otu1)
        self.assertEqual(exp_otu2, self.RMTResultsObj.otu2)
        self.assertEqual(set(exp_sig_otus), set(self.RMTResultsObj.sig_otus))
        self.assertEqual(exp_edges, self.RMTResultsObj.edges)
        self.assertEqual(exp_interactions, self.RMTResultsObj.interactions)
        self.assertFloatEqual(exp_sigs, self.RMTResultsObj.sigs)
        self.assertFloatEqual(exp_scores,self.RMTResultsObj.scores)

class SparCCParserTests(TestCase):
    '''Top level class for testing SparCC Parser.'''

    def setUp(self):
        '''Create variables needed for the SparCC tests.'''
        self.SparCCResultsObj = SparCCResults(SPARCC_PVAL_LINES, 
            SPARCC_CVAL_LINES, sig_lvl=.05)

    def test_self_properties(self):
        '''Test that properties like edges, otus, data, etc. are correct.'''

        exp_data = array([\
            [1, 0.724, 0.196, 0.032, 0.86, 0.506, 0.926, 0.126, 0.45, 0.772, 0.678],
            [0.724, 1, 0.654, 0.368, 0.806, 0.644, 0.102, 0.918, 0.176, 0.35, 0.492],
            [0.196, 0.654, 1, 0.194, 0.652, 0.298, 0.896, 0.856, 0.57, 0.592, 0.738],
            [0.032, 0.368, 0.194, 1, 0.616, 0.764, 0.494, 0.574, 0.146, 0.418, 0.114],
            [0.86, 0.806, 0.652, 0.616, 1, 0.106, 0.83, 0.552, 0.082, 0.618, 0.39],
            [0.506, 0.644, 0.298, 0.764, 0.106, 1, 0.07, 0.608, 0.142, 0.006, 0.208],
            [0.926, 0.102, 0.896, 0.494, 0.83, 0.07, 1, 0.636, 0.46, 0.78, 0.99],
            [0.126, 0.918, 0.856, 0.574, 0.552, 0.608, 0.636, 1, 0.626, 0.206, 0.334],
            [0.45, 0.176, 0.57, 0.146, 0.082, 0.142, 0.46, 0.626, 1, 0.432, 0.83],
            [0.772, 0.35, 0.592, 0.418, 0.618, 0.006, 0.78, 0.206, 0.432, 1, 0.028],
            [0.678, 0.492, 0.738, 0.114, 0.39, 0.208, 0.99, 0.334, 0.83, 0.028, 1]])


        exp_cdata = array([\
            [1, 0.042809762, 0.188155902, 0.325252076, -0.024798583, -0.097447861, -0.013592412, -0.2149632, 0.104918053, -0.042625977, -0.052934052],
            [0.042809762, 1, 0.07306177, -0.1336303, 0.036931222, 0.071362562, -0.206405692, -0.020485855, -0.191335103, -0.130767713, 0.100322858],
            [0.188155902, 0.07306177, 1, -0.180974904, 0.063132164, -0.151249484, 0.01695292, -0.027655694, -0.080379352, -0.080476622, -0.045902642],
            [0.325252076, -0.1336303, -0.180974904, 1, 0.081234874, 0.048957148, 0.110376013, 0.084476499, 0.213748379, -0.119698252, -0.224474732],
            [-0.024798583, 0.036931222, 0.063132164, 0.081234874, 1, 0.220598434, 0.035346683, 0.08494071, -0.246088506, -0.067847301, -0.115678103],
            [-0.097447861, 0.071362562, -0.151249484, 0.048957148, 0.220598434, 1, -0.249954887, 0.072032574, -0.223972682, -0.375335961, -0.185103934],
            [-0.013592412, -0.206405692, 0.01695292, 0.110376013, 0.035346683, -0.249954887, 1, -0.067077296, -0.114213068, -0.036058815, 0.002439413],
            [-0.2149632, -0.020485855, -0.027655694, 0.084476499, 0.08494071, 0.072032574, -0.067077296, 1, 0.072317055, 0.188817447, 0.135742427],
            [0.104918053, -0.191335103, -0.080379352, 0.213748379, -0.246088506, -0.223972682, -0.114213068, 0.072317055, 1, 0.107634175, -0.033746075],
            [-0.042625977, -0.130767713, -0.080476622, -0.119698252, -0.067847301, -0.375335961, -0.036058815, 0.188817447, 0.107634175, 1, 0.285243205],
            [-0.052934052, 0.100322858, -0.045902642, -0.224474732, -0.115678103, -0.185103934, 0.002439413, 0.135742427, -0.033746075, 0.285243205, 1]])

        exp_sig_edges = (array([0,5,9]), array([3,9,10]))
        exp_otu1 = ['o0', 'o5', 'o9']
        exp_otu2 = ['o3', 'o9', 'o10']
        exp_sig_otus = list(set(exp_otu1+exp_otu2))
        exp_edges = zip(exp_otu1, exp_otu2)
        exp_pvals = [0.032, 0.006, 0.028]
        exp_cvals = [0.325252076, -0.375335961, 0.285243205]
        exp_interactions = ['copresence', 'mutualExclusion', 'copresence']

        self.assertEqual(exp_otu1, self.SparCCResultsObj.otu1)
        self.assertEqual(exp_otu2, self.SparCCResultsObj.otu2)
        self.assertEqual(set(exp_sig_otus), set(self.SparCCResultsObj.sig_otus))
        self.assertEqual(exp_edges, self.SparCCResultsObj.edges)
        self.assertEqual(exp_interactions, self.SparCCResultsObj.interactions)
        self.assertFloatEqual(exp_pvals, self.SparCCResultsObj.pvals)
        self.assertFloatEqual(exp_cvals, self.SparCCResultsObj.cvals)

class LSAParserTests(TestCase):
    '''Top level class for testing LSA Parser.'''
    
    def setUp(self):
        '''Create variables needed for the LSA tests.'''
        # all_data is all relevant data from LSA lines. each function will pull
        # out what it needs to test things. 
        self.all_data = array([\
            [0.238642,0.358318,-0.063483,0.661401,-0.063483,0.661401,0.038752,0.788822,0.038752,0.788822],
            [-0.29343,0.153532,0.270707,0.057243,0.270707,0.057243,0.247059,0.083798,0.247059,0.083798],
            [0.29343,0.153532,0.270707,0.057243,0.270707,0.057243,0.247059,0.083798,0.247059,0.083798],
            [-0.142261,0.931414,-0.24658,0.084303,-0.24658,0.084303,-0.144106,0.317046,-0.144106,0.317046],
            [0.29343,0.153532,0.270707,0.057243,0.270707,0.057243,0.247059,0.083798,0.247059,0.083798],
            [-0.29343,0.153532,0.270707,0.057243,0.270707,0.057243,0.247059,0.083798,0.247059,0.083798]])

    def test_self_properties_ls(self):
        '''Test properties like edges, otus, data, when filterd by LS pvals.'''
        LSAResultsObj = LSAResults(LSA_LINES, filter='ls', sig_lvl=.2)
        
        exp_data = self.all_data[array([1,2,4,5])]
        exp_otu1 = ['o0', 'o0', 'o1', 'o2']
        exp_otu2 = ['o2', 'o3', 'o3', 'o3']
        exp_pvals = [0.153532, 0.153532, 0.153532, 0.153532]
        exp_interactions = ['mutualExclusion', 'copresence', 'copresence', 
            'mutualExclusion']
        exp_filter_ind = 9
        exp_value_filter_ind = 2
        exp_edges = zip(exp_otu1, exp_otu2)
        exp_sig_otus = list(set(exp_otu1).union(exp_otu2))

        self.assertTrue((exp_data == LSAResultsObj.data).all())
        self.assertEqual(exp_otu1, LSAResultsObj.otu1)
        self.assertEqual(exp_otu2, LSAResultsObj.otu2)
        self.assertEqual(set(exp_sig_otus), set(LSAResultsObj.sig_otus))
        self.assertEqual(exp_edges, LSAResultsObj.edges)
        self.assertEqual(exp_interactions, LSAResultsObj.interactions)
        self.assertEqual(exp_filter_ind, LSAResultsObj.filter_ind)
        self.assertEqual(exp_value_filter_ind, LSAResultsObj.value_filter_ind)


    # TO DO: 
    # Add tests for each of the other filter map indices, and function


# originaltable = array([[0, 0, 0, 1, 1], [3, 4, 1, 140, 1000], [5, 100, 0, 1, 2], [10, 1, 0, 0, 90], [0, 1000, 5, 30, 1]])
# tables = [originaltable]
# names = ['parser_testtable.biom']

# def make_ids(data):
#     sids = ['s%i' % i for i in range(data.shape[1])]
#     oids = ['o%i' % i for i in range(data.shape[0])]
#     return sids, oids

# for table, name in zip(tables,names):
#     sids, oids = make_ids(table)
#     bt = table_factory(table, sids, oids)

# # results fps
# resultstest_fp = '/Users/sophie/Documents/Knight/CoOccurance/CoNet_eval1-21/conet_testcode2.txt'

# #bt = parse_biom_table(bt)
# o = open(resultstest_fp)
# lines = o.readlines()
# o.close()
# cr = CoNetResults(lines)

# class CoNetResultsTests(TestCase):

#     def setUp(self):
#         '''sets up variables needed by other tests'''
#         self.cr = CoNetResults(lines)

#     def test_parseInputLines(self):
#         '''test that the parser works correctly'''
#         self.assertEqual(cr.otu1, ['o0', 'o0', 'o1'])
#         self.assertEqual(cr.otu2, ['o1', 'o2', 'o2'])
#         self.assertEqual(cr.interaction, ['copresence', 'mutualExclusion', 'mutualExclusion'])
#         self.assertEqual(cr.methods, ['correl_pearson', 'correl_spearman', 'dist_bray', 'dist_kullbackleibler', 'sim_brownian'])
#         self.assertEqual(cr.scores, array([[ 0.71  ,  0.91  ,  0.082 ,  0.07  ,  0.93  ],
#                                            [ -0.42  ,  0.9   ,  0.051 ,  0.015 ,  0.89  ],
#                                            [ -0.29  ,  0.91  ,  0.037 ,  0.0094,  0.92  ]]))
#         self.assertEqual(cr.pval, [0.0, 3.33e-16, 0.0])
#         self.assertEqual(cr.qval, [0.0, 1.06e-14, 0.0])
#         self.assertEqual(cr.sig, [100.0, 13.98, 100.0])

#     def test_connectionFraction(self, total_otus = 3):
#         '''test that connection fraction accurately calculated'''
#         total_sig_interactions = len(list(self.cr.interaction))
#         total_possible_interactions = total_otus*(total_otus-1)/2.
#         obs = total_sig_interactions/total_possible_interactions
#         exp = self.cr.connectionFraction(total_otus)
#         self.assertEqual(obs, exp)

#     def test_copresences(self):
#         '''test copresences are counted correctly'''
#         obs = 1.0
#         exp = self.cr.copresences()
#         self.assertEqual(obs, exp)

#     def test_exclusions(self):
#         '''test mutualExclusions are counted correctly'''
#         obs = 2.0
#         exp = self.cr.exclusions()
#         self.assertEqual(obs, exp)
        
#     def test_connectionAbundance(self):  
#         '''test abundance of connections per node accurately calculated'''
#         number_unique_otus = list(set(self.cr.otu1+self.cr.otu2))
#         expected_unique = ['o2', 'o1', 'o0']
#         obs_abs_of_cpn = array([0, 0, 3])
#         exp_abs_of_cpn = self.cr.connectionAbundance() 
#         self.assertEqual(number_unique_otus, expected_unique)
#         self.assertEqual(obs_abs_of_cpn, exp_abs_of_cpn)

#     def test_avgConnectivity(self):
#         '''test if return average number of connections per node (avg_cpn)'''
#         abs_cpn = self.cr.connectionAbundance()
#         total_abs_cpn = abs_cpn.sum().astype(float)
#         total_edge_origins = (abs_cpn*arange(len(abs_cpn))).sum()
#         obs_avg_cpn = total_edge_origins/total_abs_cpn
#         exp_avg_cpn = self.cr.avgConnectivity()
#         self.assertEqual(obs_avg_cpn, exp_avg_cpn)

#     def test_otuConnectivity(self): 
#         '''test that number of connections for each otu returned correctly'''
#         number_unique_otus = list(set(self.cr.otu1+self.cr.otu2))
#         unsorted_otuConnectivity = [(i,self.cr.otu1.count(i)+self.cr.otu2.count(i)) for i in number_unique_otus]
#         obs = sorted(unsorted_otuConnectivity, key=itemgetter(1), reverse=True)
#         exp = self.cr.otuConnectivity()
#         self.assertEqual(obs, exp)

    
#     def test_methodVals(self):
#         '''test if returns vectors of values for passed method.'''
#         methods = list(self.cr.methods)
#         exp_Vals = list(cr.scores.T)
#         for i, method in enumerate(self.cr.methods):
#             obs_vals = self.cr.methodVals(method)
#             exp_vals = exp_Vals[i]
#             self.assertEqual(obs_vals, exp_vals)
      
#     def test_extract_from_rho(self):
#         '''test that correct rho values extracted from rho matrix'''
#         rho = array([[ 1  ,  -0.9  ,  -0.01 ,  0.999  ,  0.5  ],
#                     [ -0.9  ,  1   ,  100 ,  0.00001 ,  0.3  ],
#                     [ -0.01  ,  100 ,  1 ,  0.8 ,  0.03  ],
#                     [ 0.999  ,  0.00001  ,  0.8 ,  1 ,  -0.65  ],
#                     [ 0.5  ,  0.3  ,  0.03 ,  -0.65,  1  ]])
#         for i in range(len(rho)):
#             for j in range(len(rho[i])):
#                 row = 'o%d' %(i)
#                 column = 'o%d' %(j)
#                 edge = [(row,column)]
#                 obs = extract_from_rho(rho, edge)
#                 exp = [rho[i][j]]
#                 self.assertEqual(obs,exp)

#     def test_node_stats(self):
#         '''test that the correct node statistics are returned'''
#         exp_node_stats = {'sig_otu_mean': 83.86666666666666, 'all_otu_std': 269.02371642663775, 'all_otu_sparsity': 0.28000000000000003, 'non_sig_otu_sparsity': 0.29999999999999999, 'all_otu_mean': 95.799999999999997, 'non_sig_otu_mean': 113.7, 'non_sig_otu_std': 296.62267276794597, 'sig_otu_sparsity': 0.26666666666666666, 'sig_otu_std': 248.21411903614361}
#         self.assertEqual(node_stats(cr.sig_otus,bt), exp_node_stats)
    
#     def test_ga_edge_even_odd(self):
#         '''test if Return: True if all edges are between even and odd OTUs (gene1 and gene2)'''
#         even_odd = [('o1', 'o2')]
#         even_even = [('o2', 'o4')]
#         odd_odd = [('o1', 'o3')]
#         self_self = [('o1', 'o1')]
#         self.assertEqual(ga_edge_even_odd(even_odd), True)
#         self.assertEqual(ga_edge_even_odd(even_even), False)
#         self.assertEqual(ga_edge_even_odd(odd_odd), False)
#         self.assertEqual(ga_edge_even_odd(self_self), False)

#     def test_null_sig_node_locs(self):
#         '''test that the dataset locations of sig detected OTUs are correct'''
#         num_nodes = [3]
#         sig_nodes = cr.sig_otus
#         locs = null_sig_node_locs(num_nodes, sig_nodes)
#         exp_locs = array([0, 0, 0])
#         self.assertEqual(locs, exp_locs)

#     def test_null_edge_directionality(self):
#         '''tests return of matrix of number of significant OTUs shared between distributions'''
#         num_nodes = [3]
#         directionality = null_edge_directionality(cr.otu1, cr.otu2, num_nodes)
#         exp_directionality = array([[ 3.]])

#     def test_eco_d2_counter(self):
#         '''test correct class numbers and categories returned for d2 and d1 relationships'''
#         output_edges = cr.edges
#         output_interactions = cr.interaction
#         start = 0
#         stop = 4
#         d2 = eco_d2_counter(start, stop, output_edges, output_interactions)
#         exp_d2 = (1, 2, 1.3333333333333333, 1, 0, 0, 2)
#         self.assertEqual(d2, exp_d2)

#     def test_eco_d1_counter(self):
#         '''test correct class numbers and categories returned for d1 relationships'''
#         output_edges = cr.edges
#         output_interactions = cr.interaction
#         start = 0
#         stop = 4
#         d1 = eco_d1_counter(start, stop, output_edges, output_interactions)
#         exp_d1 = (1, 2.0, 1, 0)
#         self.assertEqual(d1, exp_d1)
    





















if __name__ == '__main__':
    main()