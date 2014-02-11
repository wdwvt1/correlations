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
    SparCCResults, LSAResults, NaiveResults, BrayCurtisResults, MICResults,
    triu_from_flattened) 
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

LSA_LINES_REDUNDANT = [\
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

LSA_LINES_UNIQUE = [\
'X\tY\tLS\tlowCI\tupCI\tXs\tYs\tLen\tDelay\tP\tPCC\tPpcc\tSPCC\tPspcc\tDspcc\tSCC\tPscc\tSSCC\tPsscc\tDsscc\tQ\tQpcc\tQspcc\tQscc\tQsscc\tXi\tYi',
'o0\to1\t0.238642\t0.238642\t0.238642\t23\t23\t18\t0\t0.358318\t-0.063483\t0.661401\t-0.063483\t0.661401\t0\t0.038752\t0.788822\t0.038752\t0.788822\t0\t1\t0.991798\t0.991798\t0.996933\t0.996933\t1\t2',
'o0\to2\t-0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3',
'o0\to3\t0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3',
'o1\to2\t-0.142261\t-0.142261\t-0.142261\t1\t1\t50\t0\t0.931414\t-0.24658\t0.084303\t-0.24658\t0.084303\t0\t-0.144106\t0.317046\t-0.144106\t0.317046\t0\t1\t0.991798\t0.991798\t0.993064\t0.993064\t1\t6',
'o1\to3\t0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3',
'o2\to3\t-0.29343\t0.29343\t0.29343\t2\t2\t48\t0\t0.153532\t0.270707\t0.057243\t0.270707\t0.057243\t0\t0.247059\t0.083798\t0.247059\t0.083798\t0\t1\t0.98568\t0.98568\t0.982603\t0.982603\t1\t3']

NAIVE_PVAL_LINES = [\
'#OTU ID\to1\to2\to3\to4\to5\to6\to7\to8\to9\to10\n',
'o1\t0.0\t0.206434225923\t0.622991821571\t0.260517055327\t0.676050675514\t0.00521400441251\t0.408067313969\t0.946415552616\t0.946909870319\t0.766811229219\n',
'o2\t0.0\t0.0\t0.864396799495\t0.720947558122\t0.936234977663\t0.955233632835\t0.38334045641\t0.665532585852\t0.262485596226\t0.815959028953\n',
'o3\t0.0\t0.0\t0.0\t0.0878748237431\t0.321231600405\t0.981083070404\t0.896267086014\t0.186545651526\t0.847738459608\t0.307348720519\n',
'o4\t0.0\t0.0\t0.0\t0.0\t0.301054456679\t0.192261343941\t0.726585535447\t0.653247092948\t0.748275657427\t0.218444842777\n',
'o5\t0.0\t0.0\t0.0\t0.0\t0.0\t0.888162448983\t0.4543059837\t0.839730912896\t0.104990268968\t0.00795219600018\n',
'o6\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.283874533257\t0.637920551021\t0.655114604327\t0.75567723783\n',
'o7\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.748113327024\t0.0172690561322\t0.864886524572\n',
'o8\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.766117284199\t0.617472511472\n',
'o9\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.800251026852\n',
'o10\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0']

NAIVE_CVAL_LINES = [\
'#OTU ID\to1\to2\to3\to4\to5\to6\to7\to8\to9\to10\n',
'o1\t0.0\t0.444265009199\t-0.183703495193\t-0.40136437161\t0.156635423487\t0.784068817317\t-0.302882153153\t0.0253970414374\t-0.0251625044885\t-0.111617772775\n',
'o2\t0.0\t0.0\t0.0644592524687\t-0.134191846009\t0.0302290277995\t-0.0212141508977\t-0.318065610507\t0.161950668081\t-0.399893913221\t-0.0877433121303\n',
'o3\t0.0\t0.0\t0.0\t-0.56834723307\t0.358282750767\t0.00896170476124\t0.0492385385968\t0.461519932778\t-0.0724438545352\t0.367756741918\n',
'o4\t0.0\t0.0\t0.0\t0.0\t-0.372119766329\t-0.456463557331\t0.131399092367\t-0.168189387492\t0.120702688296\t-0.43427312006\n',
'o5\t0.0\t0.0\t0.0\t0.0\t0.0\t-0.0531030469782\t0.275507536817\t-0.0762901337041\t0.546045014206\t0.762915065103\n',
'o6\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t-0.384259859372\t0.176020718545\t-0.167238874122\t-0.117069038616\n',
'o7\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t-0.120782471357\t0.716252313528\t-0.0642248448104\n',
'o8\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t-0.111957030226\t-0.186558256664\n',
'o9\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0953429339494\n',
'o10\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0']

BC_LINES = [\
 '#OTU_ID\to1\to2\to3\to4\to5',
 'o1\t0\t6\t12\t18\t24',
 'o2\t6\t0\t18\t24\t30',
 'o3\t12\t18\t0\t30\t36',
 'o4\t18\t24\t30\t0\t42',
 'o5\t24\t30\t36\t42\t0']

MIC_LINES = [\
 '1 0.724 0.196 0.032 0.86 0.506 0.926 0.126 0.45 0.772 0.678\n',
 '0.724 1 0.654 0.368 0.806 0.644 0.102 0.918 0.176 0.35 0.492\n',
 '0.196 0.654 1 0.194 0.652 0.298 0.896 0.856 0.57 0.592 0.738\n',
 '0.032 0.368 0.194 1 0.616 0.764 0.494 0.574 0.146 0.418 0.114\n',
 '0.86 0.806 0.652 0.616 1 0.106 0.83 0.552 0.082 0.618 0.39\n',
 '0.506 0.644 0.298 0.764 0.106 1 0.07 0.608 0.142 0.006 0.208\n',
 '0.926 0.102 0.896 0.494 0.83 0.07 1 0.636 0.46 0.78 0.99\n',
 '0.126 0.918 0.856 0.574 0.552 0.608 0.636 1 0.626 0.206 0.334\n',
 '0.45 0.176 0.57 0.146 0.082 0.142 0.46 0.626 1 0.432 0.83\n',
 '0.772 0.35 0.592 0.418 0.618 0.006 0.78 0.206 0.432 1 0.028\n',
 '0.678 0.492 0.738 0.114 0.39 0.208 0.99 0.334 0.83 0.028 1\n']


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

    def test_self_properties_redundant(self):
        '''Test properties like edges, otus, data, when filterd by LS pvals.'''
        LSAResultsObj = LSAResults(LSA_LINES_REDUNDANT, filter='ls', sig_lvl=.2,
            rtype='redundant')
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

    def test_self_properties_unique(self):
        '''Test properties are correct when non-redundant input is supplied.'''
        LSAResultsObj = LSAResults(LSA_LINES_UNIQUE, filter='ls', sig_lvl=.2,
            rtype='unique')
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

class NaiveResultsTests(TestCase):
    '''Test that naive results are being properly parsed.'''

    def setUp(self):
        '''Create variables/data need for this test.'''
        self.NaiveResultsObj1 = NaiveResults(NAIVE_CVAL_LINES, NAIVE_PVAL_LINES,
            .2, empirical=False)
        self.NaiveResultsObj2 = NaiveResults(NAIVE_CVAL_LINES, NAIVE_PVAL_LINES,
            .2, empirical=True)

    def test_getSignificantData(self):
        '''Test that data is returned appropriately.'''
        # test with empirical = false, NaiveResultsObj1
        exp_pdata = array([[0.,0.20643423,0.62299182,0.26051706,0.67605068,
            0.005214,0.40806731,0.94641555,0.94690987,0.76681123],
            [0.,0.,0.8643968,0.72094756,0.93623498,
            0.95523363,0.38334046,0.66553259,0.2624856,0.81595903],
            [0.,0.,0.,0.08787482,0.3212316,
            0.98108307,0.89626709,0.18654565,0.84773846,0.30734872],
            [0.,0.,0.,0.,0.30105446,
            0.19226134,0.72658554,0.65324709,0.74827566,0.21844484],
            [0.,0.,0.,0.,0.,
            0.88816245,0.45430598,0.83973091,0.10499027,0.0079522],
            [0.,0.,0.,0.,0.,
            0.,0.28387453,0.63792055,0.6551146,0.75567724],
            [0.,0.,0.,0.,0.,
            0.,0.,0.74811333,0.01726906,0.86488652],
            [0.,0.,0.,0.,0.,
            0.,0.,0.,0.76611728,0.61747251],
            [0.,0.,0.,0.,0.,
            0.,0.,0.,0.,0.80025103],
            [0.,0.,0.,0.,0.,
            0.,0.,0.,0.,0.]])
        exp_cdata = array([[ 0.        ,  0.44426501, -0.1837035 , -0.40136437,  0.15663542,
             0.78406882, -0.30288215,  0.02539704, -0.0251625 , -0.11161777],
           [ 0.        ,  0.        ,  0.06445925, -0.13419185,  0.03022903,
            -0.02121415, -0.31806561,  0.16195067, -0.39989391, -0.08774331],
           [ 0.        ,  0.        ,  0.        , -0.56834723,  0.35828275,
             0.0089617 ,  0.04923854,  0.46151993, -0.07244385,  0.36775674],
           [ 0.        ,  0.        ,  0.        ,  0.        , -0.37211977,
            -0.45646356,  0.13139909, -0.16818939,  0.12070269, -0.43427312],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            -0.05310305,  0.27550754, -0.07629013,  0.54604501,  0.76291507],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        , -0.38425986,  0.17602072, -0.16723887, -0.11706904],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        , -0.12078247,  0.71625231, -0.06422484],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        , -0.11195703, -0.18655826],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.09534293],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
        self.assertFloatEqual(self.NaiveResultsObj1.pdata, exp_pdata)
        self.assertFloatEqual(self.NaiveResultsObj1.cdata, exp_cdata)
        sig_edges = (array([0,2,2,3,4,4,6]), array([5,3,7,5,8,9,8]))
        otu1 = ['o%s' % (i+1) for i in sig_edges[0]]
        otu2 = ['o%s' % (i+1) for i in sig_edges[1]]
        sig_otus = list(set(otu1+otu2))
        edges = zip(otu1, otu2)
        pvals = [exp_pdata[i][j] for i,j in zip(sig_edges[0],sig_edges[1])]
        interactions = ['copresence', 'copresence', 'mutualExclusion', 
            'copresence', 'mutualExclusion', 'copresence']
        self.assertEqual(sig_edges, self.NaiveResultsObj1.sig_edges)
        self.assertEqual(otu1, self.NaiveResultsObj1.otu1)
        self.assertEqual(otu2, self.NaiveResultsObj1.otu2)
        self.assertEqual(set(sig_otus), 
            set(self.NaiveResultsObj1.sig_otus))
        self.assertEqual(edges, self.NaiveResultsObj1.edges)
        self.assertFloatEqual(pvals, self.NaiveResultsObj1.pvals)


class BrayCurtisParserTests(TestCase):
    """Test that the Bray Curtis parser works as expected."""

    def setUp(self):
        '''Create data all functions need.'''
        pass

    def test_getSignificantData(self):
        '''Test that _getSignificantDaa works as intended (as well as init).'''
        sig_lvl = .3 # 7 unique vals in data, .2*7 = 2.1 -> 2 chosen.
        ro = BrayCurtisResults(BC_LINES, sig_lvl)
        # tests begin
        exp_data = array([
            [ 0,  6, 12, 18, 24],
            [ 6,  0, 18, 24, 30],
            [12, 18,  0, 30, 36],
            [18, 24, 30,  0, 42],
            [24, 30, 36, 42,  0]])
        exp_otu_ids = ['o1','o2','o3','o4','o5']
        exp_actual_sig_lvl = 2/10.
        exp_sig_edges = (array([0, 0]), array([1,2]))
        exp_otu1 = ['o1', 'o1']
        exp_otu2 = ['o2', 'o3']
        self.assertFloatEqual(exp_data, ro.data)
        self.assertEqual(exp_otu_ids, ro.otu_ids)
        self.assertFloatEqual(exp_actual_sig_lvl, ro.actual_sig_lvl)
        self.assertFloatEqual(exp_sig_edges, ro.sig_edges)
        self.assertEqual(exp_otu1, ro.otu1)
        self.assertEqual(exp_otu2, ro.otu2)


class MICResultsParserTests(TestCase):
    """Test that the MIC parser is operating as intended."""

    def setUp(self):
        pass

    def test_getSignificantData(self):
        '''Test that _getSignificantDaa works as intended (as well as init).'''
        features = ['o0','o1','o2','o3','o4','o5','o6','o7','o8','o9','o10']
        sig_lvl = .1 # len(exp_cvals) = 54 * .1 = 5.4 -> 5 -> exp_cvals[-5]=.86
        ro = MICResults(MIC_LINES, features, sig_lvl)
        # expected cvals
        # exp_cvals = array([ 
        #     0.006,  0.028,  0.032,  0.07 ,  0.082,  0.102,  0.106,  0.114,
        #     0.126,  0.142,  0.146,  0.176,  0.194,  0.196,  0.206,  0.208,
        #     0.298,  0.334,  0.35 ,  0.368,  0.39 ,  0.418,  0.432,  0.45 ,
        #     0.46 ,  0.492,  0.494,  0.506,  0.552,  0.57 ,  0.574,  0.592,
        #     0.608,  0.616,  0.618,  0.626,  0.636,  0.644,  0.652,  0.654,
        #     0.678,  0.724,  0.738,  0.764,  0.772,  0.78 ,  0.806,  0.83 ,
        #     0.856,  0.86 ,  0.896,  0.918,  0.926,  0.99 ])
        exp_data = array([
           [ 1.   ,  0.724,  0.196,  0.032,  0.86 ,  0.506,  0.926,  0.126,
             0.45 ,  0.772,  0.678],
           [ 0.724,  1.   ,  0.654,  0.368,  0.806,  0.644,  0.102,  0.918,
             0.176,  0.35 ,  0.492],
           [ 0.196,  0.654,  1.   ,  0.194,  0.652,  0.298,  0.896,  0.856,
             0.57 ,  0.592,  0.738],
           [ 0.032,  0.368,  0.194,  1.   ,  0.616,  0.764,  0.494,  0.574,
             0.146,  0.418,  0.114],
           [ 0.86 ,  0.806,  0.652,  0.616,  1.   ,  0.106,  0.83 ,  0.552,
             0.082,  0.618,  0.39 ],
           [ 0.506,  0.644,  0.298,  0.764,  0.106,  1.   ,  0.07 ,  0.608,
             0.142,  0.006,  0.208],
           [ 0.926,  0.102,  0.896,  0.494,  0.83 ,  0.07 ,  1.   ,  0.636,
             0.46 ,  0.78 ,  0.99 ],
           [ 0.126,  0.918,  0.856,  0.574,  0.552,  0.608,  0.636,  1.   ,
             0.626,  0.206,  0.334],
           [ 0.45 ,  0.176,  0.57 ,  0.146,  0.082,  0.142,  0.46 ,  0.626,
             1.   ,  0.432,  0.83 ],
           [ 0.772,  0.35 ,  0.592,  0.418,  0.618,  0.006,  0.78 ,  0.206,
             0.432,  1.   ,  0.028],
           [ 0.678,  0.492,  0.738,  0.114,  0.39 ,  0.208,  0.99 ,  0.334,
             0.83 ,  0.028,  1.   ]])
        exp_otu_ids = ['o0','o1','o2','o3','o4','o5','o6','o7','o8','o9','o10']
        exp_actual_sig_lvl = 5/55. # 11*10/2 = 55, 5 vals <= .082
        exp_sig_edges = (array([0,0,1,2,6]), array([4,6,7,6,10]))
        exp_otu1 = ['o0', 'o0', 'o1', 'o2', 'o6']
        exp_otu2 = ['o4', 'o6', 'o7', 'o6', 'o10']
        self.assertFloatEqual(exp_data, ro.data)
        self.assertEqual(exp_otu_ids, ro.otu_ids)
        self.assertFloatEqual(exp_actual_sig_lvl, ro.actual_sig_lvl)
        self.assertFloatEqual(exp_sig_edges, ro.sig_edges)
        self.assertEqual(exp_otu1, ro.otu1)
        self.assertEqual(exp_otu2, ro.otu2)






if __name__ == '__main__':
    main()