#!/usr/bin/env python

from biom.parse import parse_biom_table
from correlations.eval.parse import (sparcc_maker, conet_maker, rmt_maker, 
    lsa_maker, naive_maker, bray_curtis_maker, mic_maker)
from correlations.eval.result_eval import (interacting_edges)
import os
from numpy import cumsum
from collections import Counter
from itertools import combinations, chain

"""
ecological tables
-----------------

TS1:
    tables 6,7

TS2:
    tables 16,17,18
"""


# Input Filepaths #
biom_fps_t1 = [\
    '/Users/wdwvt1/src/correlations/tables/tables_3_8/bioms/table_6.biom',
    '/Users/wdwvt1/src/correlations/tables/tables_3_8/bioms/table_7.biom']

biom_fps_t2 = [\
    '/Users/wdwvt1/src/correlations/tables/tables_4_19/bioms/table_16.biom',
    '/Users/wdwvt1/src/correlations/tables/tables_4_19/bioms/table_17.biom',
    '/Users/wdwvt1/src/correlations/tables/tables_4_19/bioms/table_18.biom']

# original conet
conet_fps_t1_v1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/conet_rd1_tables/evaluation_ensemble_final_6.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/conet_rd1_tables/evaluation_ensemble_final_7.txt']

# new methodology including core speed up and MIC
conet_fps_t1_v2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/CoNet_tableset1_NEW/Round1TabNetworksCoNet/evaluation_ensemble_6.gdl.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/CoNet_tableset1_NEW/Round1TabNetworksCoNet/evaluation_ensemble_7.gdl.txt']

# arcane
conet_fps_t1_v3 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/rd1_arcane/evaluation_ensemble_final_6_indirect_filtered.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/rd1_arcane/evaluation_ensemble_final_7_indirect_filtered.txt']

# original conet
conet_fps_t2_v1= [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/CoNet_rd2_tables/evaluation_ensemble_final_16.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/CoNet_rd2_tables/evaluation_ensemble_final_17.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/CoNet_rd2_tables/evaluation_ensemble_final_18.txt']

# new methodology including core speed up and MIC
conet_fps_t2_v2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/CoNet_rd2_tables_NEW/evaluation_ensemble_16.gdl.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/CoNet_rd2_tables_NEW/evaluation_ensemble_17.gdl.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/CoNet_rd2_tables_NEW/evaluation_ensemble_18.gdl.txt']

# experimental procedures
conet_fps_t2_v3 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/rd2_experimental_procedure/Series2Experiment1/evaluation_ensemble_16.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/rd2_experimental_procedure/Series2Experiment1/evaluation_ensemble_17.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/conet/rd2_experimental_procedure/Series2Experiment1/evaluation_ensemble_18.txt']

sparcc_cval_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd1_tables/cvals/SparCC_correlations.xiter_0.table_6.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd1_tables/cvals/SparCC_correlations.xiter_0.table_7.txt']

sparcc_pval_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd1_tables/pvals/SparCC_pvalues.xiter_0.table_6.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd1_tables/pvals/SparCC_pvalues.xiter_0.table_7.txt']

sparcc_cval_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd2_4_19/SparCC_correlations.xiter_10.table_16.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd2_4_19/SparCC_correlations.xiter_10.table_17.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd2_4_19/SparCC_correlations.xiter_10.table_18.txt']

sparcc_pval_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd2_4_19/SparCC_pvalues.xiter_10.table_16.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd2_4_19/SparCC_pvalues.xiter_10.table_17.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/sparcc/sparcc_rd2_4_19/SparCC_pvalues.xiter_10.table_18.txt']

rmt_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/rmt/rmt_rd1_tables/Table6.RMT.list.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/rmt/rmt_rd1_tables/Table7.RMT.list.txt']

rmt_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/rmt/RMT_rd2_tables/Table16.Rob.list.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/rmt/RMT_rd2_tables/Table17.Rob.list.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/rmt/RMT_rd2_tables/Table18.Rob.list.txt']

lsa_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/lsa/lsa_rd1_tables/table_6.theo.lsa',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/lsa/lsa_rd1_tables/table_7.theo.lsa']

lsa_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/lsa/lsa_rd2/table_16.delay.theo.lsa',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/lsa/lsa_rd2/table_17.delay.theo.lsa',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/lsa/lsa_rd2/table_18.delay.theo.lsa']

mic_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/mic/rd1/tables_3_8/table_6.txt/MIC',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/mic/rd1/tables_3_8/table_7.txt/MIC']

mic_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/mic/rd2/tables_4_19/table_16.txt/MIC',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/mic/rd2/tables_4_19/table_17.txt/MIC',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/mic/rd2/tables_4_19/table_18.txt/MIC']

pz_cval_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_3_8/table_6_pearson_fisher_z_transform_cval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_3_8/table_7_pearson_fisher_z_transform_cval.txt']

pz_pval_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_3_8/table_6_pearson_fisher_z_transform_pval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_3_8/table_7_pearson_fisher_z_transform_pval.txt']

pz_cval_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_16_pearson_fisher_z_transform_cval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_17_pearson_fisher_z_transform_cval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_18_pearson_fisher_z_transform_cval.txt']

pz_pval_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_16_pearson_fisher_z_transform_pval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_17_pearson_fisher_z_transform_pval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_18_pearson_fisher_z_transform_pval.txt']

sz_cval_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_3_8/table_6_spearman_fisher_z_transform_cval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_3_8/table_7_spearman_fisher_z_transform_cval.txt']

sz_pval_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_3_8/table_6_spearman_fisher_z_transform_pval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_3_8/table_7_spearman_fisher_z_transform_pval.txt']

sz_cval_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_16_spearman_fisher_z_transform_cval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_17_spearman_fisher_z_transform_cval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_18_spearman_fisher_z_transform_cval.txt']

sz_pval_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_16_spearman_fisher_z_transform_pval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_17_spearman_fisher_z_transform_pval.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/naive_cc_tool/tables_4_19/table_18_spearman_fisher_z_transform_pval.txt']

bc_fps_t1 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/bray_curtis/rd1/table_6._dists.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/bray_curtis/rd1/table_7._dists.txt']

bc_fps_t2 = [\
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/bray_curtis/rd2_4_19/table_16._dists.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/bray_curtis/rd2_4_19/table_17._dists.txt',
    '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/tool_result_files/bray_curtis/rd2_4_19/table_18._dists.txt']


# TS1
##### table 6
t1_6_methods = [\
 'competitively_related_2d_st_3',
 'mutually_related_2d_st_5',
 'commensually_related_2d_st_5',
 'competitively_related_1d_st_3',
 'parasitically_related_2d_st_2',
 'commensually_related_1d_st_2',
 'obligate_related_2d_st_5',
 'competitively_related_2d_st_5',
 'commensually_related_1d_st_3',
 'partial-obligate-syntrophic_related_2d_st_NA',
 'commensually_related_2d_st_3',
 'parasitically_related_2d_st_5',
 'competitively_related_1d_st_5',
 'competitively_related_1d_st_2',
 'parasitically_related_1d_st_3',
 'mutually_related_1d_st_5']

t1_6_ees = [90, 90, 90, 60, 90, 60, 90, 90, 60, 90, 90, 90, 60, 60, 60, 60]

##### table 7
t1_7_methods = [\
 'mutually_related_2d_st_2',
 'commensually_related_1d_st_5',
 'competitively_related_2d_st_2',
 'parasitically_related_1d_st_2',
 'mutually_related_1d_st_3',
 'amensally_related_1d_st_5',
 'amensally_related_2d_st_2',
 'mutually_related_2d_st_3',
 'amensally_related_1d_st_3',
 'partial-obligate-syntrophic_related_1d_st_NA',
 'amensally_related_1d_st_2',
 'obligate_related_1d_st_5',
 'mutually_related_1d_st_2',
 'parasitically_related_2d_st_3',
 'parasitically_related_1d_st_5',
 'commensually_related_2d_st_2',
 'amensally_related_2d_st_5',
 'amensally_related_2d_st_3']

t1_7_ees = [90, 60, 90, 60, 60, 60, 90, 90, 60, 60, 60, 20, 60, 90, 60, 90, 90, 
    90]

# TS2
##### table 16
# tables 16, 17, and 18 have the same generating methods and engineered edges
t2_methods = [\
 'competitively_related_1d_st_3',
 'commensually_related_1d_st_2',
 'commensually_related_1d_st_3',
 'competitively_related_1d_st_5',
 'competitively_related_1d_st_2',
 'mutually_related_1d_st_5',
 'commensually_related_1d_st_5',
 'mutually_related_1d_st_3',
 'amensally_related_1d_st_5',
 'amensally_related_1d_st_3',
 'amensally_related_1d_st_2',
 'mutually_related_1d_st_2']

t2_ees = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]

def _param_getter(method):
    '''Return the parameters that the method name encodes.'''
    tmp = method.split('_')
    method_name = tmp[0]
    strength = tmp[-1]
    dim = int(tmp[2][0])
    return method_name, strength, dim

def get_params_from_methods(methods):
    '''return all paramaters from methods.'''
    tmp = map(_param_getter, methods)
    method_names, strengths, dims = map(list, zip(*tmp))
    return method_names, strengths, dims

def count_ecological_edges(methods, num_ees, ro):
    '''Automates counting of ecological edges.

    Inputs:
     methods - list of strs. names of methods that weer used to generate the 
      engineered edges. must be formatted properly (e.g. 
      'amensally_related_1d_st_3' and in the same order as the num_ees list. 
     num_ees - list of ints. numbers of otus that are generated by each 
      method in the methods list.
     ro - result object.
    '''
    starts = cumsum([0]+num_ees[:-1]) #add a 0 entry for start indices
    stops = cumsum(num_ees)
    method_names, strengths, dims = get_params_from_methods(methods)
    edge_counts = [interacting_edges(starts[i], stops[i], dims[i], ro.edges, 
        ro.interactions) for i in range(len(num_ees))] 
    return (method_names, strengths, dims, edge_counts, len(ro.edges))

def calc_tpe(ee, dim):
    '''calculate total possible edges for a given number of ees and a dim.

    each ee has dim+1 number of otus because the dim refers to only the LHS. 
    any edge is then between 2 of those nodes so we have nCr(dim+1, 2).
    generated_relats = ee/(dim + 1)
    possible_interactions_per_relat = dim*(dim+1)/2.
    tpe = generated_relats * possible_interactions_per_relat
    '''
    return (ee*dim)/2.

def eco_formatter(ie_results, methods, dims, strengths, num_ees, ue,
        de_count):
    '''Format the output of the interacting_edges script for easy writing.

    Inputs are each lists of the results from running interacting_edges and 
    associated functions through all the methods in a given table.
    '''
    pre_header = 'ue=%s\tde=%s' % (ue, de_count)
    header = ['method', 'dim', 'str', 'total_possible', 'total_detected', 
        'cis_cp', 'cis_me', 'trans_cp', 'trans_me']
    lines = [pre_header, '\t'.join(header)]
    for i in range(len(ie_results)):
        tp = calc_tpe(num_ees[i], dims[i])
        td = ie_results[i][0]
        cis_cp, cis_me = ie_results[i][2], ie_results[i][3]
        trans_cp, trans_me = ie_results[i][5], ie_results[i][6]
        nl = '\t'.join(map(str, [methods[i], dims[i], strengths[i], tp, td, 
            cis_cp, cis_me, trans_cp, trans_me]))
        lines.append(nl)
    return lines

def calc_ue(biom_fp, num_ees, dims):
    '''Calculate unengineered edges.'''
    bt = parse_biom_table(open(biom_fp))
    n = len(bt.ObservationIds)
    count_ee = sum([calc_tpe(ee, dim) for ee, dim in zip(num_ees, dims)])
    return (n*(n-1)/2.) - count_ee

def evaluate_eco_table(biom_fp, methods, num_ees, maker_fn, params_dict):
    '''
    '''
    ro = maker_fn(**params_dict)
    method_names, strengths, dims, dee_counts, de_count = \
        count_ecological_edges(methods, num_ees, ro)
    ue = calc_ue(biom_fp, num_ees, dims)
    lines = eco_formatter(dee_counts, method_names, dims, strengths, num_ees, 
        ue, de_count)
    return lines

def evaluate_ensemble_eco_table(ensemble_ro, methods, num_ees, biom_fp):
    '''
    '''
    method_names, strengths, dims, dee_counts, de_count = \
        count_ecological_edges(methods, num_ees, ensemble_ro)
    ue = calc_ue(biom_fp, num_ees, dims)
    lines = eco_formatter(dee_counts, method_names, dims, strengths, num_ees, 
        ue, de_count)
    return lines

def write_eco_output(lines, out_fp):
    '''Write the output of eco_formatter.'''
    o = open(out_fp, 'w')
    o.writelines('\n'.join(lines))
    o.close()

class HackishEdgeEnsemble:
    '''Class for combining edges hackishly.'''
    def __init__(self, results_objects):
        '''Combine results_objects; produce one set of edges/interactions.'''
        self.edges = []
        self.interactions = []
        # abuse sum to add edges together. non unique edges at this point
        all_edges = sum([ro.edges for ro in results_objects], [])
        # use sorted, map, and counter to count the number of edges. this 
        # ensures o1-o2 is counted the same as o2-o1
        tmp = Counter(map(lambda x: tuple(sorted(x)), all_edges))
        # make a list of edges
        shared_edges = [k for k,v in tmp.iteritems() if v==len(results_objects)]
        # if the shared edges don't have the same interaction type they will be
        # excluded
        for edge in shared_edges:
            tmp = []
            for ro in results_objects:
                try:
                    i = ro.edges.index(edge)
                except ValueError: #edge must have been reversed in this ro
                    i = ro.edges.index(edge[::-1])
                # append the interaction for the given edge
                tmp.append(ro.interactions[i])
            if len(set(tmp))==1: #all interactions were the same
                self.edges.append(edge)
                self.interactions.append(tmp[0])
            else: # interactions were different, not an agreement
                pass 



# Output Filepaths # 
base_output_fp = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/'
t1_6_ensemble_out = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t1_6/'
t1_7_ensemble_out = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t1_7/'
t2_16_ensemble_out = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t2_16/'
t2_17_ensemble_out = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t2_17/'
t2_18_ensemble_out = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t2_18/'

def _join(tool_fp):
    return os.path.join(base_output_fp, tool_fp)


conet_out_fps_t1_v1 = map(_join, ['conet_v1_t1.6.txt', 'conet_v1_t1.7.txt'])
conet_out_fps_t1_v2 = map(_join, ['conet_v2_t1.6.txt', 'conet_v2_t1.7.txt'])
conet_out_fps_t1_v3 = map(_join, ['conet_v3_t1.6.txt', 'conet_v3_t1.7.txt'])

conet_out_fps_t2_v1 = map(_join, ['conet_v1_t2.16.txt', 'conet_v1_t2.17.txt', 
    'conet_v1_t2.18.txt'])
conet_out_fps_t2_v2 = map(_join, ['conet_v2_t2.16.txt', 'conet_v2_t2.17.txt', 
    'conet_v2_t2.18.txt'])
conet_out_fps_t2_v3 = map(_join, ['conet_v3_t2.16.txt', 'conet_v3_t2.18.txt', 
    'conet_v3_t2.18.txt'])

sparcc_out_fps_t1_05 = map(_join, ['sparcc_t1.6_sig_05.txt', 
    'sparcc_t1.7_sig_05.txt'])
sparcc_out_fps_t1_01 = map(_join, ['sparcc_t1.6_sig_01.txt', 
    'sparcc_t1.7_sig_01.txt'])
sparcc_out_fps_t1_001 = map(_join, ['sparcc_t1.6_sig_001.txt', 
    'sparcc_t1.7_sig_001.txt'])

sparcc_out_fps_t2_05 = map(_join, ['sparcc_t2.16_sig_05.txt', 
    'sparcc_t2.17_sig_05.txt', 'sparcc_t2.18_sig_05.txt'])
sparcc_out_fps_t2_01 = map(_join, ['sparcc_t2.16_sig_01.txt', 
    'sparcc_t2.17_sig_01.txt', 'sparcc_t2.18_sig_01.txt'])
sparcc_out_fps_t2_001 = map(_join, ['sparcc_t2.16_sig_001.txt', 
    'sparcc_t2.17_sig_001.txt', 'sparcc_t2.18_sig_001.txt'])

lsa_out_fps_t1_05 = map(_join, ['lsa_t1.6_sig_05.txt', 'lsa_t1.7_sig_05.txt'])
lsa_out_fps_t1_01 = map(_join, ['lsa_t1.6_sig_01.txt', 'lsa_t1.7_sig_01.txt'])
lsa_out_fps_t1_001 = map(_join, ['lsa_t1.6_sig_001.txt', 'lsa_t1.7_sig_001.txt'])

lsa_out_fps_t2_05 = map(_join, ['lsa_t2.16_sig_05.txt', 
    'lsa_t2.17_sig_05.txt', 'lsa_t2.18_sig_05.txt'])
lsa_out_fps_t2_01 = map(_join, ['lsa_t2.16_sig_01.txt', 
    'lsa_t2.17_sig_01.txt', 'lsa_t2.18_sig_01.txt'])
lsa_out_fps_t2_001 = map(_join, ['lsa_t2.16_sig_001.txt', 
    'lsa_t2.17_sig_001.txt', 'lsa_t2.18_sig_001.txt'])

rmt_out_fps_t1 = map(_join, ['rmt_t1.6.txt', 'rmt_t1.7.txt'])

rmt_out_fps_t2 = map(_join, ['rmt_t2.16.txt', 'rmt_t2.17.txt', 'rmt_t2.18.txt'])

pz_out_fps_t1_05 = map(_join, ['pz_t1.6_sig_05.txt', 
    'pz_t1.7_sig_05.txt'])
pz_out_fps_t1_01 = map(_join, ['pz_t1.6_sig_01.txt', 
    'pz_t1.7_sig_01.txt'])
pz_out_fps_t1_001 = map(_join, ['pz_t1.6_sig_001.txt', 
    'pz_t1.7_sig_001.txt'])

pz_out_fps_t2_05 = map(_join, ['pz_t2.16_sig_05.txt', 
    'pz_t2.17_sig_05.txt', 'pz_t2.18_sig_05.txt'])
pz_out_fps_t2_01 = map(_join, ['pz_t2.16_sig_01.txt', 
    'pz_t2.17_sig_01.txt', 'pz_t2.18_sig_01.txt'])
pz_out_fps_t2_001 = map(_join, ['pz_t2.16_sig_001.txt', 
    'pz_t2.17_sig_001.txt', 'pz_t2.18_sig_001.txt'])

sz_out_fps_t1_05 = map(_join, ['sz_t1.6_sig_05.txt', 
    'sz_t1.7_sig_05.txt'])
sz_out_fps_t1_01 = map(_join, ['sz_t1.6_sig_01.txt', 
    'sz_t1.7_sig_01.txt'])
sz_out_fps_t1_001 = map(_join, ['sz_t1.6_sig_001.txt', 
    'sz_t1.7_sig_001.txt'])

sz_out_fps_t2_05 = map(_join, ['sz_t2.16_sig_05.txt', 
    'sz_t2.17_sig_05.txt', 'sz_t2.18_sig_05.txt'])
sz_out_fps_t2_01 = map(_join, ['sz_t2.16_sig_01.txt', 
    'sz_t2.17_sig_01.txt', 'sz_t2.18_sig_01.txt'])
sz_out_fps_t2_001 = map(_join, ['sz_t2.16_sig_001.txt', 
    'sz_t2.17_sig_001.txt', 'sz_t2.18_sig_001.txt'])


# # conet # 
# params_dict = {'ensemble_fp': conet_fps_t1_v1[0]}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t1_v1[0])
# params_dict = {'ensemble_fp': conet_fps_t1_v1[1]}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t1_v1[1])

# params_dict = {'ensemble_fp': conet_fps_t1_v2[0]}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t1_v2[0])
# params_dict = {'ensemble_fp': conet_fps_t1_v2[1]}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t1_v2[1])

# params_dict = {'ensemble_fp': conet_fps_t1_v3[0]}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t1_v3[0])
# params_dict = {'ensemble_fp': conet_fps_t1_v3[1]}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t1_v3[1])

# ### table set 2 

# params_dict = {'ensemble_fp': conet_fps_t2_v1[0]}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t2_v1[0])
# params_dict = {'ensemble_fp': conet_fps_t2_v1[1]}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t2_v1[1])
# params_dict = {'ensemble_fp': conet_fps_t2_v1[2]}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t2_v1[2])

# params_dict = {'ensemble_fp': conet_fps_t2_v2[0]}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t2_v2[0])
# params_dict = {'ensemble_fp': conet_fps_t2_v2[1]}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t2_v2[1])
# params_dict = {'ensemble_fp': conet_fps_t2_v2[2]}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t2_v2[2])

# params_dict = {'ensemble_fp': conet_fps_t2_v3[0]}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t2_v3[0])
# # this one won't work for some reason
# # params_dict = {'ensemble_fp': conet_fps_t2_v3[1]}
# # lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
# #     conet_maker, params_dict)
# # write_eco_output(lines, conet_out_fps_t2_v3[1])
# params_dict = {'ensemble_fp': conet_fps_t2_v3[2]}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     conet_maker, params_dict)
# write_eco_output(lines, conet_out_fps_t2_v3[2])


# # sparcc #
# params_dict = {'biom_fp': biom_fps_t1[0], 'cval_fp': sparcc_cval_fps_t1[0], 
#     'pval_fp': sparcc_pval_fps_t1[0], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t1_05[0])
# params_dict = {'biom_fp': biom_fps_t1[1], 'cval_fp': sparcc_cval_fps_t1[1], 
#     'pval_fp': sparcc_pval_fps_t1[1], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t1_05[1])

# params_dict = {'biom_fp': biom_fps_t1[0], 'cval_fp': sparcc_cval_fps_t1[0], 
#     'pval_fp': sparcc_pval_fps_t1[0], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t1_01[0])
# params_dict = {'biom_fp': biom_fps_t1[1], 'cval_fp': sparcc_cval_fps_t1[1], 
#     'pval_fp': sparcc_pval_fps_t1[1], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t1_01[1])

# params_dict = {'biom_fp': biom_fps_t1[0], 'cval_fp': sparcc_cval_fps_t1[0], 
#     'pval_fp': sparcc_pval_fps_t1[0], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t1_001[0])
# params_dict = {'biom_fp': biom_fps_t1[1], 'cval_fp': sparcc_cval_fps_t1[1], 
#     'pval_fp': sparcc_pval_fps_t1[1], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t1_001[1])

# ### tables set 2 
# params_dict = {'biom_fp': biom_fps_t2[0], 'cval_fp': sparcc_cval_fps_t2[0], 
#     'pval_fp': sparcc_pval_fps_t2[0], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_05[0])
# params_dict = {'biom_fp': biom_fps_t2[1], 'cval_fp': sparcc_cval_fps_t2[1], 
#     'pval_fp': sparcc_pval_fps_t2[1], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_05[1])
# params_dict = {'biom_fp': biom_fps_t2[2], 'cval_fp': sparcc_cval_fps_t2[2], 
#     'pval_fp': sparcc_pval_fps_t2[2], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_05[2])

# params_dict = {'biom_fp': biom_fps_t2[0], 'cval_fp': sparcc_cval_fps_t2[0], 
#     'pval_fp': sparcc_pval_fps_t2[0], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_01[0])
# params_dict = {'biom_fp': biom_fps_t2[1], 'cval_fp': sparcc_cval_fps_t2[1], 
#     'pval_fp': sparcc_pval_fps_t2[1], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_01[1])
# params_dict = {'biom_fp': biom_fps_t2[2], 'cval_fp': sparcc_cval_fps_t2[2], 
#     'pval_fp': sparcc_pval_fps_t2[2], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_01[2])

# params_dict = {'biom_fp': biom_fps_t2[0], 'cval_fp': sparcc_cval_fps_t2[0], 
#     'pval_fp': sparcc_pval_fps_t2[0], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_001[0])
# params_dict = {'biom_fp': biom_fps_t2[1], 'cval_fp': sparcc_cval_fps_t2[1], 
#     'pval_fp': sparcc_pval_fps_t2[1], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_001[1])
# params_dict = {'biom_fp': biom_fps_t2[2], 'cval_fp': sparcc_cval_fps_t2[2], 
#     'pval_fp': sparcc_pval_fps_t2[2], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     sparcc_maker, params_dict)
# write_eco_output(lines, sparcc_out_fps_t2_001[2])


# # lsa
# params_dict = {'lsa_fp': lsa_fps_t1[0], 'filter_str': 'ls', 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t1_05[0])
# params_dict = {'lsa_fp': lsa_fps_t1[1], 'filter_str': 'ls', 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t1_05[1])

# params_dict = {'lsa_fp': lsa_fps_t1[0], 'filter_str': 'ls', 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t1_01[0])
# params_dict = {'lsa_fp': lsa_fps_t1[1], 'filter_str': 'ls', 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t1_01[1])

# params_dict = {'lsa_fp': lsa_fps_t1[0], 'filter_str': 'ls', 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t1_001[0])
# params_dict = {'lsa_fp': lsa_fps_t1[1], 'filter_str': 'ls', 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t1_001[1])

# ### tables set 2 
# params_dict = {'lsa_fp': lsa_fps_t2[0], 'filter_str': 'ls', 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_05[0])
# params_dict = {'lsa_fp': lsa_fps_t2[1], 'filter_str': 'ls', 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_05[1])
# params_dict = {'lsa_fp': lsa_fps_t2[2], 'filter_str': 'ls', 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_05[2])

# params_dict = {'lsa_fp': lsa_fps_t2[0], 'filter_str': 'ls', 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_01[0])
# params_dict = {'lsa_fp': lsa_fps_t2[1], 'filter_str': 'ls', 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_01[1])
# params_dict = {'lsa_fp': lsa_fps_t2[2], 'filter_str': 'ls', 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_01[2])

# params_dict = {'lsa_fp': lsa_fps_t2[0], 'filter_str': 'ls', 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_001[0])
# params_dict = {'lsa_fp': lsa_fps_t2[1], 'filter_str': 'ls', 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_001[1])
# params_dict = {'lsa_fp': lsa_fps_t2[2], 'filter_str': 'ls', 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     lsa_maker, params_dict)
# write_eco_output(lines, lsa_out_fps_t2_001[2])


# # rmt
# params_dict = {'results_fp': rmt_fps_t1[0]}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     rmt_maker, params_dict)
# write_eco_output(lines, rmt_out_fps_t1[0])
# params_dict = {'results_fp': rmt_fps_t1[1]}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     rmt_maker, params_dict)
# write_eco_output(lines, rmt_out_fps_t1[1])

# ## table set 2
# params_dict = {'results_fp': rmt_fps_t2[0]}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     rmt_maker, params_dict)
# write_eco_output(lines, rmt_out_fps_t2[0])
# params_dict = {'results_fp': rmt_fps_t2[1]}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     rmt_maker, params_dict)
# write_eco_output(lines, rmt_out_fps_t2[1])
# params_dict = {'results_fp': rmt_fps_t2[2]}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     rmt_maker, params_dict)
# write_eco_output(lines, rmt_out_fps_t2[2])


# # pz #
# params_dict = {'cval_fp': pz_cval_fps_t1[0], 
#     'pval_fp': pz_pval_fps_t1[0], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t1_05[0])
# params_dict = {'cval_fp': pz_cval_fps_t1[1], 
#     'pval_fp': pz_pval_fps_t1[1], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t1_05[1])

# params_dict = {'cval_fp': pz_cval_fps_t1[0], 
#     'pval_fp': pz_pval_fps_t1[0], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t1_01[0])
# params_dict = {'cval_fp': pz_cval_fps_t1[1], 
#     'pval_fp': pz_pval_fps_t1[1], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t1_01[1])

# params_dict = {'cval_fp': pz_cval_fps_t1[0], 
#     'pval_fp': pz_pval_fps_t1[0], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t1_001[0])
# params_dict = {'cval_fp': pz_cval_fps_t1[1], 
#     'pval_fp': pz_pval_fps_t1[1], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t1_001[1])

# ### table set 2
# params_dict = {'cval_fp': pz_cval_fps_t2[0], 
#     'pval_fp': pz_pval_fps_t2[0], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_05[0])
# params_dict = {'cval_fp': pz_cval_fps_t2[1], 
#     'pval_fp': pz_pval_fps_t2[1], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_05[1])
# params_dict = {'cval_fp': pz_cval_fps_t2[2], 
#     'pval_fp': pz_pval_fps_t2[2], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_05[2])

# params_dict = {'cval_fp': pz_cval_fps_t2[0], 
#     'pval_fp': pz_pval_fps_t2[0], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_01[0])
# params_dict = {'cval_fp': pz_cval_fps_t2[1], 
#     'pval_fp': pz_pval_fps_t2[1], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_01[1])
# params_dict = {'cval_fp': pz_cval_fps_t2[2], 
#     'pval_fp': pz_pval_fps_t2[2], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_01[2])

# params_dict = {'cval_fp': pz_cval_fps_t2[0], 
#     'pval_fp': pz_pval_fps_t2[0], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_001[0])
# params_dict = {'cval_fp': pz_cval_fps_t2[1], 
#     'pval_fp': pz_pval_fps_t2[1], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_001[1])
# params_dict = {'cval_fp': pz_cval_fps_t2[2], 
#     'pval_fp': pz_pval_fps_t2[2], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, pz_out_fps_t2_001[2])



# # sz #
# params_dict = {'cval_fp': sz_cval_fps_t1[0], 
#     'pval_fp': sz_pval_fps_t1[0], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t1_05[0])
# params_dict = {'cval_fp': sz_cval_fps_t1[1], 
#     'pval_fp': sz_pval_fps_t1[1], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t1_05[1])

# params_dict = {'cval_fp': sz_cval_fps_t1[0], 
#     'pval_fp': sz_pval_fps_t1[0], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t1_01[0])
# params_dict = {'cval_fp': sz_cval_fps_t1[1], 
#     'pval_fp': sz_pval_fps_t1[1], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t1_01[1])

# params_dict = {'cval_fp': sz_cval_fps_t1[0], 
#     'pval_fp': sz_pval_fps_t1[0], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t1[0], t1_6_methods, t1_6_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t1_001[0])
# params_dict = {'cval_fp': sz_cval_fps_t1[1], 
#     'pval_fp': sz_pval_fps_t1[1], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t1[1], t1_7_methods, t1_7_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t1_001[1])

# params_dict = {'cval_fp': sz_cval_fps_t2[0], 
#     'pval_fp': sz_pval_fps_t2[0], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_05[0])
# params_dict = {'cval_fp': sz_cval_fps_t2[1], 
#     'pval_fp': sz_pval_fps_t2[1], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_05[1])
# params_dict = {'cval_fp': sz_cval_fps_t2[2], 
#     'pval_fp': sz_pval_fps_t2[2], 'sig_lvl': .05}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_05[2])

# params_dict = {'cval_fp': sz_cval_fps_t2[0], 
#     'pval_fp': sz_pval_fps_t2[0], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_01[0])
# params_dict = {'cval_fp': sz_cval_fps_t2[1], 
#     'pval_fp': sz_pval_fps_t2[1], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_01[1])
# params_dict = {'cval_fp': sz_cval_fps_t2[2], 
#     'pval_fp': sz_pval_fps_t2[2], 'sig_lvl': .01}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_01[2])

# params_dict = {'cval_fp': sz_cval_fps_t2[0], 
#     'pval_fp': sz_pval_fps_t2[0], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[0], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_001[0])
# params_dict = {'cval_fp': sz_cval_fps_t2[1], 
#     'pval_fp': sz_pval_fps_t2[1], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[1], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_001[1])
# params_dict = {'cval_fp': sz_cval_fps_t2[2], 
#     'pval_fp': sz_pval_fps_t2[2], 'sig_lvl': .001}
# lines = evaluate_eco_table(biom_fps_t2[2], t2_methods, t2_ees, 
#     naive_maker, params_dict)
# write_eco_output(lines, sz_out_fps_t2_001[2])




#####
# Ensemble approach
#####

# table 1.6
# [conet, sparcc, lsa, rmt, pz, sz] 
# params = [\
#     {'ensemble_fp': conet_fps_t1_v1[0]},
#     {'biom_fp': biom_fps_t1[0], 'cval_fp': sparcc_cval_fps_t1[0], 
#         'pval_fp': sparcc_pval_fps_t1[0], 'sig_lvl': .001},
#     {'lsa_fp': lsa_fps_t1[0], 'filter_str': 'ls', 'sig_lvl': .001},
#     {'results_fp': rmt_fps_t1[0]}, 
#     {'cval_fp': pz_cval_fps_t1[0], 
#         'pval_fp': pz_pval_fps_t1[0], 'sig_lvl': .001},
#     {'cval_fp': sz_cval_fps_t1[0], 
#         'pval_fp': sz_pval_fps_t1[0], 'sig_lvl': .001}]
# fns = [conet_maker, sparcc_maker, lsa_maker, rmt_maker, naive_maker, 
#     naive_maker]

# robjs = [fn(**p) for fn,p in zip(fns, params)]
# # http://stackoverflow.com/questions/1482308/whats-a-good-way-to-combinate-through-a-set
# combos = chain.from_iterable(combinations(robjs, n) for n in 
#     range(2, len(robjs)+1))
# names = ['conet', 'sparcc', 'lsa', 'rmt', 'pz', 'sz']
# name_combos = chain.from_iterable(combinations(names, n) for n in
#     range(2, len(robjs)+1))

# # make the output directory
# os.mkdir(t1_6_ensemble_out)

# for c,n in zip(combos, name_combos):
#     ensemble = HackishEdgeEnsemble(c)
#     lines = evaluate_ensemble_eco_table(ensemble, t1_6_methods, t1_6_ees, 
#         biom_fps_t1[0])
#     write_eco_output(lines, os.path.join(t1_6_ensemble_out, 
#         ('_'.join(n)+'.txt')))

# # table 1.7
# params = [\
#     {'ensemble_fp': conet_fps_t1_v1[1]},
#     {'biom_fp': biom_fps_t1[1], 'cval_fp': sparcc_cval_fps_t1[1], 
#         'pval_fp': sparcc_pval_fps_t1[1], 'sig_lvl': .001},
#     {'lsa_fp': lsa_fps_t1[1], 'filter_str': 'ls', 'sig_lvl': .001},
#     {'results_fp': rmt_fps_t1[1]}, 
#     {'cval_fp': pz_cval_fps_t1[1], 
#         'pval_fp': pz_pval_fps_t1[1], 'sig_lvl': .001},
#     {'cval_fp': sz_cval_fps_t1[1], 
#         'pval_fp': sz_pval_fps_t1[1], 'sig_lvl': .001}]
# fns = [conet_maker, sparcc_maker, lsa_maker, rmt_maker, naive_maker, 
#     naive_maker]

# robjs = [fn(**p) for fn,p in zip(fns, params)]
# # http://stackoverflow.com/questions/1482308/whats-a-good-way-to-combinate-through-a-set
# combos = chain.from_iterable(combinations(robjs, n) for n in 
#     range(2, len(robjs)+1))
# names = ['conet', 'sparcc', 'lsa', 'rmt', 'pz', 'sz']
# name_combos = chain.from_iterable(combinations(names, n) for n in
#     range(2, len(robjs)+1))

# # make the output directory
# os.mkdir(t1_7_ensemble_out)

# for c,n in zip(combos, name_combos):
#     ensemble = HackishEdgeEnsemble(c)
#     lines = evaluate_ensemble_eco_table(ensemble, t1_7_methods, t1_7_ees, 
#         biom_fps_t1[1])
#     write_eco_output(lines, os.path.join(t1_7_ensemble_out, 
#         ('_'.join(n)+'.txt')))

# 2.16
# params = [\
#     {'ensemble_fp': conet_fps_t2_v1[0]},
#     {'biom_fp': biom_fps_t2[0], 'cval_fp': sparcc_cval_fps_t2[0], 
#         'pval_fp': sparcc_pval_fps_t2[0], 'sig_lvl': .001},
#     {'lsa_fp': lsa_fps_t2[0], 'filter_str': 'ls', 'sig_lvl': .001},
#     {'results_fp': rmt_fps_t2[0]}, 
#     {'cval_fp': pz_cval_fps_t2[0], 
#         'pval_fp': pz_pval_fps_t2[0], 'sig_lvl': .001},
#     {'cval_fp': sz_cval_fps_t2[0], 
#         'pval_fp': sz_pval_fps_t2[0], 'sig_lvl': .001}]
# fns = [conet_maker, sparcc_maker, lsa_maker, rmt_maker, naive_maker, 
#     naive_maker]

# robjs = [fn(**p) for fn,p in zip(fns, params)]
# # http://stackoverflow.com/questions/1482308/whats-a-good-way-to-combinate-through-a-set
# combos = chain.from_iterable(combinations(robjs, n) for n in 
#     range(2, len(robjs)+1))
# names = ['conet', 'sparcc', 'lsa', 'rmt', 'pz', 'sz']
# name_combos = chain.from_iterable(combinations(names, n) for n in
#     range(2, len(robjs)+1))

# # make the output directory
# os.mkdir(t2_16_ensemble_out)

# for c,n in zip(combos, name_combos):
#     ensemble = HackishEdgeEnsemble(c)
#     lines = evaluate_ensemble_eco_table(ensemble, t2_methods, t2_ees, 
#         biom_fps_t2[0])
#     write_eco_output(lines, os.path.join(t2_16_ensemble_out, 
#         ('_'.join(n)+'.txt')))

# 2.17
params = [\
    {'ensemble_fp': conet_fps_t2_v1[1]},
    {'biom_fp': biom_fps_t2[1], 'cval_fp': sparcc_cval_fps_t2[1], 
        'pval_fp': sparcc_pval_fps_t2[1], 'sig_lvl': .001},
    {'lsa_fp': lsa_fps_t2[1], 'filter_str': 'ls', 'sig_lvl': .001},
    {'results_fp': rmt_fps_t2[1]}, 
    {'cval_fp': pz_cval_fps_t2[1], 
        'pval_fp': pz_pval_fps_t2[1], 'sig_lvl': .001},
    {'cval_fp': sz_cval_fps_t2[1], 
        'pval_fp': sz_pval_fps_t2[1], 'sig_lvl': .001}]
fns = [conet_maker, sparcc_maker, lsa_maker, rmt_maker, naive_maker, 
    naive_maker]

robjs = [fn(**p) for fn,p in zip(fns, params)]
# http://stackoverflow.com/questions/1482308/whats-a-good-way-to-combinate-through-a-set
combos = chain.from_iterable(combinations(robjs, n) for n in 
    range(2, len(robjs)+1))
names = ['conet', 'sparcc', 'lsa', 'rmt', 'pz', 'sz']
name_combos = chain.from_iterable(combinations(names, n) for n in
    range(2, len(robjs)+1))

# make the output directory
os.mkdir(t2_17_ensemble_out)

for c,n in zip(combos, name_combos):
    ensemble = HackishEdgeEnsemble(c)
    lines = evaluate_ensemble_eco_table(ensemble, t2_methods, t2_ees, 
        biom_fps_t2[1])
    write_eco_output(lines, os.path.join(t2_17_ensemble_out, 
        ('_'.join(n)+'.txt')))

# 2.18
params = [\
    {'ensemble_fp': conet_fps_t2_v1[2]},
    {'biom_fp': biom_fps_t2[2], 'cval_fp': sparcc_cval_fps_t2[2], 
        'pval_fp': sparcc_pval_fps_t2[2], 'sig_lvl': .001},
    {'lsa_fp': lsa_fps_t2[2], 'filter_str': 'ls', 'sig_lvl': .001},
    {'results_fp': rmt_fps_t2[2]}, 
    {'cval_fp': pz_cval_fps_t2[2], 
        'pval_fp': pz_pval_fps_t2[2], 'sig_lvl': .001},
    {'cval_fp': sz_cval_fps_t2[2], 
        'pval_fp': sz_pval_fps_t2[2], 'sig_lvl': .001}]
fns = [conet_maker, sparcc_maker, lsa_maker, rmt_maker, naive_maker, 
    naive_maker]

robjs = [fn(**p) for fn,p in zip(fns, params)]
# http://stackoverflow.com/questions/1482308/whats-a-good-way-to-combinate-through-a-set
combos = chain.from_iterable(combinations(robjs, n) for n in 
    range(2, len(robjs)+1))
names = ['conet', 'sparcc', 'lsa', 'rmt', 'pz', 'sz']
name_combos = chain.from_iterable(combinations(names, n) for n in
    range(2, len(robjs)+1))

# make the output directory
os.mkdir(t2_18_ensemble_out)

for c,n in zip(combos, name_combos):
    ensemble = HackishEdgeEnsemble(c)
    lines = evaluate_ensemble_eco_table(ensemble, t2_methods, t2_ees, 
        biom_fps_t2[2])
    write_eco_output(lines, os.path.join(t2_18_ensemble_out, 
        ('_'.join(n)+'.txt')))





















