#!/usr/bin/env python

import glob
from numpy import array

def parse_eco_eval_out(in_fp):
    '''parse the results of the ecological_tables_evaluation.py work.'''
    identifier = in_fp.split('/')[-1]
    o = open(in_fp)
    lines = o.readlines()
    o.close()
    if len(lines) == 1: #excel converted \n to \r or \r\n
        lines = lines[0]
    tmp = lines[0].strip().split('\t')
    ue, de = [float(i.split('=')[1]) for i in tmp]
    data = array([line.split('\t') for line in lines[1:]])
    ee = data[1:,3].astype(float).sum()
    dee = data[1:,4].astype(float).sum()
    due = de - dee
    TP, FP, TN, FN = roc(ue, de, ee, dee, due)
    sens, spec, prec = sens_spec(TP, FP, TN, FN)
    print identifier, (TP+FP+TN+FN)
    return identifier, TP, FP, TN, FN, sens, spec, prec

def roc(ue, de, ee, dee, due):
    '''calculate TP, FP, TN, FN.'''
    TP = dee
    FP = due
    TN = ue - due 
    FN = ee - dee
    return TP, FP, TN, FN

def sens_spec(TP, FP, TN, FN):
    '''calculate sens and spec.'''
    sensitivity = TP/float(TP+FN)
    specificity = TN/float(TN+FP)
    precision = TP/float(TP+FP)
    return sensitivity, specificity, precision

def write_stats_to_file(in_fps, out_fp):
    '''write accumulated stats to an output file.'''
    header = '\t'.join(['identifier', 'TP', 'FP', 'TN', 'FN', 'sensitivity', 
        'specificity', 'precision'])
    lines = [header] + \
        ['\t'.join(map(str, parse_eco_eval_out(i))) for i in in_fps]
    o = open(out_fp, 'w')
    o.writelines('\n'.join(lines))
    o.close()

#######
# individual tables
# table_dir = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/'
# tables = glob.glob(table_dir+'*.txt')
# write_stats_to_file(tables, table_dir+'collated_output.txt')

# ensemble tables
# table_dir = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t1_6/'
# tables = glob.glob(table_dir+'*.txt')
# write_stats_to_file(tables, table_dir+'collated_output.txt')

# table_dir = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t1_7/'
# tables = glob.glob(table_dir+'*.txt')
# write_stats_to_file(tables, table_dir+'collated_output.txt')

table_dir = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t2_16/'
tables = glob.glob(table_dir+'*.txt')
write_stats_to_file(tables, table_dir+'collated_output.txt')

table_dir = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t2_17/'
tables = glob.glob(table_dir+'*.txt')
write_stats_to_file(tables, table_dir+'collated_output.txt')

table_dir = '/Users/wdwvt1/Desktop/work/co_occurrence/cooccurrence_evaluation_paper/specific_results/ecological_results/ensembles/t2_18/'
tables = glob.glob(table_dir+'*.txt')
write_stats_to_file(tables, table_dir+'collated_output.txt')