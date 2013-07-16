#!/usr/bin/env python
# file created 6/30/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren, Sophie Weiss"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"

'''
This code is used for generating results and plotting graphs for the various
methods.
'''

from numpy import (array, bincount, arange, histogram, corrcoef, triu_indices,
    where, vstack, logical_xor, searchsorted, zeros, linspace, tril, ones,
    repeat, empty)
from matplotlib.pylab import matshow
from numpy.ma import masked_array
import matplotlib.pyplot as plt


def hist_of_metrics(data, method_strs):
    '''Plot histograms of each methods value distributions.'''
    plt.ioff()
    fig = plt.figure()
    num_figs = len(method_strs)
    for ind, method in enumerate(method_strs):
        sb = fig.add_subplot('%s1%s' % (num_figs,ind)) # row,col,which fig
        sb.hist(data[:,ind], bins=50)
        sb.set_xlabel('%s scores.' % method)
        sb.set_ylabel('Occurrences.')
    plt.tight_layout(pad=.4, h_pad=0)
    plt.show()

def plot_connection_abund(data):
    '''Plot connection abundance curve.'''
    plt.ioff()
    fig = plt.figure()
    sb = fig.add_subplot(111)
    sb.bar(left=arange(len(data)), height=data, width=1.0)
    sb.set_xlabel('Number of connections.')
    sb.set_ylabel('Number of OTUs with x connections.')
    plt.show()

def metric_metaval_corr(metric_vals, meta_vals):
    '''Returns pearson correlation between metric vals and meta_vals.'''
    return corrcoef(metric_vals, meta_vals)[0][1] # symmetric

def metric_metaval_spearman_corr(metric_vals, meta_vals):
    '''Returns pearson correlation between metric vals and meta_vals.'''
    return spearmanr(metric_vals, meta_vals)

def spearman_rho_metaval(rho_vals, meta_vals):
    '''Compare values extracted from rho matrix with meta_vals.'''
    return spearmanr(rho_vals, meta_vals)

def extract_from_rho(rho, edges):
    '''Extract values from the row matrix. edges of form ('o123','o456').'''
    # build vector of rho vals by walking through edges. all edges of the form
    # ('o123','o456')
    rho_vals = []
    for edge in edges:
        i = float(edge[0][1:]) #avoid 'o'
        j = float(edge[1][1:]) #avoid 'o'
        rho_vals.append(rho[i][j])
    return rho_vals

def plot_rho_edge_hist(rho, edges):
    '''Plot hist of all rho mat values versus those selected as significant.'''
    sig_rho_vals = extract_from_rho(rho, edges)
    inds = triu_indices(rho.shape[0], k=1) #offset to avoid main diag
    all_rho_vals = rho[inds[0],inds[1]]
    plt.ioff()
    fig = plt.figure()
    sb = fig.add_subplot(211)
    # want to display the histograms on top of one another with the same bin 
    # sizes to show what fraction of the significant edges are occuring in what
    # area of the initial rho matrix. thus, we save the bin edges.
    _, hist_bin_edges, _ = sb.hist(all_rho_vals, bins=100, 
        label='All rho matrix entries', color='y')
    sb.set_xlabel('Value')
    sb.set_ylabel('Occurrences')
    sb.legend()
    sb = fig.add_subplot(212)
    sb.hist(sig_rho_vals, bins=hist_bin_edges, color='g',
        label='Rho matrix entries selected as significant')
    sb.set_xlabel('Value')
    sb.set_ylabel('Occurrences')
    sb.legend()
    plt.show()

def node_stats(sig_nodes, bt):
    '''See if OTUs selected are statistically different than all OTUs.'''
    data = array([bt.observationData(i) for i in bt.ObservationIds])
    # all otu stats
    all_otu_mean, all_otu_std = data.mean(), data.std()
    all_otu_sparsity = (data == 0).sum()/float(data.size)
    # stats for otus not selected as sig
    ns_otus = list(set(bt.ObservationIds)-set(sig_nodes))
    if ns_otus == []:
        ns_mean, ns_std, ns_sparsity = 'NA', 'NA', 'NA'
    else:
        ns_data = data[array([bt.ObservationIds.index(i) for i in ns_otus])]
        ns_mean, ns_std = ns_data.mean(), ns_data.std()
        ns_sparsity = (ns_data == 0).sum()/float(ns_data.size)
    # stats for otus selected as sig
    s_data = data[array([bt.ObservationIds.index(i) for i in sig_nodes])]
    s_mean, s_std = s_data.mean(), s_data.std()
    s_sparsity = (s_data == 0).sum()/float(s_data.size)
    res = {'all_otu_mean':all_otu_mean, 'non_sig_otu_mean':ns_mean, 
        'sig_otu_mean':s_mean, 'all_otu_sparsity':all_otu_sparsity, 
        'non_sig_otu_sparsity':ns_sparsity, 'sig_otu_sparsity':s_sparsity,
        'all_otu_std':all_otu_std, 'non_sig_otu_std':ns_std, 
        'sig_otu_std':s_std}
    return res

def write_node_stats(results_obj, bt, out_fp):
    '''Write the result of node_stats(sig_nodes, bt).'''
    num_otus = len(bt.ObservationIds)
    connection_fraction = results_obj.connectionFraction(num_otus)
    copresences = results_obj.copresences()
    exclusions = results_obj.exclusions()
    average_connectivity = results_obj.avgConnectivity()
    node_stat_dict = node_stats(results_obj.sig_otus, bt)

    lines = [\
    'connection fraction:\t%s' % connection_fraction,
    'copresences:\t%s' % copresences, 
    'mutual exclusions:\t%s' % exclusions,
    'average connectivity:\t%s' % average_connectivity,
    'mean of all otus:\t%s' % node_stat_dict['all_otu_mean'],
    'mean of significant otus:\t%s' % node_stat_dict['sig_otu_mean'],
    'mean of non-significant otus:\t%s' % node_stat_dict['non_sig_otu_mean'],
    'std of all otus:\t%s' % node_stat_dict['all_otu_std'],
    'std of significant otus:\t%s' % node_stat_dict['sig_otu_std'],
    'std of non-significant otus:\t%s' % node_stat_dict['non_sig_otu_std'],
    'sparsity of all otus:\t%s' % node_stat_dict['all_otu_sparsity'],
    'sparsity of significant otus:\t%s' % node_stat_dict['sig_otu_sparsity'],
    'sparsity of non-significant otus:\t%s' % node_stat_dict['non_sig_otu_sparsity']]

    o = open(out_fp, 'w')
    o.writelines('\n'.join(lines))
    o.close()


def boxplot_connectivity_stats(connectivity_list, bt):
    '''See if node connectivity implies statistical difference.'''
    data = array([bt.observationData(i) for i in bt.ObservationIds])
    
    # this is grossly inelegant, it should be refactored. since the list is 
    # ordered, there should be a way to easily split it on number of connections
    # and prepare it for a boxplot. k[i-1] is not guaranteed to be different 
    # when i = 0 because all nodes may have equal numbers of connections, so 
    # we add the i == 0 check. we add the the total nodes so we can easily move
    # through the list in the box plots
    inds, vals = [], []
    for i,val in enumerate(connectivity_list):
        if val[1] != connectivity_list[i-1][1] or i == 0:
            inds.append(i)
            vals.append(val[1])
    inds.append(len(connectivity_list))
    #vals.append(connectivity_list[-1][1])

    mu_xs, std_xs, spar_xs  = [], [], []
    for i in range(1,len(inds)):
        node_vals = array([bt.observationData(k[0]) for k in \
            connectivity_list[inds[i-1]:inds[i]]])
        mu_xs.append(node_vals.sum(1))
        std_xs.append(node_vals.std(1))
        spar_xs.append((node_vals==0).sum(1)/float(node_vals.shape[1]))

    # plot without whiskers on the left and with whiskers on the right since
    # data is of massively different rnge, whiskers can cause serious display
    # problems
    fig = plt.figure()
    sb = fig.add_subplot(321)
    sb.boxplot(mu_xs, positions=vals, sym='')
    sb.set_xlabel('Number of Connections')
    sb.set_ylabel('Mean OTU Abundance')
    #sb.tick_params(axis='x', size='x-small')

    sb = fig.add_subplot(322)
    sb.boxplot(mu_xs, positions=vals, sym='b+')
    sb.set_xlabel('Number of Connections')
    sb.set_ylabel('Mean OTU Abundance')
    #sb.tick_params(axis='x', size='x-small')

    sb = fig.add_subplot(323)
    sb.boxplot(std_xs, positions=vals, sym='')
    sb.set_xlabel('Number of Connections')
    sb.set_ylabel('OTU Standard Deviation')
    #sb.tick_params(axis='x', size='x-small')

    sb = fig.add_subplot(324)
    sb.boxplot(std_xs, positions=vals, sym='b+')
    sb.set_xlabel('Number of Connections')
    sb.set_ylabel('OTU Standard Deviation')
    #sb.tick_params(axis='x', size='x-small')

    sb = fig.add_subplot(325)
    sb.boxplot(spar_xs, positions=vals, sym='')
    sb.set_xlabel('Number of Connections')
    sb.set_ylabel('OTU Sparsity')
    #sb.tick_params(axis='x', size='x-small')

    sb = fig.add_subplot(326)
    sb.boxplot(spar_xs, positions=vals, sym='b+')
    sb.set_xlabel('Number of Connections')
    sb.set_ylabel('OTU Sparsity')
    #sb.tick_params(axis='x', size='x-small')
    plt.show()

def ga_plot_edge_graphic_dissim(edges, bt, ref_gene):
    '''Plot the graphic dissimilarity of edges to ref gene.'''
    data = array([bt.observationData(i) for i in bt.ObservationIds])
    gds = []
    for edge in edges:
        gene = vstack((bt.observationData(edge[0]),bt.observationData(edge[1])))
        gds.append(fitness(gene, ref_gene))
    all_gds = []
    for i in range(data.shape[0]):
        for j in range(i+1,data.shape[0]):
            all_gds.append(fitness(vstack((data[i],data[j])),ref_gene))
    plt.ioff()
    fig = plt.figure()
    sb = fig.add_subplot(111)
    sb.hist(all_gds, bins=1000, cumulative=True, normed=True, histtype='step',
        label='All fitness', linewidth=1)
    sb.hist(gds,bins=1000, cumulative=True, normed=True, histtype='step',
        label='Significant edge fitness', linewidth=1)
    sb.set_xlim(0,1.1*max(all_gds))
    sb.set_ylim(0,1.1)
    sb.hlines(.5, sb.get_xlim()[0], sb.get_xlim()[1], linestyles='dashed', 
        alpha=.5, color='k', label='50% Accumulation')
    sb.set_xlabel('Fitness score')
    sb.set_ylabel('Cumulative fraction')
    sb.minorticks_on()
    plt.grid(True)
    plt.legend(loc=2)
    plt.show()

def ga_edge_even_odd(edges):
    '''Return: True if all edges are between even and odd OTUs (gene1 and gene2)
    '''
    return all([logical_xor(float(i[1:]) % 2, float(j[1:]) % 2) for i,j in edges])

def null_sig_node_locs(num_nodes, sig_nodes, start=0):
    '''Return location of OTUs in num_nodes.
    Assumes that num_nodes is a list like: [100,30,45,200] where each entry is 
    the maximum integer value in an OTUs name for it to have been created by
    a given method. 
    start is 1 or 0 depending on which OTU is the starting OTU number, i.e. o1 
    or o0. 
    '''
    #data = array([bt.observationData(i) for i in bt.ObservationIds])
    sn = array([float(i[1:]) for i in sig_nodes]) #avoid 'o'
    if start == 1:
        locs = searchsorted(array(num_nodes).cumsum(), sn, side='left')
    elif start == 0:
        locs = searchsorted(array(num_nodes).cumsum()-1, sn, side='left')
    return locs

def null_hist_sig_node_locs(locs, methods, num_nodes):
    '''Histogram of significant nodes and from which method they came.'''
    counts = bincount(locs)
    plt.ioff()
    fig = plt.figure()
    sb = fig.add_subplot(111)
    # plot background bars, total number of nodes of each type in the table
    # put the left position slightly offset to labels line up
    colors = [plt.cm.jet(i/float(len(methods))) for i in range(len(methods))]
    [sb.bar(i-.5, h, color=colors[i], width=1.0, alpha=.25) for i,h in enumerate(num_nodes)]
    [sb.bar(i-.5, h, color=colors[i], width=1.0, alpha=1.0) for i,h in enumerate(counts)]
    sb.set_xticks(arange(len(num_nodes)))
    sb.set_xticklabels(methods,rotation=90, size='x-small')
    sb.set_xlabel('Distributions')
    sb.set_ylabel('Counts of sig. nodes')
    plt.tight_layout() # fix cutting off the tic labels
    plt.show()

def null_edge_directionality(otu1, otu2, num_nodes):
    '''Calculate inter-method edge significance. 
    Returns a len(num_nodes)*len(num_nodes) matrix where the i,j entry is the
    number of significant edges whose first otu came from distribution i and 
    whose second otu came from distribution j.
    otu1 and otu2 are lists of the otus s.t. otu1[0],otu2[0] = edges[0]'''
    edge1 = null_sig_node_locs(num_nodes, otu1)
    edge2 = null_sig_node_locs(num_nodes, otu2)
    res = zeros((len(num_nodes), len(num_nodes)))
    # we need to force the matrix to be symmetric. because some i,j = (5,0) and
    # i,j = (0,5) are between the same two distributions we need to force this
    # to be upper triangular by always sorting the indices
    for i,j in zip(edge1,edge2):
        a,k = sorted((i,j))
        res[a][k] += 1
    return res

def null_plot_edge_directionality_heatmap(edge_dir_mat, methods):
    '''Plot heatmap of edge directionality with methods as labels.'''
    # make edge_dir_mat symmetric 
    #ed = edge_dir_mat+tril(edge_dir_mat.T,-1) #avoid main diag
    dim = edge_dir_mat.shape[0]
    ed = masked_array(edge_dir_mat, mask=tril(ones((dim,dim)),-1))
    plt.ioff()
    cmap = plt.cm.jet
    cmap.set_bad('w') # set masked entries to display as white
    matshow(ed, cmap=cmap)
    plt.xticks(arange(len(methods)), methods, rotation=90, fontsize=7)
    plt.yticks(arange(len(methods)), methods, fontsize=7)
    # bars = linspace(plt.xlim()[0], plt.xlim()[1],len(methods)+1)
    # plt.hlines(bars,bars[0],bars[-1])
    # plt.vlines(bars,bars[0],bars[-1])
    try:
        plt.tight_layout(h_pad=2, w_pad=2) # fix cutting off the tick labels
    except ValueError:
        pass
    plt.colorbar()
    plt.show()

def bt_stats(methods, num_nodes, bt):
    '''Return stats about OTUs with diff generators/methods.'''
    data = array([bt.observationData(i) for i in bt.ObservationIds])

    means = data.mean(1)
    stds = data.std(1)
    sparsity = (data==0).sum(1)/float(data.shape[1])
   
    vals = []
    i = 0
    for nn in num_nodes:
        vals.append([])
        for ind in range(i,i+nn):
            vals[-1].append([means[ind], stds[ind], sparsity[ind]])
        i+=nn
    return vals

def otus_from_edges_in_range(edges, lb, ub):
    '''Return all otus in edges that have lb <= OTUID <n.'''
    res = []
    for o1, o2 in edges:
        o1_float = float(o1[1:])
        o2_float = float(o2[1:])
        if lb <= o1_float < ub:
            res.append(o1)
        if lb <= o2_float < ub:
            res.append(o2)
    return list(set(res))

##
## Ecological 
##


def interacting_edges(start, stop, dim, edges, interactions):
    '''Check if an ecological edge is accurately detected.

    Edges will be of the form ('oX','oY') where X,Y are integers. The set of 
    all OTUs in a given table is assumed to be sequentially numbered from 0 
    (i.e. there exists an 'O0') to whatever integer. 

    Some of the generators create relationships between groups of OTUs (pairs, 
    triplets, etc.). In some cases these relationships are simple 
    'one-dimentional' relationships (e.g. OX^OY -> OY+=10), and in other cases
    they are two or more dimensions (e.g. OX^OY^...OZ -> OX,OY,...-=10). 

    This function calculates for each edge if its part of a pair generated with
    some relationship. In python, n mod m = n/m + n%m. It suffices to check that
    an edge has the same integer part to check that it came from the same
    relationship. There are two classes of relationships for dim >= 2 
    generators:
    (1) Both of the edge OTUs are on the left hand side (LHS) of the generator 
    rule. e.g. rule = OA^OB...OY -> OZ+=K, edge = (OX,OY) for X,Y in {A,B...Y}
    (2) One of the OTUs in the edge is in the RHS of the rule, the other is in 
    the LHS of the rule. 

    Direction of the edge is irrelevent for calculations of this function (i.e.
    (OX,OY)=(OY,OX). This function *does* assume that edges only appear once in 
    the input edges. 

    Inputs:
     start - int, starting index of OTU exhbiting relationship of interest
      (inclusive).
     stop - int, final index of OTU exhibiting relationship of interest 
      (exclusive).
     dim - int, number of OTUs in the LHS of the rule. 
     edges - list of OTU tuples
     interactions - list of strs, either mutualExclusion or copresence.
    '''
    # cis edges (LHS,LHS), trans edges (LHS,RHS or RHS,LHS), mes = mutual 
    # exclusions, cps = copresensces 
    cis_edges = 0
    trans_edges = 0
    total_detected = 0
    cis_mes = 0
    cis_cps = 0
    trans_mes = 0
    trans_cps = 0

    int_edges = map(lambda x: (int(x[0][1:]), int(x[1][1:])), edges)
    for ind, edge in enumerate(int_edges):
        o1, o2 = edge
        interaction = interactions[ind]
        if start <= o1 < stop and start <= o2 < stop:
            # calculate integer parts of o1,o2
            i1 = o1/(dim+1)
            i2 = o2/(dim+1)
            if i1 == i2: #same relationship
                total_detected+=1
                r1 = o1%(dim+1)
                r2 = o2%(dim+1)
                if r1 != dim and r2 != dim: #relationship is cis
                    cis_edges+=1
                    if interaction == 'copresence':
                        cis_cps+=1
                    else: #interaction == 'mutualExclusion'
                        cis_mes+=1
                else: # r1 == dim or r2 == dim
                    trans_edges+=1
                    if interaction == 'copresence':
                        trans_cps+=1
                    else: #interaction == 'mutualExclusion'
                        trans_mes+=1

    return (total_detected, cis_edges, cis_cps, cis_mes, trans_edges, trans_cps,
        trans_mes)


####################
####################
####################
####################
# ELIMINATED BY NEWER FUNCTIONS
# RETAINED FOR INTEROPERABILITY


# def eco_d2_counter(start, stop, edges, interactions):
#     '''Count the number of class1 and class 2 edges for a 2d eco relationship.
#     Inputs:
#      start - int, the index to start at for determining what edges should be 
#      considered. 
#      stop - int, the index to stop at for determining what edges should be
#      considered. 
#      edges - list of tuples, list of edges of form ('o1','o2).
#      interactions - list of strings, either 'copresence' or 'mutualExclusion'

#     class1 edges are edges of the form o1 - o2 or o2 - o1. for ecological 
#     relationships that are 2d, the otus will be grouped into blocks of 3. the 
#     first two otus act as a network, i.e. they given ecological relationship is
#     only triggered if both o1 and o2 are present. o3 is the other member of the
#     interaction, and also must be present for the relationship effect to be 
#     triggered. thus, class2 edges are o1 - o3, o3 - o1, o2 - o3, o3 - o2. 
#     '''
#     d2_i_0 = ['o%s' % s for s in arange(start,stop,3)]
#     d2_i_1 = ['o%s' % s for s in arange(start+1,stop,3)]
#     d2_i_2 = ['o%s' % s for s in arange(start+2,stop,3)]
#     # make classes of interest
#     class1 = zip(d2_i_0,d2_i_1) + zip(d2_i_1,d2_i_0)
#     class2 = zip(d2_i_0,d2_i_2) + zip(d2_i_2,d2_i_0) + zip(d2_i_1,d2_i_2) + \
#         zip(d2_i_2,d2_i_1)

#     class1_count = 0
#     class2_count = 0
#     class1_cop = 0
#     class2_cop = 0
#     class1_me = 0
#     class2_me = 0
#     total = (stop - start)/3. # total number of triplet otus generated

#     for interaction, edge in zip(interactions,edges):
#         if edge in class1:
#             class1_count += 1
#             if interaction == 'copresence':
#                 class1_cop += 1
#             elif interaction == 'mutualExclusion':
#                 class1_me += 1
#         elif edge in class2:
#             class2_count += 1
#             if interaction == 'copresence':
#                 class2_cop += 1
#             elif interaction == 'mutualExclusion':
#                 class2_me += 1
#     return (class1_count, class2_count, total, class1_cop, class1_me,
#         class2_cop, class2_me)

# def eco_d1_counter(start, stop, edges, interactions):
#     '''Like other eco counters. This one is for 1d (pairwise) relationships.'''
#     d2_i_0 = ['o%s' % s for s in arange(start,stop,2)]
#     d2_i_1 = ['o%s' % s for s in arange(start+1,stop,2)]
#     # make classes of interest. only one class because its a pairwise 
#     # interaction
#     class1_cop = 0
#     class1_me = 0
#     class1_count = 0

#     total = (stop - start)/2. # total num of doublet otus generated
#     class1 = zip(d2_i_0,d2_i_1) + zip(d2_i_1,d2_i_0)
#     for interaction,edge in zip(interactions,edges):
#         if edge in class1:
#             class1_count += 1
#             if interaction == 'copresence':
#                 class1_cop += 1
#             elif interaction == 'mutualExclusion':
#                 class1_me += 1
#     return class1_count, total, class1_cop, class1_me

# def eco_paired_between_class(edges, start, stop, dim):
#     '''Stats on 3 node graphs that are not cycles and edge nodes are related.
#     graphs have this structure: n1---n2---n3
#     n1 and n3 are the 'edge nodes', not to be confused with the edges. we are
#     searching for graphs s.t. n1 and n3 come from the same relationship and the
#     same doublet or triplet of related otus. example:
#     assume that o0-o89 are 2d related, i.e. o1^o2 -> o3. this function would 
#     select graphs of the form ox---oy---oz where y has no restriction, and 
#     x and z must be in {o1,o2,o3} or any integer multiple thereof. 
#     Inputs: 
#      edges - list of tuples, list of edges of form ('o1','o2)
#      start - int, the index to start at for determining what edges should be 
#      considered. 
#      stop - int, the index to stop at for determining what edges should be
#      considered. 
#      dims - the dimension of the method generating the otus. 
#     '''
#     # sort the edges based on the first otu

#     int_edges = [(int(i[1:]),int(j[1:])) for i,j in cr.edges]
#     sorted_edges = sorted(int_edges,key=itemgetter(0,1))

#     if dim == 1:
#         d_i_0 = ['o%s' % s for s in arange(start,stop,2)]
#         d_i_1 = ['o%s' % s for s in arange(start+1,stop,2)]
#         class1 = zip(d_i_0,d_i_1) + zip(d_i_1,d_i_0)

#         for edge in edges:
#             if edge in class1:
#                 for e2 in 



#     if dim == 2:
#         d_i_0 = ['o%s' % s for s in arange(start,stop,3)]
#         d_i_1 = ['o%s' % s for s in arange(start+1,stop,3)]
#         d_i_2 = ['o%s' % s for s in arange(start+2,stop,3)]
#         class1 = zip(d_i_0,d_i_1) + zip(d_i_1,d_i_0)
#         class2 = zip(d_i_0,d_i_2) + zip(d_i_2,d_i_0) + zip(d_i_1,d_i_2) + \
#             zip(d_i_2,d_i_1)

#     se = sorted(edges, key=lambda x:float(x[0][1:]))



################################################################################



def rmt_hist_of_metrics(data, method_str='Pearson'):
    '''Plot histograms of each methods value distributions for RMT.'''
    plt.ioff()
    fig = plt.figure()
    sb = fig.add_subplot(111)
    sb.hist(data, bins=50)
    sb.set_xlabel('%s scores.' % method_str)
    sb.set_ylabel('Occurrences.')
    plt.show()