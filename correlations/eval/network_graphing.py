#!/usr/bin/env python

import networkx as nx
import matplotlib.pyplot as plt

def make_ensemble_networkx_graph(ro, bt, text=False, save=False, show=False,
                                 alpha=1.0, ensemble_index=0, ax=None):
    '''Make a networkx graph with default properties from parsed ro.

    Networkx graphs don't store the order of inputs so we have to add the edges 
    we want (nodes are added automatically) and then we need to step through the 
    lists of edges/nodes in the order they are given by the graph object and 
    apply the properties we want.
    '''
    G = nx.Graph()
    # position the nodes so that they are comparable across graphs
    G.add_nodes_from(bt.observation_ids)
    positions = nx.circular_layout(G)
    # Order of G.edges() != ro.edges(). This auto adds nodes as well. 
    G.add_edges_from(ro.edges)
    # get node sizes, the default is the mean of the otu abundance
    node_sizes = [bt.data(i, axis='observation').mean() for i in G.nodes()]
    # get edge colors, mutual exclusion is red, copresence is green
    # Order of G.edges() != ro.edges so we must build a map to ro order. To be
    # even more frustrating, G.edges() may reverse the order of edges.
    edge_map = []
    for edge in G.edges():
        try:
            edge_map.append(ro.edges.index(edge))
        except ValueError: #it wasn't in the list, must have been reversed
            edge_map.append(ro.edges.index(edge[::-1]))
    edge_colors = ['blue' if ro.cvals[ensemble_index][i] >= 0.0 else 'pink' 
                   for i in edge_map]
    # draw the figure using a custom axis
    fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)
    nx.draw(G, pos=positions, ax=ax, node_size=node_sizes, 
            edge_color=edge_colors, with_labels=False, node_color='gray',
            alpha=alpha)
    # title also adds legend and data
    if text:
        ax.set_title(text)
        ax.text(.95, 1.1, 'Copresences: %s' % ro.copresences(), color='g')
        ax.text(.95, 1.0, 'Exclusions: %s' % ro.exclusions(), color='r')
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    return G