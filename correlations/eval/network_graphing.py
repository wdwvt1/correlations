#!/usr/bin/env python

import networkx as nx
import matplotlib.pyplot as plt

def make_ensemble_networkx_graph(ro, nodes, node_sizes, positions=None,
                                 text=False, save=False,
                                 show=False, alpha=1.0, ensemble_index=0,
                                 ax=None):
    '''Make a networkx graph with default properties from parsed ro.

    Networkx graphs don't store the order of inputs so we have to add the edges 
    we want (nodes are added automatically) and then we need to step through the 
    lists of edges/nodes in the order they are given by the graph object and 
    apply the properties we want.
    '''
    G = nx.Graph()
    # position the nodes so that they are comparable across graphs
    G.add_nodes_from(nodes)
    positions = nx.circular_layout(G)
    # Order of G.edges() != ro.edges(). This auto adds nodes as well. 
    G.add_edges_from(ro.edges)
    # node sizes is a dict since the order of nodes will be altered by G
    ns = [node_sizes[i] for i in G.nodes()]
    # get edge colors, mutual exclusion is red, copresence is green
    # Order of G.edges() != ro.edges so we must build a map to ro order. To be
    # even more frustrating, G.edges() may reverse the order of edges.
    edge_map = []
    for edge in G.edges():
        try:
            edge_map.append(ro.edges.index(edge))
        except ValueError: #it wasn't in the list, must have been reversed
            edge_map.append(ro.edges.index(edge[::-1]))
    edge_colors = ['green' if ro.cvals[ensemble_index][i] >= 0.0 else 'red' 
                   for i in edge_map]
    # draw the figure using a custom axis
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    nx.draw(G, pos=positions, ax=ax, node_size=ns, 
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