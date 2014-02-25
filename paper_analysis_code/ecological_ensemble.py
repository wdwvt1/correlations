#!/usr/bin/env python

from collections import Counter

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

