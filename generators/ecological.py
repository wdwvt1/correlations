#!/usr/bin/env/python
#file created 2/26/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"


"""
This code is meant to create tables with simple (ecologically based)
relationships between OTUs to test if the tools can accurately recapture 
relationships that are defined by a mechanism rather than by a high 
correlation score. The reason we chose this method is because we want a way to 
assess if relationships we know to exist in biological contexts can be revealed
through co-occurrence analysis as is frequently reported.

All interactions are linear and dependent on OTU abundance
0 = no effect, - = negative effect, + = positive effect
OTU1    OTU2    INTERACTION TYPE
 0       0        Neutralism 
 0       -        Amensalism  
 0       +        Commensalism
 +       +        Mutualism 
 +       -        Parasatism 
 -       -        Competition

The things that have a '_1d' in them are meant to operate on two otus. The
functions which have '_nd' operate on any number of otus where the last otu
is akin to otu2 and the preceeding otus are a network that collectively
represent otu 1. otus should be 1xn arrays or mxn arrays where m is the number
of otus and n is the number of samples (like a normal qiime otu table).

Some interaction types not covered by the above are implemented including 
obligate syntrophy (dependence of OTU2 on OTU1's presence) and ...
"""

from numpy import array, where, vstack

def amensal_1d(otu1, otu2, strength):
    '''Depress abundance of OTU2 when OTU1 is present by strength*OTU1.
    Models interaction type 'amensalism' where OTU1 is unaffected by the 
    presence of OTU2, but OTU2 is depressed by presence of OTU1.
    '''
    tmp = otu2 - otu1*strength
    return where(tmp>0,tmp,0)

def amensal_nd(otus, strength):
    '''Depress abundance of otus[-1] when otus[:-1] are present. 
    Strength of reduction is otus[:-1].mean()*strength for each index.'''
    tmp = otus[:-1].all(axis=0) #check that all otus[:-1] at each index > 0
    tmp_vals = strength*otus[:-1].mean(axis=0) #mean value used for subtraction
    depressed_vals = where(tmp == True, otus[-1] - tmp_vals, otus[-1])
    return where(depressed_vals > 0, depressed_vals, 0)

def commensal_1d(otu1, otu2, strength):
    '''Increase abundance of otu2 when otu1 is present by strength*otu1.
    Models interaction of type 'commensalism' where otu1 is unaffected by the
    presence of otu2, but otu2 is increased by the presence of otu1.'''
    tmp = vstack((otu1, otu2))
    tmp2 = tmp.all(axis=0) #check both present
    return where(tmp2 == True, otu2 + otu1*strength, otu2)

def commensal_nd(otus, strength):
    '''Increase abundance of otus[-1] when otus[:-1] are present. 
    Strength of increase is otus[:-1].mean()*strength for each index.'''
    tmp = otus[:-1].all(axis=0) #check that all otus[:-1] > 0 at each index
    tmp_vals = strength*otus[:-1].mean(axis=0) #mean value used for subtraction
    return where(tmp == True, otus[-1] + tmp_vals, otus[-1])

def mutual_1d(otu1, otu2, strength):
    '''Increase abundance of otu1 and otu2 when both are present.
    Models interaction type 'mutualism' where otus increase in abundance when
    both are present. Strength of increase proportional to the abundance of the
    other otu.'''
    tmp = vstack((otu1, otu2))
    tmp2 = tmp.all(axis=0) #check both present
    motu1 = where(tmp2 == True, otu1+otu2*strength, otu1)
    motu2 = where(tmp2 == True, otu2+otu1*strength, otu2)
    return motu1, motu2

def mutual_nd(otus, strength):
    '''Increase abundance of otus[-1] and otus[-1:] if all are present.
    Models a situation where a mutualist interaction occurs between otu[-1] 
    and the network formed by otus[:-1]. If not all the components of the 
    network are there then mutualism doesn't occur. Strength of increase for 
    otus[-1] proportional to average of network. Strength of increase for 
    otus[:-1] proportional to otus[-1].'''
    tmp = otus.all(axis=0) #check that all otus > 0 at each index
    o1 = where(tmp == True, otus[-1]+strength*otus[:-1].mean(axis=0), otus[-1])
    ntwrk_os = where(tmp == True, otus[:-1]+strength*otus[-1], otus[:-1])
    return vstack((ntwrk_os,o1)) #make input,output format the same

def parasite_1d(otu1, otu2, strength):
    '''Increase abundance of otu1, decrease abundance of otu2 when both present.
    Models interaction type 'parasatism' where otu1 grows/feeds/gains at the 
    expense of otu2. Strength is proportional to abuandance of other otu.'''
    tmp = vstack((otu1, otu2))
    tmp2 = tmp.all(axis=0) #check both present
    motu1 = where(tmp2 == True, otu1+otu2*strength, otu1)
    motu2 = where(tmp2 == True, otu2-otu1*strength, otu2)
    motu2 = where(motu2 > 0, motu2, 0)
    return motu1, motu2

def parasite_nd(otus, strength):
    '''Increase abundance of otus[-1] at expense of otus[:-1].
    Strength of increase proportional to abundance of other parasitized otu. 
    Strength of decrease is proportional to abundance of parasitizing otu.'''
    parasite_otu = otus[-1]
    parasitized_otus = []
    for otu in otus[:-1]: #otus that are parasatized
        parasite_otu, otu_i = parasite_1d(parasite_otu, otu, strength)
        parasitized_otus.append(otu_i)
    parasitized_otus.append(parasite_otu) #make output format same as input
    return array(parasitized_otus)

def competition_1d(otu1, otu2, strength):
    '''Depress abundance of both otus if both otus present. 
    Models interaction type 'competition' where otu1 and otu2 are both competing 
    for some limiting resource. Strength of decrease proportional to other otu.
    '''
    tmp = vstack((otu1, otu2))
    tmp2 = tmp.all(axis=0) #check both present
    motu1 = where(tmp2 == True, otu1-otu2*strength, otu1)
    motu2 = where(tmp2 == True, otu2-otu1*strength, otu2)
    motu1 = where(motu1 > 0, motu1, 0)
    motu2 = where(motu2 > 0, motu2, 0)
    return motu1, motu2

def competition_nd(otus, strength):
    '''Depress abundance of otu[-1] and otus[:-1] if all are present.
    Models a situation where a competitive interaction occurs between otu[-1] 
    and the network formed by otus[:-1]. If not all the components of the 
    network are there then competition doesn't occur. Strength of decrease for 
    otus[-1] proportional to average of network. Strength of decrease for 
    otus[:-1] proportional to otus[-1].'''
    tmp = otus.all(axis=0) #check co-presence
    o1 = where(tmp == True, otus[-1]-strength*otus[:-1].mean(axis=0), otus[-1])
    o1 = where(o1 > 0, o1, 0)
    ntwrk_os = where(tmp == True, otus[:-1]-strength*otus[-1], otus[:-1])
    ntwrk_os = where(ntwrk_os > 0, ntwrk_os, 0)
    return vstack((ntwrk_os,o1)) #make input,output format the same

def obligate_syntroph_1d(otu1, strength):
    '''Allow otu2 only when otu1 present at abudance proportional to strength.
    Models interaction type 'obligate syntrophy' where otu2 depends on the 
    presence of otu1 and can't exist without it. The abundance of otu2 in any 
    given sample will be proportional to strength*otu1.'''
    return otu1*strength

def obligate_syntroph_nd(otus, strength):
    '''Allow new otu only when all otus present. 
    Abundance proportional to strength*otus.mean().'''
    tmp = otus.all(axis=0)
    return where(tmp == True, strength*otus.mean(0), 0)




    


