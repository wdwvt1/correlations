#!/usr/env/bin/python

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"


'''This code for generating OTU vector pairs with given summary statistics but
different entries. A genetic algorithm is employed that was adapted directly 
from:
Generating Data with Identical Statistics but Dissimilar Graphics. Sangit 
Chatterjee and Aykut Firat 2007. The American Statistician 61:3, 248-254.
Small modifications were made to the mutation variance function to make it 
have a slower decline trajectory.'''


from numpy import (ones, dot, cov, array, argsort, searchsorted, vstack, 
    linspace, where, mean)
from numpy.linalg import norm
from numpy.random import randint, normal, shuffle
from scipy.linalg.matfuncs import sqrtm
from copy import deepcopy
from scipy.stats.distributions import norm as gaussian
# numpy.dot acts as dot product for 1D vectors, matrix multiplication for >=2D

def coerce_gene(X, X_star):
    """Coerces vector X to have same summary statistics as X_star.
    Using the process outlined in 'Generating Data with Identical Statistics but
    Dissimilar Graphics' by Sangit Chatterjee and Aykut Firat, this function
    takes a randomly drawn vector X and a vector X_star and produces a new 
    vector from X that has the same mean, std, correlation and possibly other 
    statistics as X_star. The reference is:
    Generating Data with Identical Statistics but Dissimilar Graphics. Sangit 
    Chatterjee and Aykut Firat 2007. The American Statistician 61:3, 248-254.
    Inputs:
     X, X_star - nX2 arrays.
     Reshaping is required because the way numpy handles matrix multiplication 
     elementwise unless it recognizes the array as 2D. 
    """
    # step ii: set mean value of columns of X to 0 using X=X-e_nx1*X_mean
    n = X.shape[0]
    X_new = X - dot(ones((n,1)),X.mean(0).reshape(1,2)) #a long col axis
    # step iii: orthonormalize cols of X_new with Gram-Schmidt process
    x, y = X_new[:,0], X_new[:,1]
    u1 = x
    u2 = y-(dot(x,y)/dot(x,x))*x
    e1, e2 = u1/norm(u1), u2/norm(u2)
    X_on = array([e1,e2]).T #orthnormalized X_new, .T to keep nX2
    # step iv: transform X_on to ensure summary stat agreement
    tmp_a = ((n-1.)**.5)*X_on
    tmp_b = sqrtm(cov(X_star.T)).astype(float)
    tmp_c = dot(ones((n,1)),X_star.mean(0).reshape(1,2))
    return dot(tmp_a, tmp_b)+tmp_c

def fitness(gene, ref_gene, method='graphic_dissimilarity'):
    """Calculates the fitness of a gene based on ref_gene and the method.
    Firat and Chatterjee calculated the fitness of genes in several different 
    ways as detailed in their paper.
    graphic_dissimilarity - g(gene, ref_gene) = sum over all i
    |x_i - x_star_i|+|y_i - y_star_i|. This attempts to maximize how different
    the graphics of the plotted gene and ref gene look.
    Inputs:
     gene, ref_gene = nX2 arrays.
    """
    if method=='graphic_dissimilarity':
        return abs(gene-ref_gene).sum()

def stochastic_uniform(arr, k):
    """Select k elements from arr where pr(i is selected) = arr[i]/arr.sum().
    This function implements what Firat and Chatterjee describe as the Matlab
    'stochastic uniform procedure'. In their words it: 
        'first lays out a line in which each parent corresponds to a 
         section of the of length proporitional to its scaled fitness value.
         Then the algorithm moves along the line in steps of equal size. At 
         each step, the algorithm allocates a parent from the section it 
         lands on.' pg 250. 
    Assumes arr has only positive values, otherwise will fail in an ugly 
    manner. 
    """
    line = [arr[:i].sum() for i in range(1,len(arr)+1)]
    return searchsorted(line, linspace(0, line[-1], k))

def cross_genes(gene1, gene2):
    """Breaks gene1 at a random point, appends head of gene1 to tail of gene2.
    See Figure 2 for a diagrammatic explanation."""
    i = randint(gene1.shape[0]) #gene1.shape = gene2.shape
    return vstack((gene1[:i,:], gene2[i:,:]))

def mutate_gene(gene, df_and_params):
    """Mutate gene by adding noise to each index with given variance.
    Avoid negative abundances.
    Inputs:
     gene - nX2 array.
     df_and_params - list, first entry is distribution function from scipy.stats
     .distributions and following entries are params in order.
    """
    noise = df_and_params[0].rvs(*df_and_params[1:], size=gene.shape)
    #m_gene = gene+(gene*noise)
    m_gene = gene+noise
    return where(m_gene>0.0, m_gene, 0.0)

def var_gen(generations):
    """Recursively generate variance weights. Return generator object."""
    var_0 = 1.0
    variances = [var_0]
    k = 1
    while len(variances)<generations+1:
        var_i = variances[-1]*(1.-.1*(k/float(generations)))
        variances.append(var_i)
        yield variances[k-1]
        k+=1


def selection(gene_pop, ref_gene, df_and_params, elite_children, 
    crossover_children, mutation_children, fitness_function):
    """Selects the next generation of genes based on the gene_pop fitness.
    This function assesses the fitness of each gene in the current population
    and then creates the next generation. The different fractions for children 
    in the inputs control how much of each type of child is created. 
    elite_children - these children are the most fit children from the current
    population. they will pass to the next generation unaltered. 
    crossover_children - these children will be formed by the crossover 
    function and the input genes will be controlled by the stochastic_uniform
    procedure (essentially sexual reproduction with mating chance controlled by 
    fitness value). 
    mutation_children - these children will be formed by mutating each element 
    of the parent gene at some frequency (essentially asexual reproduction). the
    input genes will be controlled by the stochastic_uniform procedure.
    Inputs:
    """
    pop_size = len(gene_pop)
    num_elites = int(pop_size*elite_children)
    num_crossovers = int(pop_size*crossover_children)
    num_mutations = int(pop_size*mutation_children)
    
    # assess fitness of population
    fitness_pop = array([fitness(gene, ref_gene, method=fitness_function) for 
        gene in gene_pop])

    # select elites, no mutation or cross overs required, these genes pass on
    elite_children = [gene_pop[i] for i in argsort(fitness_pop)[-num_elites:]]
    
    # select crosses, select one set of parents and then after shuffling the 
    # fitness_pop (the stochastic uniform process is deterministic if applied to 
    # the same data set) select a second set of parents. send these parent lists
    # to the cross_genes function.  
    parent1 = [gene_pop[i] for i in \
        stochastic_uniform(fitness_pop, num_crossovers)]
    cp_fitness_pop = deepcopy(fitness_pop)
    parent2 = [gene_pop[i] for i in \
        stochastic_uniform(cp_fitness_pop, num_crossovers)]
    crossover_children = [cross_genes(g1,g2) for g1,g2 in zip(parent1,parent2)]
    
    # select mutation children, first select parents, then use mutate_gene
    # to change the gene
    mc = [gene_pop[i] for i in stochastic_uniform(fitness_pop, num_mutations)]
    mutation_children = [mutate_gene(gene, df_and_params) for gene in mc]

    return elite_children, crossover_children, mutation_children

def evolve(inital_gene_pop, ref_gene, generations):
    """Evolve a gene population to maximized graphic dissimilarity.
    Convinience function with reduced number of params."""
    # initial, generation 0 children 
    gene_children = [coerce_gene(gene, ref_gene) for gene in inital_gene_pop]
    # make mutation variance generator
    vg = var_gen(generations)
    fitness_means = []
    fitness_maxs = []
    # run through generation number of selection cycles
    for gen in range(generations):
        # DEFINE ME
        df_and_params = [gaussian, 0, vg.next()]
        ec, cc, mc = selection(gene_children, ref_gene, df_and_params,
            elite_children=.02, crossover_children=.8, mutation_children=.18, 
            fitness_function='graphic_dissimilarity')
        # coerce cc and mc since elite children haven't been mutated or crossed
        # so their summary stats are still the same.
        cc = [coerce_gene(i, ref_gene) for i in cc]
        mc = [coerce_gene(i, ref_gene) for i in mc]
        gene_children = ec+cc+mc
        # shuffle may be unnecessary
        shuffle(gene_children)
        tmp =[fitness(i,ref_gene) for i in gene_children]
        fitness_means.append(mean(tmp))
        fitness_maxs.append(max(tmp))
    # final coerce step because gene_children have different summary stats
    #res = [coerce_gene(i,ref_gene) for i in gene_children]
    return gene_children, fitness_means, fitness_maxs

def select_fittest(genes, ref_gene, fitness_function, k):
    """Select the fittest k members of the population."""
    fitness_pop = array([fitness(gene, ref_gene, method=fitness_function) for 
        gene in genes])
    return fitness_pop, [genes[i] for i in argsort(fitness_pop)[-k:]]








