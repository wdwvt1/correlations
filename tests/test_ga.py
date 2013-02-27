#!/usr/bin/env python
# File created on 21 Feb 2013
from __future__ import division

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__version__ = "1.6.0-dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"
__status__ = "Development"


from shutil import rmtree
from os.path import exists, join
from cogent.util.unit_test import TestCase, main
from qiime.test import initiate_timeout, disable_timeout
from generators.ga import (coerce_gene, fitness, stochastic_uniform, 
    cross_genes, mutate_gene, var_gen, select_fittest)
from numpy import cov, array
from numpy.random import seed, normal
from numpy.testing import assert_array_almost_equal
from scipy.stats.distributions import lognorm

class TestGAGenerators(TestCase):
    
    def setUp(self):
        """Define setup data."""
        # Anscombes first vector as reported in Firat and Chatterjee
        self.anscombe_1 = array([[10,8,13,9,11,14,6,4,12,7,5],
            [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]]).T

    
    def test_coerce_gene(self):
        """Test that gene1 is coerced to have summary stats of gene2."""
        X_star = self.anscombe_1
        # X is drawn from lognorm(2,0,size=(2,5))
        X = array([[  6.27031733,   0.04223667],
                   [  0.07774097,   8.11511416],
                   [  0.61373009,   0.04524205],
                   [  0.74105751,  35.13672993],
                   [  0.45349999,   0.63991403],
                   [  0.3955539 ,   0.06494558],
                   [  0.17649918,   0.23867307],
                   [  0.04330733,   1.63722982],
                   [  2.62156629,   6.6979756 ],
                   [  0.12218307,   0.4649725 ],
                   [  0.03774385,   1.44339406]])

        expected_out = array([[ 17.28665683,  10.1867084 ],
                              [  7.70453594,   7.36262649],
                              [  7.71729605,   6.41679174],
                              [ 11.82029054,  12.16812971],
                              [  7.5121025 ,   6.4060161 ],
                              [  7.35037308,   6.27454792],
                              [  6.99902724,   6.15658985],
                              [  6.92863419,   6.29364546],
                              [ 11.85112874,   8.82937967],
                              [  6.93220682,   6.15692756],
                              [  6.89774806,   6.25863711]])
        actual_out = coerce_gene(X, X_star)
        assert_array_almost_equal(expected_out,actual_out)
        # check that means, standard deviations, and covariances are the same
        assert_array_almost_equal(X_star.mean(0), actual_out.mean(0))
        assert_array_almost_equal(X_star.std(0), actual_out.std(0))
        assert_array_almost_equal(cov(X_star.T), cov(actual_out.T))


    def test_fitness(self):
        """Test that fitness is calculated correctly for two vectors."""
        # test graphical_dissimilarity method
        # via numpy
        v1 = array([[1,2,3,4,5,6],[8,0.0,4,12,7,1.3]]).T
        v2 = array([[9,3.3,4,16.2,0.,2.1],[13.1,12.6,9.9,1.1,23.1,2.5]]).T
        exp_gd = 83.2
        self.assertFloatEqual(exp_gd,fitness(v1,v2,method='graphic_dissimilarity'))
        # via hand calculations
        v1 = array([[1,4,6],[12,3,2]]).T
        v2 = array([[2,3,7],[1,1,13]]).T
        exp_gd = 27
        self.assertFloatEqual(exp_gd,fitness(v1,v2,method='graphic_dissimilarity'))

    def test_stochastic_uniform(self):
        """Test that the stochastic_uniform function works properly."""
        # test via numpy
        seed(0) #seed for random reproducibility
        arr = abs(normal(size=10)) #fitness function is never negative so abs
        exp = array([0, 2, 3, 5, 9])
        obs = stochastic_uniform(arr, 5)
        assert_array_almost_equal(exp, obs)
        # test via hand calculations
        arr = array([.1, .4, .01, 3.1, .53, .4, .9, .2, .1])
        # [.1,.5,.51,3.61,4.14,4.54,5.44,5.64,5.74], pick 3 [0, 2.87,5.74]
        exp = array([0,3,8])
        obs = stochastic_uniform(arr, 3)
        assert_array_almost_equal(exp, obs)

    def test_cross_genes(self):
        """Test that genes are crossed as expected."""
        seed(0)
        v1 = array([[1,2,3,4,5,6],[8,0.0,4,12,7,1.3]]).T
        v2 = array([[9,3.3,4,16.2,0.,2.1],[13.1,12.6,9.9,1.1,23.1,2.5]]).T
        exp = array([[  1. ,   8. ],
                   [  2. ,   0. ],
                   [  3. ,   4. ],
                   [  4. ,  12. ],
                   [  0. ,  23.1],
                   [  2.1,   2.5]])
        obs = cross_genes(v1,v2)
        assert_array_almost_equal(exp,obs)

    def test_mutate_gene(self):
        """Test that gene is mutated correctly given passed params."""
        seed(0) #seed works for scipy as well
        v1 = array([[1,2,3,4,5,6],[8,0.0,4,12,7,1.3]]).T
        exp = array([[ 35.05935343,  10.22624079],
                       [  9.08143073,  88.39243583],
                       [ 44.89288403,   4.14162738],
                       [ 10.68707685,  12.73881006],
                       [  5.81347694,   9.27321926],
                       [  7.33387354,  19.63014576]])
        obs = mutate_gene(v1, [lognorm,2,0])
        assert_array_almost_equal(exp,obs)
    
    # hard to test this one since we keep making changes to suit how we want
    # the variance decrease to function  
    # def test_var_gen(self):
    #     """Test the variance generator works for 100 generations."""
    #     vg = var_gen(100)
    #     vals = 
    #     self.assertEqual(vals, [i for i in vg])

    # def test_select_fittest


if __name__ == "__main__":
    main()