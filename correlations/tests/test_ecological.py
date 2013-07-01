#!/usr/bin/env python
# File created on 23 Feb 2013
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
from correlations.generators.ecological import (amensal_1d, amensal_nd, 
    commensal_1d, commensal_nd, mutual_1d, mutual_nd, parasite_1d, parasite_nd, 
    competition_1d, competition_nd, obligate_syntroph_1d, obligate_syntroph_nd,
    partial_obligate_syntroph_1d, partial_obligate_syntroph_nd)
from numpy import array, where
from numpy.random import seed
from numpy.testing import assert_array_almost_equal
from scipy.stats.distributions import lognorm, uniform, beta, norm
from scipy.stats import kendalltau, spearmanr

class TestEcologicalGenerators(TestCase):
    
    def setUp(self):
        """Data used by several tests."""
        self.otu1 = array([0, 3, 10, 50, 6, 0, 10, 13.3, 12, 100])
        self.otu2 = array([9, 5, 100, 1, 2, 19, 0, 1, 80.54, 250])
        self.otus = \
            array([[ 33.,   8.,   2.,  20.,  12.,   9.,  11.,   3.,   4.,   0.],
                   [  2.,  13.,   6.,   0.,   2.,   1.,  16.,   0.,   2.,   3.],
                   [ 91.,   0.,   0.,   8.,   3.,   1.,   1.,   0.,   2.,   0.],
                   [100.,   0.,   7.,   6.,   6.,   8., 24.3,   9.,   5.,   7.]])

    def test_amensal_1d(self):
        """Tests amensal_1d subtracts the correct amount from otu2."""
        strength = .6
        exp = array([9, 3, 94, 0, 0, 19, 0, 0, 73, 190])
        obs = amensal_1d(self.otu1, self.otu2, strength).round(0)
        assert_array_almost_equal(obs, exp)

    def test_amednsal_nd(self):
        """Tests amensal_nd correctly subtracts correct amount from otu."""
        strength = .4
        exp = array([83, 0, 7, 6, 4, 7, 21, 9, 4, 7])
        obs = amensal_nd(self.otus, strength).round(0)
        assert_array_almost_equal(obs, exp)

    def test_commensal_1d(self):
        """Tests commensal_1d correctly adds to otu2."""
        strength = .4
        exp = array([9, 6, 104, 21, 4, 19, 0, 6, 85, 290])
        obs = commensal_1d(self.otu1, self.otu2, strength).round(0)
        assert_array_almost_equal(obs, exp)

    def test_commensal_nd(self):
        """Tests commensal_nd correctly adds to all otus."""
        strength = .3
        exp = array([113, 0, 7, 6, 8, 9, 27, 9, 6, 7])
        obs = commensal_nd(self.otus, strength).round(0)
        assert_array_almost_equal(obs, exp)

    def test_mutual_1d(self):
        """Tests that mutual_1d adds to both otus correctly."""
        strength = .7001 # because numpys rounding is terrible, see np.around
        exp_o1 = array([0,7,80,51,7,0,10,14,68,275])
        exp_o2 = array([9,7,107,36,6,19,0,10,89,320])
        obs_o1, obs_o2 = mutual_1d(self.otu1, self.otu2, strength)
        obs_o1, obs_o2 = obs_o1.round(0), obs_o2.round(0)
        assert_array_almost_equal(exp_o1, obs_o1)
        assert_array_almost_equal(exp_o2, obs_o2)

    def test_mutual_nd(self):
        """Tests that mutal_nd adds to all otus correctly."""
        strength = .2
        exp = array([
            [53,8,2,20,13,11,16,3,5,0],
            [22,13,6,0,3,3,21,0,3,3],
            [111,0,0,8,4,3,6,0,3,0],
            [108,0,7,6,7,9,26,9,6,7]])
        obs = mutual_nd(self.otus, strength).round(0)
        assert_array_almost_equal(obs, exp)

    def test_parasite_1d(self):
        """Test that the parsitic OTU takes from the other otu as expected."""
        strength = .501 # numpy is a pos
        exp_o1 = array([0,6,60,51,7,0,10,14,52,225])
        exp_o2 = array([9,3,95,0,0,19,0,0,75,200])
        obs_o1, obs_o2 = parasite_1d(self.otu1, self.otu2, strength)
        obs_o1 = obs_o1.round(0)
        obs_o2 = obs_o2.round(0)
        assert_array_almost_equal(exp_o1, obs_o1)
        assert_array_almost_equal(exp_o2, obs_o2)

    def test_parasite_nd(self):
        """Tests parasite operates on all OTUs correctly."""
        strength = .1
        exp = array([
            [23,8,1,19,11,8,9,2,4,0],
            [0,13,5,0,1,0,14,0,2,2],
            [81,0,0,7,2,0,0,0,2,0],
            [113, 0, 8, 9, 8, 9, 27, 9, 6, 7]])
        obs = parasite_nd(self.otus, strength)

    def test_competition_1d(self):
        """Test competition takes from both otus correctly."""
        strength = 1.0
        exp_o1 = array([0, 0, 0, 49, 4, 0, 10, 12, 0, 0])
        exp_o2 = array([9, 2, 90, 0, 0, 19, 0, 0, 69, 150])
        obs_o1, obs_o2 = competition_1d(self.otu1, self.otu2, strength)
        obs_o1, obs_o2 = obs_o1.round(0), obs_o2.round(0)
        assert_array_almost_equal(exp_o1, obs_o1)
        assert_array_almost_equal(exp_o2, obs_o2)

    def test_competition_nd(self):
        """Tests competition_nd is correctly evaluating the network."""
        strength = .6
        exp = array([
            [0,8,2,20,8,4,0,3,1,0],
            [0,13,6,0,0,0,1,0,0,3],
            [31,0,0,8,0,0,0,0,0,0],
            [75,0,7,6,5,7,18,9,4,7]])
        obs = competition_nd(self.otus, strength).round(0)

    def test_obligate_syntroph_1d(self):
        '''Test that obligate syntrophic relationship created.'''
        strength = .25
        exp = array([2,1,25, 0, 0, 5, 0, 0, 20, 62])
        obs = obligate_syntroph_1d(self.otu2, strength).round(0)
        assert_array_almost_equal(exp, obs)

    def test_obligate_syntroph_nd(self):
        '''Test that obligate syntrophic relationship created.'''
        strength = .4
        exp = array([23, 0, 0, 0, 2, 2, 5, 0, 1, 0])
        obs = obligate_syntroph_nd(self.otus, strength).round(0)
        assert_array_almost_equal(exp, obs)

    def test_partial_obligate_syntroph_1d(self):
        '''Test that partial obligate syntrophic relationship made correctly.'''
        o1 = array([0,0,0,12,6,10,1,0,0,0,0, 1,5,0,0,0,0,4,1,1])
        o2 = array([0,3,9,1, 5,0, 0,5,7,0,50,1,0,1,2,0,0,0,6,0])
        ex = array([0,0,0,1, 5,0, 0,0,0,0,0, 1,0,0,0,0,0,0,6,0])
        ob = partial_obligate_syntroph_1d(o1, o2)
        assert_array_almost_equal(ob, ex)

    def test_partial_obligate_syntroph_nd(self):
        '''Test partial obligate syntrophic relationship in nd.'''
        otus = array([[68,3, 0,1,0,31,4,0,0,5, 0, 1,  0,1,0,0,3, 5, 1,0],
                      [1, 34,0,0,1,6, 1,0,6,2, 4, 191,0,7,0,0,0, 0, 2,0],
                      [0, 3, 0,0,1,0, 1,0,0,24,1, 1,  0,2,2,1,26,1, 0,4],
                      [0, 1, 0,9,0,5, 3,0,1,0, 41,0,  4,0,8,0,4, 23,0,0]])
        exp = array([[68,3, 0,1,0,31,4,0,0,5, 0,1,  0,1,0,0,3, 5,1,0],
                     [1, 34,0,0,1,6, 1,0,6,2, 4,191,0,7,0,0,0, 0,2,0],
                     [0, 3, 0,0,1,0, 1,0,0,24,1,1,  0,2,2,1,26,1,0,4],
                     [0, 1, 0,0,0,0, 3,0,0,0, 0,0,  0,0,0,0,0, 0,0,0]])
        obs = partial_obligate_syntroph_nd(otus)
        assert_array_almost_equal(obs, exp)






if __name__ == "__main__":
    main()