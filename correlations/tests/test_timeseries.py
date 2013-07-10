#!/usr/bin/env python
# File created on 4 Mar 2013
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
from correlations.generators.timeseries import (add_noise, signal,
    superimpose_signals, make_otu, subsample_otu_random, subsample_otu_choose,
    subsample_otu_evenly, cube_d5_indices, subsample_otu_zero, 
    generate_otu_from_pt_in_R5, random_inds)
from numpy import pi, sin, cos, array, arange, where, hstack
from numpy.random import seed
from numpy.testing import assert_array_almost_equal
from scipy.stats.distributions import uniform
from scipy.signal import square, sawtooth

class TestTimeSeriesGenerator(TestCase):
    
    def setUp(self):
        """Define setup data."""

    def test_signal(self):
        '''Test that signal is generated correctly given various params.'''
        # test with a single numpy function 
        alpha = 1.
        phi = 2. #should make two cycles
        omega = .5*pi 
        signal_func = sin
        obs = signal(alpha, phi, omega, signal_func, 
            sampling_freq=25, lb=0, ub=2*pi)
        exp = array([  1.22464680e-16,  -5.00000000e-01,  -8.66025404e-01,
        -1.00000000e+00,  -8.66025404e-01,  -5.00000000e-01,
        -2.44929360e-16,   5.00000000e-01,   8.66025404e-01,
         1.00000000e+00,   8.66025404e-01,   5.00000000e-01,
         3.67394040e-16,  -5.00000000e-01,  -8.66025404e-01,
        -1.00000000e+00,  -8.66025404e-01,  -5.00000000e-01,
        -4.89858720e-16,   5.00000000e-01,   8.66025404e-01,
         1.00000000e+00,   8.66025404e-01,   5.00000000e-01,
         6.12323400e-16])
        assert_array_almost_equal(exp, obs)
        # test with a single scipy function
        alpha = 3.
        phi = 1.5
        omega = .25*pi 
        signal_func = sawtooth
        obs = signal(alpha, phi, omega, signal_func,
            sampling_freq=25, lb=0, ub=2*pi)
        exp = array([-1.875, -1.5  , -1.125, -0.75 , -0.375,  0.   ,  0.375,  0.75 ,
        1.125,  1.5  ,  1.875,  2.25 ,  2.625, -3.   , -2.625, -2.25 ,
       -1.875, -1.5  , -1.125, -0.75 , -0.375,  0.   ,  0.375,  0.75 ,
        1.125])
        assert_array_almost_equal(exp, obs)

    def test_add_noise(self):
        '''Test that noise is added predictably.'''
        # test uniform noise, seed for reproducibility
        inp_data = \
            array([-1.875, -1.5  , -1.125, -0.75 , -0.375,  0.,  0.375,  0.75 ,
                1.125,  1.5  ,  1.875,  2.25 ,  2.625, -3.   , -2.625, -2.25 ,
               -1.875, -1.5  , -1.125, -0.75 , -0.375,  0.   ,  0.375,  0.75 ,
                1.125])
        nfap = [uniform, -.1, .2]
        seed(0)
        exp = uniform.rvs(-.1, .2, size=25)+inp_data
        seed(0)
        obs = add_noise(nfap, inp_data)
        assert_array_almost_equal(exp, obs)

    def test_superimpose_signals(self):
        '''Test that signals are correctly superimposed.'''
        # test with one numpy and one scipy func, seed for reproducibility 
        seed(0)
        gen1 = [[8, 2, 0, sin, 25, 0, 2*pi],
                [8, 2, .5*pi, sawtooth, 25, 0, 2*pi]]
        # noise function and params as well as the general y_shift
        nfap = [uniform, -3, 6]
        gys = 20
        obs = superimpose_signals(gen1, gys, nfap)
        exp = array([ 20.29288102,  26.62446953,  30.21145015,  32.2692991 ,
        31.80346536,  31.54203135,  11.62552327,  11.68397134,
        10.52044   ,   7.30064911,  12.15548033,  14.84003619,
        20.40826737,  27.88691316,  27.02108625,  29.5227758 ,
        29.38284695,  32.66238574,  13.66894051,  11.55340622,
        10.61017349,   9.79495139,  10.17400628,  16.34984172,  17.70964656])
        assert_array_almost_equal(exp, obs)
        # test with a single function
        # [gen1[0]] required for superimpose to work since its expecting a list
        # of lists and the unwrap call will not function correctly unles it is.
        seed(0)
        obs = superimpose_signals([gen1[0]], gys, nfap)
        exp = array([ 20.29288102,  25.2911362 ,  27.54478349,  28.2692991 ,
        26.47013203,  24.87536468,  19.62552327,  18.350638  ,
        15.85377333,  11.30064911,  14.822147  ,  16.17336952,
        20.40826737,  26.55357983,  24.35441958,  25.5227758 ,
        24.04951361,  25.99571907,  21.66894051,  18.22007289,
        15.94350682,  13.79495139,  12.84067294,  17.68317506,  17.70964656])
        assert_array_almost_equal(exp, obs)

    def test_make_otu(self):
        '''Test that signals are added correctly.'''
        o1 = arange(-10,20)
        o2 = arange(30)
        obs = make_otu([o1, o2])
        exp = where(hstack([o1, o2]) > 0, hstack([o1, o2]), 0)
        assert_array_almost_equal(exp, obs)

    def test_subsample_otu_random(self):
        '''Test that samples are chosen correctly, reproducibly.'''
        seed(0)
        otu = array([ 0.29288102,  1.2911362 ,  0.61658026,  0.2692991 , -0.4580712 ,
        0.87536468, -0.37447673,  2.350638  ,  2.78197656, -0.69935089])
        exp_inds = array([1, 2, 4, 8, 9])
        exp_otu = array([ 1.2911362 ,  0.61658026, -0.4580712 ,  2.78197656, -0.69935089])

    def test_subsample_otu_choose(self):
        '''Test that otus chosen correctly.'''
        otu = arange(10)
        indices = [5,8,9]
        exp_otu = [5,8,9]
        obs_otu = subsample_otu_choose(otu, indices)
        assert_array_almost_equal(obs_otu, exp_otu)

    def test_subsample_otu_evenly(self):
        '''Test that samples are chosen correctly.'''
        # test with proper divisor
        otu = arange(10)
        fraction = .5
        exp_inds = array([0,3,5,7,9])
        exp_otu = otu[exp_inds]
        obs_otu = subsample_otu_evenly(otu, fraction)
        assert_array_almost_equal(obs_otu, exp_otu)

    def test_cube_d5_indices(self):
        '''Test the the hypercube in R5 is generated correctly.
        The order is as follows
            3  .    .    .    .    .
            2  .    .    .    .    .
            1  x    x    x    x    x
               d1   d2   d3   d4   d5

            3  .    .    .    .    .
            2  .    .    .    .    x
            1  x    x    x    x    .
               d1   d2   d3   d4   d5

            3  .    .    .    .    x
            2  .    .    .    .    .
            1  x    x    x    x    .
               d1   d2   d3   d4   d5
        
            3  .    .    .    .    .
            2  .    .    .    x    .
            1  x    x    x    .    x
               d1   d2   d3   d4   d5

            3  .    .    .    .    .
            2  .    .    .    x    x
            1  x    x    x    .    .
               d1   d2   d3   d4   d5
        '''
        # only going to test specific indices
        inds = [1,2,3]
        obs = cube_d5_indices(inds,inds,inds,inds,inds)
        self.assertEqual(obs[0], [1,1,1,1,1])
        self.assertEqual(obs[14], [1,1,2,2,3])
        self.assertEqual(obs[242], [3,3,3,3,3])
        self.assertEqual(obs[159], [2,3,3,3,1])

    def test_subsample_otu_zero(self):
        '''Tests zeroes placed in the right location.'''
        seed(0)
        otu = arange(20)
        ss_fraction = .5
        zero_fraction = .5
        exp_otu = subsample_otu_evenly(otu, ss_fraction)
        exp_otu[array([1, 2, 4, 8, 9])] = 0
        #exp_inds = array([0,2,4,6,8,10,12,14,16,18])
        #es_otu = otu[exp_inds]
        obs_otu = subsample_otu_zero(otu, ss_fraction, zero_fraction)
        assert_array_almost_equal(obs_otu, exp_otu)

    def test_generate_otu_from_pt_in_R5(self):
        '''Test that an OTU is correctly generated.'''
        seed(0)
        pt = (2, 10, 0, .5, [subsample_otu_evenly, .5])
        nfap = [uniform, -5, 10]
        y_shift = 100
        wave_f = sawtooth
        base_otu = 100 + signal(pt[1], pt[0], pt[2], wave_f)
        noisy_otu = add_noise(nfap, base_otu)
        sampling_f = subsample_otu_evenly
        exp_otu = sampling_f(noisy_otu, .5)
        seed(0)
        obs_otu = generate_otu_from_pt_in_R5(pt, wave_f, y_shift)
        assert_array_almost_equal(obs_otu, exp_otu)

    def test_random_inds(self):
        '''Test that random indices are returned predictably.'''
        seed(0)
        exp_inds = array([1, 2, 4, 8, 9])
        assert_array_almost_equal(random_inds(10, 5), exp_inds)

if __name__ == "__main__":
    main()