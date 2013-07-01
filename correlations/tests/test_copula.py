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
from generators.copula import (copula, scipy_corr, make_symmetric,
    generate_rho_matrix)
from numpy import array
from numpy.random import seed
from numpy.testing import assert_array_almost_equal
from scipy.stats.distributions import lognorm, uniform, beta, norm
from scipy.stats import kendalltau, spearmanr

class TestCopulaGenerators(TestCase):
    
    def setUp(self):
        """Not all tests use the same data."""
        pass

    def test_copula(self):
        """Test that the copula procedure works correctly."""
        seed(0)
        num_samples = 10
        mu_mat = [0,1,2]
        rho_mat = array([[1.,0.73080377,-0.60594323],
                         [0.73080377,1.,-0.72229433],
                         [-0.60594323,-0.72229433,1.]])
        methods = [[lognorm, 2, 0], [beta, 2, 5, 100, 20], [norm, 100, 25]]
        mv_draw = array([[-1.57785057, -0.93810752, -0.87473452, -0.60727304, -0.70750813,
         0.40230008, -0.09980138, -0.03676401, -2.62783557,  0.51672757],
       [-0.24491845, -1.41760817,  0.08143723,  1.18347231,  0.47052926,
         0.63433346, -0.2845943 ,  0.12327517, -1.09735385,  1.76061097],
       [ 3.90625252,  4.60643697,  2.73871773,  2.70413972,  2.80352243,
         2.92654096,  1.39205188,  2.81635277,  3.32713456,  2.81473476]])
        U = array([[ 0.05729997,  0.17409458,  0.19085919,  0.27183486,  0.23962538,
         0.6562684 ,  0.460251  ,  0.48533659,  0.0042965 ,  0.69732683],
       [ 0.40325979,  0.0781526 ,  0.53245288,  0.88168899,  0.68101153,
         0.73706839,  0.38797749,  0.5490554 ,  0.13624337,  0.96084786],
       [ 0.99995313,  0.99999795,  0.99691603,  0.99657593,  0.99747261,
         0.99828623,  0.91804666,  0.99757138,  0.99956128,  0.99755912]])
        otu_table = array([[  4.26085162e-02,   1.53168748e-01,   1.73866235e-01,
          2.96844727e-01,   2.42921665e-01,   2.23580235e+00,
          8.19056042e-01,   9.29110111e-01,   5.21784318e-03,
          2.81076064e+00],
       [  1.04478412e+02,   1.01610062e+02,   1.05571981e+02,
          1.09814565e+02,   1.07000976e+02,   1.07632716e+02,
          1.04353237e+02,   1.05719759e+02,   1.02217089e+02,
          1.12079784e+02],
       [  1.97656313e+02,   2.15160924e+02,   1.68467943e+02,
          1.67603493e+02,   1.70088061e+02,   1.73163524e+02,
          1.34801297e+02,   1.70408819e+02,   1.83178364e+02,
          1.70368369e+02]])
        assert_array_almost_equal(copula(num_samples, rho_mat, mu_mat, methods),
            otu_table)

    def test_generate_rho_matrix(self):
        """Test that rho matrix generated predictably."""
        seed(0)
        distribution = uniform
        params = [-.1,.2]
        num_otus = 5
        iters = 10
        obs = generate_rho_matrix(uniform, [-.1,.2], 5,10)
        exp = array([[ 1.,0.0722167,0.07889768,-0.0735975,0.08045463],
                     [0.0722167,1.,0.08413358,-0.00322377,0.03652002],
                     [0.07889768,0.08413358,1.,0.1516433,-0.09349692],
                     [-0.0735975,-0.00322377,0.1516433,1.,0.13010826],
                     [0.08045463,0.03652002,-0.09349692,0.13010826,1.]])
        assert_array_almost_equal(obs,exp)
        # test that the function fails when the params aren't conducive to 
        # creating a positive definite matrix. 
        seed(0)
        self.assertRaises(ValueError,generate_rho_matrix, uniform, [-.5,.5], 
            5, 10)

    def test_make_symmetric(self):
        """Test matrix is made symmetric and trace=1 if desired."""
        arr = array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
        exp = array([[0,5,10,15],[5,5,15,20],[10,15,10,25],[15,20,25,15]])
        obs = make_symmetric(arr, trace_1=False)
        assert_array_almost_equal(obs, exp)
        obs = make_symmetric(arr, trace_1=True)
        exp = array([[1,5,10,15],[5,1,15,20],[10,15,1,25],[15,20,25,1]])
        assert_array_almost_equal(obs, exp)

    def test_scipy_corr(self):
        """Test scipy_corr is moving through arr correctly."""
        arr = array([[0,5,10,15],[5,5,15,20],[10,15,10,25],[15,20,25,15]])
        exp = array([[1.,0.91287093,0.54772256,0.18257419],
                     [0.,1.,0.4,0.],
                     [0.,0.,1.,-0.4],
                     [0.,0.,0.,1.]])
        obs = scipy_corr(arr, kendalltau)
        assert_array_almost_equal(obs, exp)
        exp = array([[1.,0.9486833,0.63245553,0.10540926],
                     [0.,1.,0.5,-0.05555556],
                     [0.,0.,1.,-0.38888889],
                     [0.,0.,0.,1.]])
        obs = scipy_corr(arr, spearmanr)
        assert_array_almost_equal(obs, exp)



if __name__ == "__main__":
    main()