#!/usr/bin/env python
# File created 3/4/2013
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
from generators.lokta_volterra import (dX_dt_template, lokta_volterra)
from numpy import array, linspace
from numpy.testing import assert_array_almost_equal
from scipy import integrate


class TestLVGenerator(TestCase):
    
    def setUp(self):
        """Not all tests use the same data."""
        pass

    def test_dX_dt_template(self):
        '''Test that the function returned by dX_dt_template evals correctly.'''
        C = array([[.5, .6, .7],
                   [.1, .2, 1.5],
                   [6, .4, .4]])
        X = array([10,20,-6])
        exp = array([83,-156,-410.4])
        obs = dX_dt_template(C)(X)
        assert_array_almost_equal(obs, exp)

    def test_lokta_volterra(self):
        '''Test that LV is calculating correctly.
        WARNING: lokta_volterra function is being tested less rigorously than 
        other functions because hand calculating the answers for many steps 
        would be prohibitively time consuming. The function is being tested 
        by seeing if it agrees with the downloaded code from the scipy tutorial
        and by testing if it correctly evaluates fixed or stable points.
        scipy tutorial available at:
        http://www.scipy.org/Cookbook/LoktaVolterraTutorial.'''
        # from the scipy tutorial
        a = 1.
        b = 0.1
        c = 1.5
        d = 0.75
        def dX_dt(X, t=0):
            """ Return the growth rate of fox and rabbit populations. """
            return array([ a*X[0] -   b*X[0]*X[1] ,
                          -c*X[1] + d*b*X[0]*X[1] ])
        t = linspace(0, 15,  1000)
        X0 = array([10, 5])
        exp = integrate.odeint(dX_dt, X0, t)
        # calculate with our function
        C = array([[1., -.1],
                   [.075, -1.5]])
        f = dX_dt_template(C)
        Y = lokta_volterra(f, array([10,5]), 0, 15, 1000)
        assert_array_almost_equal(exp, Y.T)


if __name__ == "__main__":
    main()