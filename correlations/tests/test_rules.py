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
from correlations.generators.rules import (model1_eval_rule, model1_eval_rules,
    model1_otu, model2_eval_rules, model2_otu)
from numpy import array, inf
from numpy.random import seed
from numpy.testing import assert_array_almost_equal

from scipy.stats.distributions import lognorm, uniform

class TestRulesGenerators(TestCase):
    
    def setUp(self):
        """Not all tests use the same data."""
        pass

    def test_model1_eval_rule(self):
        """Tests rules are evaluated correctly."""
        self.assertTrue(model1_eval_rule(0.0,1.0,.5))
        self.assertFalse(model1_eval_rule(0.0,1.0,5.))
        self.assertFalse(model1_eval_rule(-inf,0.0,0.0))
        self.assertTrue(model1_eval_rule(0,0,0))
        self.assertTrue(model1_eval_rule(0.,0,0))
        self.assertFalse(model1_eval_rule(0,0.,1))

    def test_model1_eval_rules(self):
        """Test that multiple rules are evaluated correctly."""
        rules = [[0.0,100.0], [30,40], [0.0,0.0]]
        vals = array([[0.,10.,101.,8.,0.,1.,0.,1.,2.,1.],
                      [1.,0.,4.,2.,9.,40.,6.,30.,2.,35.],
                      [0.,4.,9.,0.,0.,0.,2.,0.,0.,8.]])
        rs = [model1_eval_rules(rules, i) for i in vals.T] 
        assert_array_almost_equal(rs, array([2,1,0,2,2,2,1,3,2,2]))

    def test_model1_otu(self):
        """Test OTU is created correctly."""
        seed(0)
        weights = [1.0, .9, .8, 0.0]
        df_and_params = [lognorm, 2, 0]
        rules = [[0.0,100.0], [30,40], [0.0,0.0]]
        inducer_arr = array([[0.,10.,101.,8.,0.,1.,0.,1.,2.,1.],
                             [1.,0.,4.,2.,9.,40.,6.,30.,2.,35.],
                             [0.,4.,9.,0.,0.,0.,2.,0.,0.,8.]])
        exp = array([34,  2,  7, 88,  0,  0,  0,  0,  0,  2])
        assert_array_almost_equal(model1_otu(inducer_arr, df_and_params, 
            array(weights), rules), exp)

    def test_model2_eval_rules(self):
        """Tests model2 rules are evaluated correctly."""
        rules = [[0.4, 'sub'], [2.0, 'add'], [0.5, 'add']]
        inducer_arr = array([[0.,10.,101.,8.,0.,1.,0.,1.,2.,1.],
                             [1.,0.,4.,2.,9.,40.,6.,30.,2.,35.],
                             [0.,4.,9.,0.,0.,0.,2.,0.,0.,8.]])
        exp = array([2.0, 0, 0, 0.79999999999999982, 18.0, 79.599999999999994,
            13.0, 59.600000000000001, 3.2000000000000002, 73.599999999999994])
        obs = array([model2_eval_rules(rules, i) for i in inducer_arr.T])
        assert_array_almost_equal(obs, exp)

    def test_model2_otu(self):
        """Tests that model2 otus are created correctly."""
        rules = [[0.4, 'sub'], [2.0, 'add'], [0.5, 'add']]
        inducer_arr = array([[0.,10.,101.,8.,0.,1.,0.,1.,2.,1.],
                             [1.,0.,4.,2.,9.,40.,6.,30.,2.,35.],
                             [0.,4.,9.,0.,0.,0.,2.,0.,0.,8.]])
        out_otu=array([2.0, 0, 0, 0.79999999999999982, 18.0, 79.599999999999994,
            13.0, 59.600000000000001, 3.2000000000000002, 73.599999999999994])
        seed(0) #seed for unifrom random call
        urd = array([ 1.0097627 ,  1.04303787,  1.02055268,  1.00897664,  0.98473096,
        1.02917882,  0.98751744,  1.0783546 ,  1.09273255,  0.9766883 ])
        exp = urd*out_otu
        obs = model2_otu(inducer_arr, rules)
        assert_array_almost_equal(obs, exp)





if __name__ == "__main__":
    main()