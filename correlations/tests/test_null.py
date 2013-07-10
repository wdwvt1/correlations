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
from correlations.generators.null import (model1_otu, model1_table, 
    model2_table, model3_table, alter_table)
from numpy import array
from numpy.random import seed
from numpy.testing import assert_array_almost_equal
from scipy.stats.distributions import lognorm, beta, norm

# insert seed for the COMMAND_STR to get reproducbile results
import generators.null as gn
gn.COMMAND_STR = \
"""#!/usr/bin/env Rscript
library('gtools');
prior_vals = %s
total_prior_knowledge = %s
prior_vals = total_prior_knowledge*(prior_vals/sum(prior_vals))
taxa = length(prior_vals)
samples = %s
sequencing_depth = round(%s)
d = matrix(0,nrow=taxa,ncol=samples)
set.seed(0)
for (i in 1:samples){
    pvs = rdirichlet(1,alpha=prior_vals)
    d[,i]=table(factor(sample(taxa,sequencing_depth,pvs,replace=TRUE),1:taxa))
}
write.table(d, file=%s, sep=',', quote=FALSE, col.names=FALSE, row.names=FALSE)
"""

class TestNullGenerators(TestCase):
    
    def setUp(self):
        """Define setup data."""
        pass

    def test_model1_otu(self):
        """Tests that otus are created correctly according to given params."""
        seed(0) #seed for reproducibility
        exp = lognorm.rvs(2,0,size=10)
        seed(0)
        obs = model1_otu([lognorm, 2, 0], 10)
        assert_array_almost_equal(exp, obs)

    def test_model1_table(self):
        """Tests that otu table created reliably with model1."""
        seed(0)
        inp = [[lognorm,2,0],[beta,.5,1,2,10],[norm,0,10]]
        sams = 5
        exp = \
            array([[34.05935343,2.22624079,7.08143073,88.39243583,41.89288403],
                   [2.54744763, 2.0049072 , 4.14358164, 2.21392612, 2.21290674],
                   [-9.7727788,-25.52989816,6.53618595,8.64436199,-7.4216502 ]])
        obs = model1_table(inp, 5)
        assert_array_almost_equal(exp, obs)
    
    def test_model2_table(self):
        """Tests model2 table is created correctly."""
        # seed R at 0 by altering the command str
        exp = array([[33.,19.,53.,32.,4.,14.,17.,10.,9.,7.],
                     [17.,29.,10.,34.,7.,30.,5.,14.,68.,51.],
                     [10.,19.,16.,21.,54.,33.,43.,26.,12.,17.],
                     [40.,33.,21.,13.,35.,23.,35.,50.,11.,25.]])
        obs = model2_table([1,1.1,1.4,1.5],10,100,10)
        assert_array_almost_equal(exp, obs)
        # test with a random array to make sure sequencing depth (col sums) are
        # preserved
        obs = model2_table(abs(norm.rvs(size=10)),50,1000,100)
        self.assertTrue(all(obs.sum(0)==1000))

    def test_model3_table(self):
        """Tests model3 table is created correctly."""
        exp = array([[8.,0.,28.,7.,0.,0.,0.,0.,0.,0.],
                     [0.,4.,0.,9.,0.,5.,0.,0.,43.,26.],
                     [0.,0.,0.,0.,29.,8.,18.,1.,0.,0.],
                     [15.,8.,0.,0.,10.,0.,10.,25.,0.,0.]])
        obs = model3_table([1,1.1,1.4,1.5],10,100,10)
        assert_array_almost_equal(obs, exp)

    def test_alter_table(self):
        """Tests that table is not being altered incorrectly."""
        data = array([[8.,0.,28.,7.,0.,0.,0.,0.,0.,0.],
                     [0.,4.,0.,9.,0.,5.,0.,0.,43.,26.],
                     [0.,0.,0.,0.,29.,8.,18.,1.,0.,0.],
                     [15.,8.,0.,0.,10.,0.,10.,25.,0.,0.]])
        data_float = data + .46 #make float
        obs = alter_table(data_float, as_int=True, sparsity=0.0)
        assert_array_almost_equal(data, obs)
        obs = alter_table(data, as_abund=False, as_int=False, sparsity=0.0)
        exp = \
            array([[ 0.34782609,0.,1.,0.4375,0.,0.,0.,0.,0.,0.],
                   [0.,0.33333333,0.,0.5625,0.,0.38461538,0.,0.,1.,1.],
                   [0.,0.,0.,0.,0.74358974,0.61538462,0.64285714,0.03846154,0.,0.],
                   [0.65217391,0.66666667,0.,0.,0.25641026,0.,0.35714286,0.96153846,0.,0.]])
        assert_array_almost_equal(exp, obs)
        seed(0) #seed for reproducibility of sparsity calculations
        inp = array([[33.,19.,53.,32.,4.,14.,17.,10.,9.,7.],
                     [17.,29.,10.,34.,7.,30.,5.,14.,68.,51.],
                     [10.,19.,16.,21.,54.,33.,43.,26.,12.,17.],
                     [40.,33.,21.,13.,35.,23.,35.,50.,11.,25.]])
        exp = array([[ 33.,   0.,   0.,  32.,   0.,   0.,  17.,   0.,   0.,   7.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  51.],
       [  0.,  19.,   0.,  21.,   0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  11.,   0.]])
        assert_array_almost_equal(alter_table(inp,sparsity=.8), exp)


if __name__ == "__main__":
    main()