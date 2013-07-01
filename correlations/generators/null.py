#!/usr/bin/env/python
#file created 2/22/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"

'''Code for creating data tables from null distributions of several types with
the goal of testing the FDR rates of the various tools. Several methods are 
implemented:

Method 1: OTU table is created by randomly drawing sample vectors from a given
distribution and parameters.

Method 2: OTU table is created with compositionality in mind and therefore the 
sum of each sample is constrained. Tables can be abundance or relative 
abundance and are produced by the Dirichlet distribution. 

Method 3: OTU is created with compositionality in mind ala model 2, but with 
higher sparsity than is normally created with the Dirichlet procedure by 
subtracting mean value of the table from all entries (entires < 0 -> 0).
'''

from numpy import array, where
from numpy.random import shuffle
from qiime.util import get_tmp_filename
import os

# command string used by R for methods 2 and 3
COMMAND_STR = \
"""#!/usr/bin/env Rscript
library('gtools');
prior_vals = %s
total_prior_knowledge = %s
prior_vals = total_prior_knowledge*(prior_vals/sum(prior_vals))
taxa = length(prior_vals)
samples = %s
sequencing_depth = round(%s)
d = matrix(0,nrow=taxa,ncol=samples)
for (i in 1:samples){
    pvs = rdirichlet(1,alpha=prior_vals)
    d[,i]=table(factor(sample(taxa,sequencing_depth,pvs,replace=TRUE),1:taxa))
}
write.table(d, file=%s, sep=',', quote=FALSE, col.names=FALSE, row.names=FALSE)
"""

def model1_otu(df_and_params, samples):
    """Return an otu vector drawn from df given params and of length samples."""
    return df_and_params[0].rvs(*df_and_params[1:],size=samples)

def model1_table(dfs_and_params, samples):
    """Return an OTU table drawn from given dfs and params."""
    return array([model1_otu(i,samples) for i in dfs_and_params])

def model2_table(otu_sums, samples, seq_depth, tpk):
    """Return OTU table drawn from dirichlet distribution with given params.
    Inputs:
     otu_sums - array of floats, weights you want to give to each otu in terms 
     of its probability mass for sampling. basically how large you want the sum
     of that otu to be compared to the others.
     samples - int, number of samples for each otu.
     seq_depth - the sum of each col of the data table, the number of 
     observations. 
     tpk - total prior knowledge. controls how spiky the distribution will be. 
     higher total prior knowledge will allow it to be much spikier which means
     less deviation away from otu_sums. 
    """
    prior_vals = otu_sums
    pvs_str =  'c(%s)' % ','.join(map(str,prior_vals))
    out_fp = get_tmp_filename()
    command_str = COMMAND_STR % (pvs_str, tpk, samples, seq_depth, out_fp)
    command_file = get_tmp_filename()
    o = open(command_file, 'w')
    o.write(command_str)
    o.close()
    os.system('R --slave < ' + command_file)
    o = open(out_fp)
    lines = map(str.rstrip ,o.readlines())
    o.close()
    return array([map(float,line.split(',')) for line in lines])

def model3_table(otu_sums, samples, seq_depth, tpk):
    """Uses model2_table but subtracts mean value to get higher sparsity."""
    data = model2_table(otu_sums, samples, seq_depth, tpk)
    d = data-data.mean()
    return where(d>0,d,0)


def alter_table(data, as_abund=True, as_int=False, sparsity=.8):
    """Change table to RA, to int, and/or add sparsity."""
    res = data
    if not as_abund:
        res = res/res.sum(0)
    if as_int:
        res = res.round(0)
    if sparsity:
        r,c = where(data==data)
        q = zip(r,c)
        shuffle(q)
        for i in range(int(len(q)*sparsity)):
            res[q[i]] = 0.
    return res
