#!/usr/bin/env python

from scipy.stats.distributions import lognorm
from numpy import linspace
from biom.table import table_factory
import os

def _generate_data(features, samples):
    '''Make featuresXsamples data matrix.'''
    return lognorm.rvs(3, 0, 1, size=features*samples).round(0).reshape(
        features, samples)

fmin, fmax = (10, 2000)
smin, smax = (10, 2000)
nsteps = 21

x = linspace(0, fmax, nsteps)
x[0] = fmin
y = linspace(0, smax, nsteps)
y[0] = smin

otu_ids = ['O_%s' % i for i in range(fmax)]
sample_ids = ['S_%s' % i for i in range(smax)]

out_dir = '/Users/wdwvt1/src/correlations/tables/timings/'

for num_features in x:
    for num_samples in y:
        data = _generate_data(num_features, num_samples)
        bt = table_factory(data, sample_ids[:int(num_samples)], 
            otu_ids[:int(num_features)])
        out_path = os.path.join(out_dir, 'table_f_%s_s_%s.biom' % 
            (num_features, num_samples))
        o = open(out_path, 'w')
        o.writelines(bt.getBiomFormatJsonString('will'))
        o.close()

'''
import glob
os.mkdir(os.join(out_dir, 'text_tables'))
tables = glob(out_dir+'*.biom')
for t in tables:
    out_fp = os.path.join(t.split('/')[:-1], 'text_tables/') + t.split('/')[-1]
    !biom convert -i $t -o $out_fp -b
'''


