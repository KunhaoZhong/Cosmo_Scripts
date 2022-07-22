import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np

num_points_thin = 5000

analysissettings={'smooth_scale_1D':0.35,'smooth_scale_2D':0.35,'ignore_rows': u'0.5',
'range_confidence' : u'0.005'}

samples = loadMCSamples('./EXAMPLE_MCMC1',settings=analysissettings)

samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC1_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC2',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC2_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC3',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC3_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC4',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC4_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC5',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC5_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC6',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC6_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC7',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC7_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC8',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC8_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC9',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC9_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC10',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC10_THIN')

samples = loadMCSamples('./EXAMPLE_MCMC11',settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
samples.saveAsText('EXAMPLE_MCMC11_THIN')