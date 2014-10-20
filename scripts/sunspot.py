## plot the smoothed number of sunspots time series and 
## the smoothed reconstruction against year 

import pandas
import numpy as np
from scipy.stats import pearsonr

from pydendro.normalize import spline

import pylab as plt

precip = pandas.read_csv('../output/reconPrecip.csv', index_col=0)
ss     = pandas.read_fwf('../output/sunspot.fwf', widths=[4,4], index_col=0, names=['number'])

start = 1750
end   = 1981

precip = precip['recon'].ix[start:end].values
ss     = ss['number'].ix[start-1:end-1].values

precip = precip / spline(precip, nyears=5)
precip = precip - np.mean(precip)
precip = precip / np.std(precip)#abs(precip).max()
ss     = ss / spline(ss, nyears=5)
ss     = ss - np.mean(ss)
ss     = ss / np.std(ss)#abs(ss).max()

r, pval = pearsonr(precip, ss)

print r, pval

plt.plot(precip, '-k', linewidth=1.2)
plt.plot(ss, color='0.0', linestyle=':', linewidth=2)
plt.show()
