""" Plot the sample size by year for the set of chronologies used
    in the PCA.
"""

import matplotlib.pylab as plt
from pydendro.rwl import RWL

##############################################################################
# config

# base names of rwl files to build chronologies from
rwl_base_names = [ 'nc', 'lh', 'wd', 'cc', 'oc' ]
pub_base_names = rwl_base_names

rwl_names = [ (x, '../data/' + x + '.rwl') for x in rwl_base_names ]

###############################################################################

class Container():
  pass

samples = []
for name, fname in rwl_names:
  rwl = RWL(fname)
  for sample in rwl.samples:
    c = Container()
    c.chron = name
    c.fyog = sample.fyog
    c.lyog = sample.lyog
    samples.append(c)

def nsamples(samples, start, end, chrons):
  ss = []
  years = range(start, end+1)
  for year in years:
    ss.append(len([ x for x in samples if x.chron in chrons 
                                        and x.fyog <= year
                                        and year <= x.lyog ]))
  return years, ss

t1, ss1 = nsamples(samples, 1750, 1844, [ 'lh', 'wd', 'cc', 'oc' ])
t2, ss2 = nsamples(samples, 1845, 1981, [ 'nc', 'lh', 'wd', 'cc', 'oc' ])

plt.plot(t1, ss1, '--k')
plt.plot(t2, ss2, '-k')

plt.xlabel('year')
plt.ylabel('sample depth')
plt.xlim([1740, 1990])

plt.savefig('plots/sample_depth.png')
plt.savefig('plots/sample_depth.pdf')

plt.show()




