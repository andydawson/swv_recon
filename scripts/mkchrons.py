"""Build chronologies using splines and save chronologies in
data/chronsGen as drop-in replacements for the chronologies in
data/chronsPub.
"""

import pandas

from pydendro.rwl import RWL
from pydendro.normalize import spline
from pydendro.chronology import build_chronology

##############################################################################
# config

from utils.chronologies import pub_base_names

nyears = { }


###############################################################################
# detrend each series individually (so we can use different nyears)

for name in pub_base_names:
  fname = 'data/%s.rwl' % name
  rwl = RWL(fname)
  for sample in rwl.samples:
    sample.widths = spline(sample.widths, nyears=nyears.get(name, 50))
  
  chron = build_chronology(rwl.samples)

  with open('data/chronsGen/%s' % name, 'w') as f:
      for year in chron.index:
          f.write(" %7d %7.4f\n" % (year, chron.chron[year]))


