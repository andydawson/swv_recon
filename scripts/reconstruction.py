"""Plot reconstruction."""

import matplotlib.pylab as plt

import utils.chronologies
from   utils.pcaFuns import *
import utils.precipProxyFuns as precipProxyFuns
from utils.plot_recon import * 

cut_eps = True

chrons = utils.chronologies.published_chronologies(cut_eps=cut_eps)

# compute reconstruction

S1, T1, E1, _ = pca(chrons, 1845, 1981)
S2, T2, E2, _ = pca(chrons, 1750, 1981)

if cut_eps:
  #S2 = -S2
  S = S2.ix[1750:1844].append(S1.ix[1845:1981])
else:
  S = S2.ix[1750:1844].append(S1.ix[1845:1981])

#precipProxyFuns.precipProxy_var(S, 'precip')
#precipProxyFuns.precipProxy(S, 'precip')

plot_vals = pandas.read_csv(base_path + 'csv/plot_vals.csv', index_col=[0])
plot_vals_std = pandas.read_csv(base_path + 'csv/plot_vals_std.csv', index_col=[0])
#plot_pc_vs_precip(S, plot_vals, 'precip')
#plot_recon(S, plot_vals)
plot_paper_fig(S, plot_vals_std, 'precip')

plt.show()
