
import matplotlib.pylab as plt
import utils.chronologies
import pandas
from pydendro.normalize import spline

from utils.config import base_path
from utils.compareFuns import get_window, moving_correlation
from utils.pcaFuns import pca 

chrons = utils.chronologies.published_chronologies(cut_eps=True)

cut_chrons = chrons.ix[1740:1780]
print cut_chrons

nyears = 10
smooth = lambda x: x / spline(x, nyears=nyears)

# f, ax = plt.subplots(5, sharex=True, sharey=True)

# #xmin, xmax, xtext = 1620, 2020, 2030
# xmin, xmax, xtext = 1725, 2020, 2030
# plt.xlim([xmin, xmax])
# for i in range(5):
#     ax[i].plot([xmin, xmax], [1.0, 1.0], color='0.5', linestyle='--')

# ax[0].plot(chrons.index, chrons.nc, color='grey')
# ax[1].plot(chrons.index, chrons.lh, color='grey')
# ax[2].plot(chrons.index, chrons.wd, color='grey')
# ax[3].plot(chrons.index, chrons.cc, color='grey')
# ax[4].plot(chrons.index, chrons.oc, color='grey')
# #ax[5].plot(chrons.index, chrons.ch)

# ax[0].plot(chrons.index, smooth(chrons.nc), color='k')
# ax[1].plot(chrons.index, smooth(chrons.lh), color='k')
# ax[2].plot(chrons.index, smooth(chrons.wd), color='k')
# ax[3].plot(chrons.index, smooth(chrons.cc), color='k')
# ax[4].plot(chrons.index, smooth(chrons.oc), color='k')

# for a in ax:
#   a.spines['top'].set_visible(False)
#   a.spines['bottom'].set_visible(False)
#   a.spines['right'].set_visible(False)
#   a.tick_params(bottom="off")
#   a.tick_params(top="off")
#   a.tick_params(right="off")

# ax[4].spines['bottom'].set_visible(True)
# a.tick_params(bottom="on")

# plt.subplots_adjust(right=0.8)  # make room for labels
# ax[0].text(xtext, 1.0, 'BM', verticalalignment='center')
# ax[1].text(xtext, 1.0, 'LH', verticalalignment='center')
# ax[2].text(xtext, 1.0, 'WD', verticalalignment='center')
# ax[3].text(xtext, 1.0, 'CC', verticalalignment='center')
# ax[4].text(xtext, 1.0, 'OC', verticalalignment='center')
# #ax[5].text(xtext, 1.0, 'Charlottesville', verticalalignment='center')


# ax[4].yaxis.set_ticks([0.6, 1.0, 1.4])

# f.subplots_adjust(hspace=-0.1)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# plt.xlabel('Year')

plt.figure()
#xmin, xmax, xtext = 1620, 2020, 2030
xmin, xmax, xtext = 1725, 2020, 2030
plt.xlim([xmin, xmax])

yvals = [i*0.9-0.4 for i in range(5)]
plt.ylim([0,yvals[4]+1.5])
plt.xlabel('Year')
plt.ylabel('Site growth index chronologies')

for i in range(5):
    plt.plot([xmin, xmax], [1.0+yvals[i], 1.0+yvals[i]], color='0.5', linestyle='--')

plt.plot(chrons.index, chrons.nc+yvals[0], color='0.4')
plt.plot(chrons.index, chrons.lh+yvals[1], color='0.4')
plt.plot(chrons.index, chrons.wd+yvals[2], color='0.4')
plt.plot(chrons.index, chrons.cc+yvals[3], color='0.4')
plt.plot(chrons.index, chrons.oc+yvals[4], color='0.4')
# plt.plot(chrons.index, chrons.ch)

plt.plot(chrons.index, smooth(chrons.nc)+yvals[0], color='0.1')
plt.plot(chrons.index, smooth(chrons.lh)+yvals[1], color='0.1')
plt.plot(chrons.index, smooth(chrons.wd)+yvals[2], color='0.1')
plt.plot(chrons.index, smooth(chrons.cc)+yvals[3], color='0.1')
plt.plot(chrons.index, smooth(chrons.oc)+yvals[4], color='0.1')

plt.subplots_adjust(right=0.8)  # make room for labels
plt.text(xtext, 1.0+yvals[0], 'BM', verticalalignment='center')
plt.text(xtext, 1.0+yvals[1], 'LH', verticalalignment='center')
plt.text(xtext, 1.0+yvals[2], 'WD', verticalalignment='center')
plt.text(xtext, 1.0+yvals[3], 'CC', verticalalignment='center')
plt.text(xtext, 1.0+yvals[4], 'OC', verticalalignment='center')

ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left="off")
ax.tick_params(right="off")
ax.tick_params(top="off")
ax.set_yticklabels([])

plt.savefig('plots/stacked_chrons.png')
plt.savefig('plots/stacked_chrons.pdf')

#plt.show()

pens = {
    'CC':  { 'marker': 'o', 'linestyle': '-', 'color': 'm' },
    'LH':  { 'marker': '^', 'linestyle': '-', 'color': 'r' },
    'OC':  { 'marker': 'v', 'linestyle': '-', 'color': 'g' },
    'WD':  { 'marker': 'd', 'linestyle': '-', 'color': 'b' },
    'BM':  { 'marker': '*', 'linestyle': '-', 'color': 'c' },
    'PC1': { 'marker': 's', 'linestyle': '-', 'color': 'k' },
}

def plot_moving_correlation(series, chrons, start, end, 
                            crit=None, window=get_window('boxcar', 31)):

  w = len(window) / 2
  t = range(start + w, end - w + 1)
  
  p = series.ix[start:end].values.flatten()
 
  for col in chrons:
    c = chrons[col].ix[start:end].values
    corrs, pvals = moving_correlation(p, c, window)
    plt.plot(t, corrs, label=col.upper(), markevery=(t[-1]-t[0])/12, **pens[col.upper()])

  if crit is not None:
    plt.axhline(y=crit, linestyle='--', color='black')  


precip = pandas.read_csv(base_path + 'csv/mjPrecip.csv', index_col=[0])
pdsiCook = pandas.read_csv(base_path + 'otherRecons/pdsi1.csv', index_col=[0], names=['pdsi'])

chrons['bm'] = chrons['nc']
del chrons['nc']

plt.figure()
plot_moving_correlation(pdsiCook, chrons, 1750, 2000)
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('Correlation')
plt.savefig('plots/cookPdsiRunningCorr.png')


S1, T1, E1, _ = pca(chrons, 1845, 1981)
chrons['pc1'] = S1

plt.figure()
plot_moving_correlation(precip, chrons, 1901, 1981, crit=0.355)
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('Correlation')
plt.savefig('plots/precipRunningCorr.png')




plt.show() 




