"""Compare..."""

import matplotlib.pylab as plt
import matplotlib.ticker
import numpy as np
import pandas

from scipy.stats import pearsonr
from utils.compareFuns import moving_correlation, get_window

from utils.config import base_path

from pydendro.normalize import spline


# other recons nearby
pdsiCook = pandas.read_csv(base_path + 'otherRecons/pdsi1.csv', index_col=[0], names=['pdsi'])
vaPrecip = pandas.read_csv(base_path + 'otherRecons/vaPrecip.csv', index_col=[0])
drought  = pandas.read_csv(base_path + 'otherRecons/stahleDrought.csv', index_col=[0])
stahle   = pandas.read_csv(base_path + 'otherRecons/stahlePrecip.csv', index_col=[0], na_values = ['.'],
                           names=['Mean', 'NC', 'SC', 'GA' ], header=None)

# my recons
reconPrecip = pandas.read_csv(base_path + 'csv/reconPrecip.csv', index_col=[0])
reconPdsi   = pandas.read_csv(base_path + 'csv/reconPdsi.csv', index_col=[0])

#print reconPdsi.ix[1750:1800]

# weather data
pdsi   = pandas.read_csv(base_path + 'csv/jjPdsi.csv', index_col=[0])
precip = pandas.read_csv(base_path + 'csv/mjPrecip.csv', index_col=[0])

recon1 = reconPrecip #reconPdsi
recon2 = reconPdsi   #reconPdsi

start = 1750
end   = 1981

d1 = drought.ix[start:end].values.flatten() # JT
d2 = pdsiCook.ix[start:end].values.flatten() # NADA

#print stahle.ix[1800:end]

p1 = stahle['NC'].ix[start:end].values.flatten()
p2 = stahle['SC'].ix[start:end].values.flatten()
p3 = stahle['GA'].ix[start:end].values.flatten()
pm = stahle['Mean'].ix[start:end].values.flatten()
p4 = vaPrecip['Recon'].ix[start:end].values.flatten() #starts at 1784, end at 1966

r1 = recon1.ix[start:end].values.flatten()
r2 = recon2.ix[start:end].values.flatten()

# mat = np.vstack([r1, d1, d2, p1, p2, p3])#, p4])#, pdsi, precip])

# if start >= 1901:
#   pdsi = pdsi.ix[1901:1981].values.flatten()
#   precip = precip.ix[1901:1981].values.flatten()
  
# if start==1784:  
#   mat = np.vstack([r1, d1, d2, p1, p2, p3, p4])#, pdsi, precip])

# def corrMat(mat):
  
#   cols = mat.shape[0]
#   corr = np.zeros([cols, cols])
#   pval = np.zeros([cols, cols])
  
#   for i in range(cols):
#      for j in range(cols):
#        corr[i,j], pval[i,j] = pearsonr(mat[i,:], mat[j,:])
#   return corr, pval
   
# c,p = corrMat(mat)

# print c
# print p



nada = pandas.DataFrame(d2, index = range(start,end+1), columns=['pdsi'])
nadaMonthly = pandas.DataFrame({ x: d2 for x in ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] }, index = range(start,end+1))
nadaMonthly.to_csv('csv/nadaMonthly.csv', cols=['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

#corr, pval = pearsonr(v1, r1)

#print "corr1, pval1"
#print corr, pval

#corr, pval = pearsonr(v1, r2)
#print "corr2, pval2"
#print corr, pval

start = 1750
end   = 1981
t = range(start, end+1)

r1 = recon1.ix[start:end].values.flatten()
d2 = pdsiCook.ix[start:end].values.flatten()

start2 = 1784
end2   = 1966
t2 = range(start2, end2+1)
p4 = vaPrecip['Recon'].ix[start2:end2].values.flatten() #starts at 1784, end at 1966

rstd  = (r1 - np.mean(r1))/ np.std(r1)
d1std = (d1 - np.mean(d1))/ np.std(d1)
d2std = (d2 - np.mean(d2))/ np.std(d2)
p1std = (p1 - np.mean(p1))/ np.std(p1)
p4std = (p4 - np.mean(p4))/ np.std(p4)

r2std = rstd[start2-start:end2-end]

print len(r2std)
print len(p4)

print rstd.shape, d2std.shape
corr, pval = pearsonr(rstd, d2std)
print "corr2, pval2"
print corr, pval

# plot mj precip and cook pdsi

nyears = 10
smooth = lambda x: x / spline(x, nyears=nyears)

plt.figure()
plt.subplot(211)
plt.xlim([1745, 1985])
plt.plot(t, d2std, color='0.5', linestyle='-', label='Cook PDSI')
plt.plot(t, rstd, color='0.0', linestyle='-', label='MJ Precip')
plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
#plt.xlabel('Year')
plt.ylabel('Standardized value')
#plt.legend(loc='lower right')

ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left="off")
ax.tick_params(right="off")
ax.tick_params(top="off")

plt.subplot(212)
plt.xlim([1745, 1985])
# plt.plot(t, d2std/spline(d2std, nyears=10), color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t, rstd/spline(rstd, nyears=10), color='0.0', linestyle='-', label='MJ Precip')
plt.plot(t, smooth(d2std+10)-10, color='0.5', linestyle='-', label='Cook PDSI')
plt.plot(t, smooth(rstd+10)-10, color='0.0', linestyle='-', label='MJ Precip')
plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
plt.xlabel('Year')
plt.ylabel('Standardized value')
#plt.legend(loc='lower right')


ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(right="off")
ax.tick_params(top="off")

plt.savefig('plots/reconCompare.png')

########################################################################
# plot comparison with all recons...
########################################################################

plt.figure()


#plt.subplot(411)
plt.ylim([-16,20])
plt.xlim([1745, 1985])
plt.xlabel('Year')
plt.ylabel('Standardized reconstruction values')

plt.plot(t, 17+d2std, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t, 17+rstd, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, 13+smooth(d2std+10)-10, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t, 13+smooth(rstd+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')


plt.plot(t, 8+d1std, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t, 8+rstd, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, 4+smooth(d1std+10)-10, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t, 4+smooth(rstd+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')


plt.plot(t, -1+p1std, color='0.1', linestyle='-', label='Stahle NC')
plt.plot(t, -1+rstd, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, -5+smooth(p1std+10)-10, color='0.1', linestyle='-', label='Stahle NC')
plt.plot(t, -5+smooth(rstd+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')


plt.plot(t2, -10+p4std, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t2, -10+r2std, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t2, -14+smooth(p4std+10)-10, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t2, -14+smooth(r2std+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.xlabel('Year')

xtext = 1995

plt.subplots_adjust(right=0.8)  # make room for labels
plt.text(xtext, 15, 'NADA', verticalalignment='center')
plt.text(xtext, 6, 'JT', verticalalignment='center')
plt.text(xtext, -3, 'NC', verticalalignment='center')
plt.text(xtext, -12, 'MP', verticalalignment='center')

ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left="off")
ax.tick_params(right="off")
ax.tick_params(top="off")
ax.yaxis.set_ticks([])

#plt.tight_layout()

# fig = plt.gcf()
# fig.subplots_adjust(right=0.8, hspace=0.1)  # make room for labels

# plt.subplot(411)
# plt.ylim([-6,3])
# plt.xlim([1745, 1985])
# plt.plot(t, d2std, color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t, rstd, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# #plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

# plt.xlim([1745, 1985])
# plt.plot(t, -3+smooth(d2std+10)-10, color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t, -3+smooth(rstd+10)-10, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

# ax=plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(left="off")
# ax.tick_params(right="off")
# ax.tick_params(top="off")

# # plt.subplot(212)
# # plt.xlim([1745, 1985])
# # # plt.plot(t, dstd/spline(dstd, nyears=10), color='0.5', linestyle='-', label='Cook PDSI')
# # # plt.plot(t, rstd/spline(rstd, nyears=10), color='0.0', linestyle='-', label='MJ Precip')
# # plt.plot(t, smooth(dstd+10)-10, color='0.5', linestyle='-', label='Cook PDSI')
# # plt.plot(t, smooth(rstd+10)-10, color='0.0', linestyle='-', label='MJ Precip')
# # plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# # plt.xlabel('Year')
# # plt.ylabel('Standardized value')
# # #plt.legend(loc='lower right')


# # ax=plt.gca()
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.tick_params(right="off")
# # ax.tick_params(top="off")

# plt.subplot(412)
# plt.ylim([-6,3])
# plt.xlim([1745, 1985])
# plt.plot(t, d1std, color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t, rstd, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# #plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

# plt.xlim([1745, 1985])
# plt.plot(t, -3+smooth(d1std+10)-10, color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t, -3+smooth(rstd+10)-10, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

# plt.subplot(413)
# plt.ylim([-6,3])
# plt.xlim([1745, 1985])
# plt.plot(t, p1std, color='0.5', linestyle='-', label='Stahle NC')
# plt.plot(t, rstd, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# #plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

# plt.xlim([1745, 1985])
# plt.plot(t, -3+smooth(p1std+10)-10, color='0.5', linestyle='-', label='Stahle NC')
# plt.plot(t, -3+smooth(rstd+10)-10, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# #plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

# plt.subplot(414)
# plt.ylim([-6,3])
# plt.xlim([1745, 1985])
# plt.plot(t2, p4std, color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t2, r2std, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# #plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

# plt.xlim([1745, 1985])
# plt.plot(t2, -3+smooth(p4std+10)-10, color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t2, -3+smooth(r2std+10)-10, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# #plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

plt.savefig('plots/drought_recons_compare.png')
plt.savefig('plots/drought_recons_compare.pdf')

########################################################################
# plot moving correlation and pval
########################################################################

wlen = 31
t = range(1750 + wlen/2, 1981+1 - wlen/2)

diff, pval = moving_correlation(d2std, rstd, get_window('boxcar', wlen))
#sdiff, spval = moving_correlation(d2std, rstd, get_window('hann', 51))

f, ax = plt.subplots(2, sharex=True, sharey=False)
ax[0].plot(t, diff, '-k')
ax[1].plot(t, pval, '-k')
ax[1].axhline(0.05, linestyle='--', color='k')

ax[1].set_xlabel('Year')
#ax[1].xlabel('Year')
ax[0].set_ylabel('Correlation')
ax[1].set_ylabel('P-value')

#plt.ylabel('Correlation')
#plt.plot(t, sdiff, '-b')
plt.savefig('plots/reconRunningCorr.png')


#plt.figure()
#plt.plot(t, pval, '.k')
#plt.savefig('plots/reconRunningCorrPval.png')

#plt.plot(t, r2, '-b')
#plt.plot(t, v3, '-r')



# wet/dry

start = min(precip.index)
end   = max(reconPrecip.index)

fig, ax = plt.subplots(2, sharex=True, sharey=False)

y  = reconPrecip.ix[start:end]
t  = y.index
y1 = y.values.flatten()

z1 = np.polyval(np.polyfit(t, y1, 1), t)
d1 = y1[1:] - y1[:-1]

y  = precip.ix[start:end]
t  = y.index
y2 = y.values.flatten()

z2 = np.polyval(np.polyfit(t, y2, 1), t)
d2 = y2[1:] - y2[:-1]

a = (d2>0) & (d1>0)
b = (d2<0) & (d1<0)
ax[0].plot(t[1:], a | b, 'ok')
c = (a | b) == False
ax[0].plot(t[1:][c], len(t[1:][c])*[False], color='0.8', marker='o', linestyle='')
ax[0].set_ylim(-1, 2)
print len(t[1:]), len(t[1:][c])


# both the same
a = (y2>z2) & (y1>z1)
b = (y2<z2) & (y1<z1)
ax[1].plot(t, a | b, 'ok')
c = (a | b) == False

ax[1].plot(t[c], len(t[c])*[False], color='0.8', marker='o', linestyle='')

ax[1].set_ylim(-1, 2)
print len(t), len(t[c])


fig.subplots_adjust(hspace=0)
plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)

ax[1].set_xlabel('Year')

plt.xlim([start-2, end+2])

plt.savefig('plots/wetdry_dots.png')
plt.savefig('plots/wetdry_dots.pdf')


start = min(reconPrecip.index)
end   = max(reconPrecip.index)

fig, ax = plt.subplots(2, sharex=True, sharey=False)

y  = reconPrecip.ix[start:end]
t  = y.index
y1 = y.values.flatten()


z1 = np.polyval(np.polyfit(t, y1, 1), t)
ax[0].plot(t, z1, color='0.5')
ax[0].plot(t, y1, color='0.0')
ax[0].fill_between(t, z1, y1, where=y1>z1, facecolor='0.8', interpolate=True)

y  = precip.ix[start:end]
t  = y.index
y2 = y.values.flatten()

z2 = np.polyval(np.polyfit(t, y2, 1), t)
ax[1].plot(t, z2, color='0.5')
ax[1].plot(t, y2, color='0.0')
ax[1].fill_between(t, z2, y2, where=y2>z2, facecolor='0.8', interpolate=True)


fig.subplots_adjust(hspace=0)
plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)

ax[1].set_xlabel('Year')

plt.xlim([start-2, end+2])

plt.savefig('plots/wetdry.png')





# plot all precip recons

t = range(1750, 1981+1)
t2 = range(1784, 1966+1)
p4=vaPrecip['Recon'].ix[1784:1966].values.flatten() #starts at 1784, end at 1966

f, ax = plt.subplots(7, sharex=True, sharey=False)

xmin, xmax, xtext = 1745, 1985, 1995
#xmin, xmax, xtext = 930, 2010, 2025
plt.xlim([xmin, xmax])

def plot_recon(ax, series, label, ylim, yticks):
  y = series.values.flatten()
  t = series.index
  z = y / spline(y, nyears=10)
  ax.plot(t, y, linestyle='-', color='0.5')
  ax.plot(t, z, linestyle='-', color='0.0')  
  ax.text(xtext, yticks[1], label, verticalalignment='center')
  ax.yaxis.set_ticks(yticks)
  ax.set_ylim(ylim)
  

plot_recon(ax[0], recon1, 'mjPR', [40, 160], [60, 100, 140])
plot_recon(ax[1], pdsiCook, 'NADA', [-6, 6], [-4, 0, 4])
plot_recon(ax[2], stahle['NC'], 'NC', [150, 450], [200, 300, 400])
plot_recon(ax[3], stahle['SC'], 'SC', [200, 700], [250, 500, 750])
plot_recon(ax[4], stahle['GA'], 'GA', [200, 800], [250, 500, 750])
plot_recon(ax[5], vaPrecip['Recon'], 'MP', [-4, 4], [-2, 0, 2])
plot_recon(ax[6], drought, 'JC', [-6, 6], [-2, 0, 2])


plt.subplots_adjust(right=0.8, hspace=0.1)  # make room for labels
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
plt.xlabel('Year')
plt.savefig('plots/reconsStackedZoom.png')



plt.show()


