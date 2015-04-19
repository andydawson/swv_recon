"""Compare..."""

import matplotlib.pylab as plt
import matplotlib.ticker
import numpy as np
import pandas

from pprint import pprint

np.set_printoptions(linewidth=120)

from scipy.stats import pearsonr
from utils.compareFuns import moving_correlation, get_window

from utils.config import base_path

from pydendro.normalize import spline


# other recons nearby
# the one in the paper; from knmi climate explorer
nada_cook = pandas.read_csv(base_path + 'otherRecons/pdsi1.csv', index_col=[0], names=['pdsi'])
# from the NOAA website for NADAa 2004; gridpoint 247
nada247  = pandas.read_csv(base_path + 'otherRecons/nada_point247.csv', index_col=[0])
mp_precip = pandas.read_csv(base_path + 'otherRecons/va_precip_MP.csv', index_col=[0])
JT  = pandas.read_csv(base_path + 'otherRecons/stahle_drought_JT.csv', index_col=[0])
stahle   = pandas.read_csv(base_path + 'otherRecons/stahlePrecip.csv', index_col=[0], na_values = ['.'],
                           names=['Mean', 'NC', 'SC', 'GA' ], header=None)
maxwell_precip = pandas.read_csv(base_path + 'otherRecons/maxwellPrecip.csv', index_col=[0], na_values = ['.'])#, names=['Recon'], header=None)
maxwell_potomac = pandas.read_csv(base_path + 'otherRecons/maxwellPotomac.csv', index_col=[0], na_values = ['.'])#, names=['Recon'], header=None)

print nada247

# my recons
recon_precip = pandas.read_csv(base_path + 'csv/reconPrecip.csv', index_col=[0])
recon_pdsi   = pandas.read_csv(base_path + 'csv/reconPdsi.csv', index_col=[0])

#print reconPdsi.ix[1750:1800]

# weather data
pdsi   = pandas.read_csv(base_path + 'csv/jjPdsi.csv', index_col=[0])
precip = pandas.read_csv(base_path + 'csv/mjPrecip.csv', index_col=[0])

start = 1750
end   = 1981

# start = 1784
# end   = 1966

d_cook = nada_cook.ix[start:end].values.flatten()
d_247  = nada247.ix[start:end].values.flatten()
d_jt   = JT.ix[start:end].values.flatten()

p_nc   = stahle['NC'].ix[start:end].values.flatten()
p_sc   = stahle['SC'].ix[start:end].values.flatten()
p_ga   = stahle['GA'].ix[start:end].values.flatten()
p_mean = stahle['Mean'].ix[start:end].values.flatten()
p_mp   = mp_precip['Recon'].ix[start:end].values.flatten() #starts at 1784, end at 1966
p_max  = maxwell_precip['Recon'].ix[start:end].values.flatten()

sf_max = maxwell_potomac['Recon'].ix[start:end].values.flatten()

rprecip = recon_precip.ix[start:end].values.flatten()
rpdsi   = recon_pdsi.ix[start:end].values.flatten()

mat = np.vstack([rprecip, d_cook, d_247, d_jt, p_nc, p_sc, p_ga, p_max, sf_max])#, p4])#, pdsi, precip])

if start >= 1901:
  pdsi = pdsi.ix[1901:1981].values.flatten()
  precip = precip.ix[1901:1981].values.flatten()
  
if start==1784:  
 # mat = np.vstack([rprecip, d1, d2, p1, p2, p3, p4, m1, m2])#, pdsi, precip])
  mat = np.vstack([rprecip, d_cook, d_247, d_jt, p_nc, p_sc, p_ga, p_max, sf_max, p_mp])

def corrMat(mat):
  
  cols = mat.shape[0]
  corr = np.zeros([cols, cols])
  pval = np.zeros([cols, cols])
  
  for i in range(cols):
     for j in range(cols):
       corr[i,j], pval[i,j] = pearsonr(mat[i,:], mat[j,:])
  return corr, pval
   
c,p = corrMat(mat)

print c
print p



nada        = pandas.DataFrame(d_cook, index = range(start,end+1), columns=['pdsi'])
nadaMonthly = pandas.DataFrame({ x: d_cook for x in ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] }, index = range(start,end+1))
nadaMonthly.to_csv('csv/nadaMonthly.csv', cols=['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

start = 1750
end   = 1981
t = range(start, end+1)

rprecip = recon_precip.ix[start:end].values.flatten()
d2      = nada_cook.ix[start:end].values.flatten()

rstd = (rprecip - np.mean(rprecip))/ np.std(rprecip)
dstd = (d2 - np.mean(d2))/ np.std(d2)

print rstd.shape, dstd.shape
corr, pval = pearsonr(rstd, dstd)
print "corr2, pval2"
print corr, pval

# plot mj precip and cook pdsi

plt.figure()
plt.subplot(211)
plt.xlim([1745, 1985])
plt.plot(t, dstd, color='0.5', linestyle='-', label='Cook PDSI')
plt.plot(t, rstd, color='0.0', linestyle='-', label='MJ Precip')
plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
#plt.xlabel('Year')
plt.ylabel('Standardized value')
#plt.legend(loc='lower right')

plt.subplot(212)
plt.xlim([1745, 1985])
plt.plot(t, dstd/spline(dstd, nyears=10), color='0.5', linestyle='-', label='Cook PDSI')
plt.plot(t, rstd/spline(rstd, nyears=10), color='0.0', linestyle='-', label='MJ Precip')
plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
plt.xlabel('Year')
plt.ylabel('Standardized value')
#plt.legend(loc='lower right')
plt.savefig('plots/reconCompare.png')



# plot moving correlation and pval

wlen = 31
t = range(1750 + wlen/2, 1981+1 - wlen/2)

diff, pval = moving_correlation(dstd, rstd, get_window('boxcar', wlen))
#sdiff, spval = moving_correlation(dstd, rstd, get_window('hann', 51))

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

def rle(input_string, times):
    count = 1
    prev = None
    lst = []
    for i, character in enumerate(input_string):
        if character != prev:
            if prev is not None:
                entry = (prev,count,times[i])
                lst.append(entry)
                #print lst
            count = 1
            prev = character
        else:
            count += 1
    else:
        entry = (character,count,times[0])
        lst.append(entry)
    return lst

def wet_dry(recon):

  start = min(recon.index)
  end   = max(recon.index)

  if start < 1750:
    start = 1750

  if end > 1981:
    end = 1981

  fig, ax = plt.subplots(2, sharex=True, sharey=False)

  y  = recon.ix[start:end]
  t  = y.index
  y1 = y.values.flatten()

  z1 = np.polyval(np.polyfit(t, y1, 1), t)
  d1 = y1[1:] - y1[:-1]

  #yrs = y1 < z1
  yrs = y1 > z1
  # print y1 > z1
  # print "y1 < z1"
  # print t[yrs]
  # print d1
  foo = rle(yrs, t)
  pprint(foo)
  print("Dry spells")
  for i in range(1, len(foo)-1):
    if foo[i][2] > 1750 and foo[i][0] == False and foo[i][1] >= 3 and (foo[i-1][1] >= 2 or foo[i+1][1] >= 2):
      yr1 = foo[i][2]-foo[i][1]
      yr2 = foo[i][2]-1
      print foo[i], foo[i][2]-foo[i][1], foo[i][2]-1, y.ix[yr1:yr2].sum() #y.ix[yr1:yr2].values.flatten(), y.ix[yr1:yr2].sum()
  print("Wet spells")
  for i in range(1, len(foo)-1):
    if foo[i][2] > 1750 and foo[i][0] == True and foo[i][1] >= 3 and (foo[i-1][1] >= 2 or foo[i+1][1] >= 2):
      yr1 = foo[i][2]-foo[i][1]
      yr2 = foo[i][2]-1
      print foo[i], foo[i][2]-foo[i][1], foo[i][2]-1, y.ix[yr1:yr2].sum()


# d_cook = nada_cook.ix[start:end].values.flatten()
# d_247  = nada247.ix[start:end].values.flatten()
# d_jt   = JT.ix[start:end].values.flatten()

# p_nc   = stahle['NC'].ix[start:end].values.flatten()
# p_sc   = stahle['SC'].ix[start:end].values.flatten()
# p_ga   = stahle['GA'].ix[start:end].values.flatten()
# p_mean = stahle['Mean'].ix[start:end].values.flatten()
# p_mp   = mp_precip['Recon'].ix[start:end].values.flatten() #starts at 1784, end at 1966
# p_max  = maxwell_precip['Recon'].ix[start:end].values.flatten()

# sf_max = maxwell_potomac['Recon'].ix[start:end].values.flatten()

for r, l in [ (recon_precip, "RECON PRECIP"),
              (nada_cook, "NADA"),
              (JT, "JT"),
              (stahle['NC'], "NC"),
              (maxwell_precip['Recon'], "WV"),
              (maxwell_potomac['Recon'], "PR"),
              (mp_precip['Recon'], "MP") ]:
  print " >>> ", l, " <<< "
  wet_dry(r)


# start = min(precip.index)
# end   = max(recon_precip.index)

# y  = precip.ix[start:end]
# t  = y.index
# y2 = y.values.flatten()

# z2 = np.polyval(np.polyfit(t, y2, 1), t)
# d2 = y2[1:] - y2[:-1]



# a = (d2>0) & (d1>0)
# b = (d2<0) & (d1<0)
# ax[0].plot(t[1:], a | b, 'ok')
# c = (a | b) == False
# ax[0].plot(t[1:][c], len(t[1:][c])*[False], color='0.8', marker='o', linestyle='')
# ax[0].set_ylim(-1, 2)
# print len(t[1:]), len(t[1:][c])


# # both the same
# a = (y2>z2) & (y1>z1)
# b = (y2<z2) & (y1<z1)
# ax[1].plot(t, a | b, 'ok')
# c = (a | b) == False

# ax[1].plot(t[c], len(t[c])*[False], color='0.8', marker='o', linestyle='')

# ax[1].set_ylim(-1, 2)
# print len(t), len(t[c])


# fig.subplots_adjust(hspace=0)
# plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)

# ax[1].set_xlabel('Year')

# plt.xlim([start-2, end+2])

# plt.savefig('plots/wetdry_dots.png')
# plt.savefig('plots/wetdry_dots.pdf')





start = min(recon_precip.index)
end   = max(recon_precip.index)

#fig, ax = plt.subplots(2, sharex=True, sharey=False)
fig, ax = plt.subplots(1, sharex=True, sharey=False)

y  = recon_precip.ix[start:end]
t  = y.index
y1 = y.values.flatten()


z1 = np.polyval(np.polyfit(t, y1, 1), t)
ax.plot(t, z1, color='0.5')
ax.plot(t, y1, color='0.0')
ax.fill_between(t, z1, y1, where=y1>z1, facecolor='0.8', interpolate=True)

y  = precip.ix[start:end]
t  = y.index
y2 = y.values.flatten()

# z2 = np.polyval(np.polyfit(t, y2, 1), t)
# ax[1].plot(t, z2, color='0.5')
# ax[1].plot(t, y2, color='0.0')
# ax[1].fill_between(t, z2, y2, where=y2>z2, facecolor='0.8', interpolate=True)


# fig.subplots_adjust(hspace=0)
# plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)

#ax[1].set_xlabel('Year')
ax.set_xlabel('Year')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.tick_params(left="off")
ax.tick_params(right="off")
ax.tick_params(top="off")
#ax.yaxis.set_ticks([])

plt.xlim([start-2, end+2])

plt.savefig('plots/wetdry.png')





# plot all precip recons


t = range(1750, 1981+1)
t2 = range(1784, 1966+1)
p4=mp_precip['Recon'].ix[1784:1966].values.flatten() #starts at 1784, end at 1966

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
  

plot_recon(ax[0], recon_precip, 'mjPR', [40, 160], [60, 100, 140])
plot_recon(ax[1], nada_cook, 'NADA', [-6, 6], [-4, 0, 4])
plot_recon(ax[2], stahle['NC'], 'NC', [150, 450], [200, 300, 400])
plot_recon(ax[3], stahle['SC'], 'SC', [200, 700], [250, 500, 750])
plot_recon(ax[4], stahle['GA'], 'GA', [200, 800], [250, 500, 750])
plot_recon(ax[5], mp_precip['Recon'], 'MP', [-4, 4], [-2, 0, 2])
plot_recon(ax[6], JT, 'JT', [-6, 6], [-2, 0, 2])


plt.subplots_adjust(right=0.8, hspace=0.1)  # make room for labels
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
plt.xlabel('Year')
plt.savefig('plots/reconsStackedZoom.png')

########################################################################
# plot comparison with all recons...
########################################################################

start = 1750
end   = 1981
t = range(start, end+1)

rprecip = recon_precip.ix[start:end].values.flatten()
d_cook  = nada_cook.ix[start:end].values.flatten()

start2 = 1784
end2   = 1966
t2 = range(start2, end2+1)
p_mp = mp_precip['Recon'].ix[start2:end2].values.flatten() #starts at 1784, end at 1966

rprecip_std = (rprecip - np.mean(rprecip))/ np.std(rprecip)
d_cook_std  = (d_cook - np.mean(d_cook))/ np.std(d_cook)
d_jt_std    = (d_jt - np.mean(d_jt))/ np.std(d_jt)
p_nc_std    = (p_nc - np.mean(p_nc))/ np.std(p_nc)
p_max_std    = (p_max - np.mean(p_max))/ np.std(p_max)
sf_max_std    = (sf_max - np.mean(sf_max))/ np.std(sf_max)
p_mp_std    = (p_mp - np.mean(p_mp))/ np.std(p_mp)

rprecip_std_cut = rprecip_std[start2-start:end2-end]

print len(rprecip_std_cut)
print len(p_mp_std)

# print rstd.shape, d2std.shape
# corr, pval = pearsonr(rstd, d2std)
# print "corr2, pval2"
# print corr, pval

# plot mj precip and cook pdsi

nyears = 10
smooth = lambda x: x / spline(x, nyears=nyears)

# plt.figure()
# plt.subplot(211)
# plt.xlim([1745, 1985])
# plt.plot(t, d2std, color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t, rstd, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# #plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')

# ax=plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(left="off")
# ax.tick_params(right="off")
# ax.tick_params(top="off")

# plt.subplot(212)
# plt.xlim([1745, 1985])
# # plt.plot(t, d2std/spline(d2std, nyears=10), color='0.5', linestyle='-', label='Cook PDSI')
# # plt.plot(t, rstd/spline(rstd, nyears=10), color='0.0', linestyle='-', label='MJ Precip')
# plt.plot(t, smooth(d2std+10)-10, color='0.5', linestyle='-', label='Cook PDSI')
# plt.plot(t, smooth(rstd+10)-10, color='0.0', linestyle='-', label='MJ Precip')
# plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')
# plt.xlabel('Year')
# plt.ylabel('Standardized value')
# #plt.legend(loc='lower right')


# ax=plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(right="off")
# ax.tick_params(top="off")

# plt.savefig('plots/reconCompare.png')

plt.figure()


#plt.subplot(411)
plt.ylim([-34,20])
plt.xlim([1745, 1985])
plt.xlabel('Year')
plt.ylabel('Standardized reconstruction values')

plt.plot(t, 17+d_cook_std, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t, 17+rprecip_std, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, 13+smooth(d_cook_std+10)-10, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t, 13+smooth(rprecip_std+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')


plt.plot(t, 8+d_jt_std, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t, 8+rprecip_std, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, 4+smooth(d_jt_std+10)-10, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t, 4+smooth(rprecip_std+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')


plt.plot(t, -1+p_nc_std, color='0.1', linestyle='-', label='Stahle NC')
plt.plot(t, -1+rprecip_std, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, -5+smooth(p_nc_std+10)-10, color='0.1', linestyle='-', label='Stahle NC')
plt.plot(t, -5+smooth(rprecip_std+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, -10+p_max_std, color='0.1', linestyle='-', label='Maxwell Precip')
plt.plot(t, -10+rprecip_std, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, -14+smooth(p_max_std+10)-10, color='0.1', linestyle='-', label='Maxwell Precip')
plt.plot(t, -14+smooth(rprecip_std+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, -19+sf_max_std, color='0.1', linestyle='-', label='Potomac Streamflow')
plt.plot(t, -19+rprecip_std, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t, -23+smooth(sf_max_std+10)-10, color='0.1', linestyle='-', label='Potomac Streamflow')
plt.plot(t, -23+smooth(rprecip_std+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')


plt.plot(t2, -28+p_mp_std, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t2, -28+rprecip_std_cut, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.plot(t2, -32+smooth(p_mp_std+10)-10, color='0.1', linestyle='-', label='Cook PDSI')
plt.plot(t2, -32+smooth(rprecip_std_cut+10)-10, color='0.4', linestyle='-', label='MJ Precip')
#plt.hlines(0, 1745, 1985, colors='grey', linestyles='solid')

plt.xlabel('Year')

xtext = 1995

plt.subplots_adjust(right=0.8)  # make room for labels
plt.text(xtext, 15, 'NADA', verticalalignment='center')
plt.text(xtext, 6, 'JT', verticalalignment='center')
plt.text(xtext, -3, 'NC', verticalalignment='center')
plt.text(xtext, -12, 'WV', verticalalignment='center')
plt.text(xtext, -21, 'PR', verticalalignment='center')
plt.text(xtext, -30, 'MP', verticalalignment='center')

ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left="off")
ax.tick_params(right="off")
ax.tick_params(top="off")
ax.yaxis.set_ticks([])

plt.savefig('plots/drought_recons_compare.png')
plt.savefig('plots/drought_recons_compare.pdf')


#plt.show()
