import matplotlib.pylab as plt
import pandas
from utils.config import base_path
import math

nada   = pandas.read_fwf(base_path + 'csv/spectral/nada_spectral_cut.txt', widths=[12,12,12,12],  header=None, names=['freq', 'power', 'sig', 'sigp'])  
nada.name='nada'                         

potomac   = pandas.read_fwf(base_path + 'csv/spectral/maxwell_potomac_spectral_cut.txt', widths=[12,12,12,12],  header=None, names=['freq', 'power', 'sig', 'sigp'])
potomac.name='potomac'

maxwell_precip = pandas.read_fwf(base_path + 'csv/spectral/maxwell_precip_spectral_cut.txt', widths=[12,12,12,12],  header=None, names=['freq', 'power', 'sig', 'sigp'])
maxwell_precip.name='maxwell_precip'

stahle_drought = pandas.read_fwf(base_path + 'csv/spectral/stahle_drought_spectral_cut.txt', widths=[12,12,12,12],  header=None, names=['freq', 'power', 'sig', 'sigp'])
stahle_drought.name='stahle_drought'

NC = pandas.read_fwf(base_path + 'csv/spectral/stahle_precip_NC_spectral_cut.txt', widths=[12,12,12,12],  header=None, names=['freq', 'power', 'sig', 'sigp'])
NC.name='NC'

MP = pandas.read_fwf(base_path + 'csv/spectral/va_precip_MP_spectral.txt', widths=[12,12,12,12],  header=None, names=['freq', 'power', 'sig', 'sigp'])
MP.name='MP'

recon   = pandas.read_fwf(base_path + 'csv/spectral/recon_spectral.txt', widths=[12,12,12,12],  header=None, names=['freq', 'power', 'sig', 'sigp'])
recon.name='recon'

#print 1.0/recon.freq, 1.0/nada.freq    

sites = [recon, nada, potomac, maxwell_precip, stahle_drought, NC, MP]
#sites = [recon, nada, potomac]

for site in sites:
  idx=site.power > site.sig
  print site.name
  print 1.0/site.freq[idx]


# #plt.semilogx(1.0/nada['freq'], nada['power'], color='black')
# plt.semilogx(1.0/recon['freq'], recon['power'], color='0.0', label='periodogram')
# plt.semilogx(1.0/recon['freq'], recon['sig'], color='0.0', linestyle=':', label='significance line (95%)')
# plt.xlabel('Period (year)')
# plt.ylabel('Power')
# plt.xlim([2, 100])
# plt.annotate(r"11", xy=(11.5,4.4), ha='center', va='center', fontsize=14)
# plt.annotate(r"24", xy=(24,6.5), ha='center', va='center', fontsize=14)
# plt.annotate(r"17", xy=(17,4.4), ha='center', va='center', fontsize=14)

# plt.savefig('plots/spectralRecon.pdf', format='pdf')
# plt.savefig('plots/spectralRecon.png', format='png')

# #plt.legend(loc='upper right')

# #plt.semilogx(1.0/nada['freq'], nada['power'], color='black')
# plt.semilogx(1.0/stahle_drought['freq'], stahle_drought['power'], color='0.0', label='periodogram')
# plt.semilogx(1.0/stahle_drought['freq'], stahle_drought['sig'], color='0.0', linestyle=':', label='significance line (95%)')
# plt.xlabel('Period (year)')
# plt.ylabel('Power')
# plt.xlim([2, 100])
# # plt.annotate(r"11", xy=(11.5,4.4), ha='center', va='center', fontsize=14)
# # plt.annotate(r"24", xy=(24,6.5), ha='center', va='center', fontsize=14)
# # plt.annotate(r"17", xy=(17,4.4), ha='center', va='center', fontsize=14)

fig, ax = plt.subplots(3, sharex=True, sharey=True)

ax[0].semilogx(1.0/recon['freq'], recon['power'], color='0.0', label='periodogram')
ax[0].semilogx(1.0/recon['freq'], recon['sig'], color='0.0', linestyle=':', label='significance line (95%)')
ax[0].set_xlim([2, 150])

ax[1].semilogx(1.0/stahle_drought['freq'], stahle_drought['power'], color='0.0', label='periodogram')
ax[1].semilogx(1.0/stahle_drought['freq'], stahle_drought['sig'], color='0.0', linestyle=':', label='significance line (95%)')
#plt.xlabel('Period (year)')
#plt.ylabel('Power')
###plt.xlim([2, 100])              # 
# plt.annotate(r"11", xy=(11.5,4.4), ha='center', va='center', fontsize=14)
# plt.annotate(r"24", xy=(24,6.5), ha='center', va='center', fontsize=14)
# plt.annotate(r"17", xy=(17,4.4), ha='center', va='center', fontsize=14)


ax[2].semilogx(1.0/potomac['freq'], potomac['power'], color='0.0', label='periodogram')
ax[2].semilogx(1.0/potomac['freq'], potomac['sig'], color='0.0', linestyle=':', label='significance line (95%)')

for a in ax:
  a.axvline(10.744139, color='grey')
  
  a.spines['top'].set_visible(False)
  a.spines['right'].set_visible(False)
  #a.tick_params(left="off")
  a.tick_params(right="off")
  a.tick_params(which="both", top="off")

for a in ax[:-1]:
  a.tick_params(which="both", bottom="off")
  

#fig.subplots_adjust(hspace=0)
ax[2].set_xlabel('Period (year)')
ax[1].set_ylabel('Power')

xtext = 180#math.log10(180)
plt.subplots_adjust(right=0.8)  # make room for labels
plt.text(xtext, 17, 'rSWV', verticalalignment='center')
plt.text(xtext, 10, 'JT', verticalalignment='center')
plt.text(xtext, 3, 'PR', verticalalignment='center')

#plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
#plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)

# plt.xlabel('Period (year)')
# plt.ylabel('Power')
# #plt.xlim([2, 100])
# # plt.annotate(r"11", xy=(11.5,4.4), ha='center', va='center', fontsize=14)
# # plt.annotate(r"24", xy=(24,6.5), ha='center', va='center', fontsize=14)
# # plt.annotate(r"17", xy=(17,4.4), ha='center', va='center', fontsize=14)

plt.savefig('plots/periodograms.pdf', format='pdf')
plt.savefig('plots/prediodograms.png', format='png')

#plt.legend(loc='upper right')

plt.show()   
                           
