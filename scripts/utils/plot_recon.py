import pandas
from config import base_path
from pylab import *

from pydendro.normalize import spline
# def spline(x, nyears=10):
#     from pydendro.normalize import spline
#     x = np.asarray(x)
#     return spline(x, nyears=nyears)

def smooth(x):
    from pymc import gp
    M = gp.Mean(lambda x: zeros(len(x)))
    C = gp.Covariance(gp.matern.euclidean, amp=1, scale=15, diff_degree=2)
    gp.observe(M, C, range(len(x)), x, .5)
    return M(range(len(x)))

def plot_pc_vs_precip(scores, plot_vals, flag):

    scores_late = scores.ix[1901:1981]
    chron = scores.values

    #plot data and fitted line
    idx = np.argsort(chron)

    if flag == 'pdsi':
      pdsi = pandas.read_csv(base_path + 'csv/jjPdsi.csv', index_col=[0])
      climVar = pdsi['p'].ix[1901:1981].values
    else:
      precip = pandas.read_csv(base_path + 'csv/mjPrecip.csv', index_col=[0])
      ref_mean = np.mean(precip['precip'].ix[1961:1990].values)
      climVar = precip['precip'].ix[1901:1981].values
      climVar_anom = climVar - ref_mean*np.ones(np.shape(climVar)[0])
      #center the climate variable
      climVar_cent = climVar - climVar.mean()

    plot(scores_late, climVar, 'ob')
    plot(chron[idx], smooth(plot_vals['pred50'][idx]), '-k')
    fill_between(chron[idx], smooth(plot_vals['pred5'][idx]), smooth(plot_vals['pred95'][idx]), color='0.8')
    xlabel('PC1')
    ylabel('MJ Precipitation')

    show()

    if flag == 'precip':
      savefig('plots/pc_vs_precip.pdf')
    else:
      savefig('plots/pc_vs_pdsi.pdf')  


def plot_recon(scores, plot_vals):

  t    = scores.index

  top  = plot_vals['pred95']
  bot  = plot_vals['pred5']
  mean = plot_vals['pred50']

  fill_between(t, bot, top, color='0.8')    
  plot(t, mean, color='0.3')
  xlim([1745, 1985])
  ylabel('Average MJ Precipitation (mm)')
  xlabel('Year')

  smoothed_mean = plot_vals['pred50'] / spline(plot_vals['pred50'], nyears=10)
  plot(t, smoothed_mean, color='0.1', linewidth=1.5)

  show()

#     savefig('plots/recon.pdf')
#     savefig('plots/recon.png', dpi=300)


    
#     #plot(years, climVar, color='0.0', linewidth=2, label='MJ Precip data')

#     figure(3)
#     fill_between(t, bot, top, color='0.8')    
#     plot(years, climVar, color='0.0', label='MJ Precip data')
#     xlim([1895, 1987])
    
    
    
#     #axis([1845, 1981, -1, 1])
#     #legend(loc='upper left')

#     if flag == 'precip':
#       savefig('plots/reconPrecip.pdf')
#     else:
#       savefig('plots/reconPdsi.pdf')
#     #plt.savefig('reconPrecip.pdf', format='pdf')
#     #show()


#     #recon = DataFrame(plot_vals[50], index = t, columns=['recon'])
#     #recon.to_csv('reconPrecip.csv')

#     figure(8)
#     plot(scores_late, plot_vals[50][ np.logical_and(1901 <= t, t <= 1981)] + climVar.mean(), 'ob', label='Predictions raw')

#     pred_raw = pred
#     plot_vals_raw = plot_vals

#     # why do we need to rescale variance? i find this weird
#     for i in range(betas.shape[0]):
#       pred[i, :] = pred[i, :]/plot_vals[50].std()*climVar_anom.std() + climVar.mean()

#     plot_vals = quantiles(pred, (5, 50, 95))
#     idx = np.logical_and(1901 <= t, t <= 1981)
    
#     # figure(6)
#     plot(scores_late, climVar_cent + climVar.mean(), 'or', label='Raw')

#     # figure(7)
#     plot(scores_late, plot_vals[50][idx], 'ok', label='Rescaled')
#     legend()
#     savefig('plots/scatter_check_scaling.pdf')

#     figure(9)
#     lw=2.5
#     plot(range(1901,1982), climVar, '-r', lw=lw, label='Raw') 
#     plot(range(1901,1982), plot_vals[50][idx], '-k', lw=lw, label='Rescaled')
#     plot(range(1901,1982), plot_vals_raw[50][idx]+climVar.mean(), '-b', lw=lw, label='Raw predictions')
#     xlim([1896, 1986])
#     legend()
#     savefig('plots/preds_check_scaling.pdf')


def plot_paper_fig(scores, plot_vals, flag):

  t    = scores.index
  pred_top  = plot_vals['pred95']
  pred_bot  = plot_vals['pred5']
  pred_mean = plot_vals['pred50']

  lw=2.5
    
  idx = np.logical_and(1901 <= t, t <= 1981)

  if flag == 'pdsi':
      pdsi = pandas.read_csv(base_path + 'csv/jjPdsi.csv', index_col=[0])
      climVar = pdsi['p'].ix[1901:1981].values
  else:
      precip = pandas.read_csv(base_path + 'csv/mjPrecip.csv', index_col=[0])
      ref_mean = np.mean(precip['precip'].ix[1961:1990].values)
      climVar = precip['precip'].ix[1901:1981].values
      climVar_anom = climVar - ref_mean*np.ones(np.shape(climVar)[0])
      #center the climate variable
      climVar_cent = climVar - climVar.mean()
    
  subplot(311)
  locator_params(axis='y', nbins=4)
  #plot(range(1901,1982), climVar_cent, '-r', lw=lw) 
  plot(range(1901,1982), climVar, '-r', lw=lw) 
  plot(range(1901,1982), pred_mean[idx], '-k', lw=lw)
  plot(range(1890,2000), 110*[climVar.mean()], '0.3', linestyle='--')
  xlim([1896, 1986])
  
  
  subplot(312)
  locator_params(axis='y', nbins=4)
  fill_between(t, pred_bot, pred_top, color='0.8')    
  plot(t, pred_mean, '-k', lw=lw)
  xlim([1745, 1985])
  #ylabel('Average MJ Precipitation (mm)')
  #xlabel('Year')
  
###    plot(range(1901,1982), climVar_cent, '-r', lw=lw/2) 
  plot(range(1901,1982), climVar, '-r', lw=lw/2) 
  plot(range(1740,1990), 250*[climVar.mean()], '0.3', linestyle='--')
    
  subplot(313)
  locator_params(axis='y', nbins=4)
   
  nyears = 10
  smooth = lambda x: x / spline(x, nyears=nyears)

  top = smooth(pred_top) #- climVar.mean()
  bot = smooth(pred_bot+1000)-1000 #-1000 #- climVar.mean() - 1000
  mean = smooth(pred_mean) #- climVar.mean()
  fill_between(t, bot, top, color='0.8')    
  plot(t, mean, '-k', lw=lw)
  xlim([1745, 1985])
  #ylim([-200, 200])
  #ylabel('Average MJ Precipitation (mm)')
  xlabel('Year')
  
    
  ax = gca()
  ax.yaxis.set_label_coords(-0.1, 1.6)
  ax.set_ylabel('Average MJ Precipitation (mm)')
    
  plot(range(1901,1982), smooth(climVar), '-r', lw=lw)#'0.2', linestyle='--', lw=lw) 
#    plot(range(1901,1982), smooth(climVar) - climVar.mean(), '-r', lw=lw) 
  plot(range(1740,1990), 250*[climVar.mean()], '0.3', linestyle='--')

  for a in gcf().get_axes():
      a.spines['top'].set_visible(False)
      a.spines['right'].set_visible(False)
      a.tick_params(top="off")
      a.tick_params(right="off")

    # #ax[4].spines['bottom'].set_visible(True)
    # #a.tick_params(bottom="on")

    # #smoothed_mean = plot_vals[50] / spline(plot_vals[50], nyears=10)
    # #plot(t, smoothed_mean, color='0.1', linewidth=1.5)
  savefig('plots/reconsmoothed10.pdf')
  savefig('plots/reconsmoothed10.png', dpi=300)

  show()







#     reconnb = pandas.read_csv(base_path + 'csv/comparePrecipRecons.csv', index_col=[0])
#     reconnb = reconnb['non_bayes'].ix[1750:1981].values

#     figure(5)
        
#     locator_params(axis='y', nbins=4)
#     top = plot_vals[95]
#     bot = plot_vals[5]
#     mean = plot_vals[50]
#     fill_between(t, bot, top, color='0.8')    
#     plot(t, mean, '-k', lw=lw, label='Rescaled')
#     xlim([1745, 1985])
#     #ylabel('Average MJ Precipitation (mm)')
#     #xlabel('Year')
    
#     plot(range(1901,1982), climVar, '-r', lw=lw/2, label='Raw') 
#     plot(t, reconnb, '-b', lw=lw/2, label='Non-bayes')     
    
#     xlim([1745, 1985])
#     ylabel('Average MJ Precipitation (mm)')
#     xlabel('Year')
#     legend()
    
#     savefig('plots/recon_check_scaling.pdf')
#     savefig('plots/sucky.png', dpi=300)
