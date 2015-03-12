import pandas
from config import base_path
from pymc import *
from numpy import *
from matplotlib.pylab import *
from pymc.utils import quantiles
from scipy import stats
from pydendro.normalize import spline


# load data
#pandas.io.parsers.read_csv("pcaBrm_AD.csv", na_values=['#NULL!'])
#scores = pandas.io.parsers.read_csv('pcaNested.csv', index_col=[0])#pandas.io.parsers.read_csv('scores.csv', index_col=[0])
#scores=scores.dropna()

def precipProxy(scores, flag):
    """Compute...
    
    Notes: *scores* is assumed to be a pandas Series.
    """

    scores_late = scores.ix[1901:1981]

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
      
    years = range(1901,1981+1)

    # define priors
    beta  = Normal('beta', mu=zeros(2), tau=.001, value=zeros(2))
    sigma = Uniform('sigma', lower=0., upper=100., value=1.)

    # define predictions
    @deterministic
    def mu(beta=beta, chron=scores_late):
        return beta[0] + beta[1]*chron

    @deterministic
    def predicted(mu=mu, sigma=sigma):
        return rnormal(mu, sigma**-2.)

    # define likelihood
    @observed
    def y(value=climVar_cent, mu=mu, sigma=sigma):
        return normal_like(value, mu, sigma**-2.)

    # generate MCMC samples
    vars = [beta, sigma, mu, predicted, y]
    mc = MCMC(vars)
    mc.use_step_method(Metropolis, beta)
    mc.sample(iter=20000, thin=10, burn=10000, verbose=1)

    betas  = beta.trace.gettrace()
    sigmas = sigma.trace.gettrace()
    chron  = scores.values
    pred   = zeros((betas.shape[0], chron.shape[0]))


    for i in range(betas.shape[0]):
        pred[i, :] = predicted._eval_fun(mu=mu._eval_fun(beta=betas[i], chron=chron), sigma=sigmas[i])

    # plotting setup
    #t = range(1845, 1981+1)
    #t = range(1750, 1981+1)
    t = scores.index

    plot_vals = quantiles(pred, (5, 50, 95))

    recon = pandas.DataFrame(plot_vals[50], index = t, columns=['recon'])
    reconMonthly = pandas.DataFrame({ x: plot_vals[50] for x in ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] }, index=t)
    
    if flag == 'precip':
      recon.to_csv('csv/reconPrecip.csv')
      reconMonthly.to_csv('csv/reconPrecipMonthly.csv', cols=['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    else:
      recon.to_csv('csv/reconPdsi.csv')
      reconMonthly.to_csv('csv/reconPdsiMonthly.csv', cols=['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    #print recon.ix[1750:1850]

    # why do we need to rescale variance? i find this weird
    pred_std   = zeros((betas.shape[0], chron.shape[0]))
    for i in range(betas.shape[0]):
      pred_std[i, :] = pred[i, :]/plot_vals[50].std()*climVar_anom.std() + climVar.mean()
    
    plot_vals = pandas.DataFrame({'pred5': plot_vals[5],
                                  'pred50': plot_vals[50], 
                                  'pred95': plot_vals[95]}, index = t)#, columns=['pred5', 'pred50', 'pred95'])
    plot_vals.to_csv('csv/plot_vals.csv')


    plot_vals_std = quantiles(pred_std, (5, 50, 95))
    
    plot_vals_std = pandas.DataFrame({'pred5': plot_vals_std[5],
                                      'pred50': plot_vals_std[50], 
                                      'pred95': plot_vals_std[95]}, index = t)#, columns=['pred5', 'pred50', 'pred95'])
    plot_vals_std.to_csv('csv/plot_vals_std.csv')
    # return recon, betas, sigmas
    
    #show()

    #plot_vals[50].to_csv('recon.csv')
    #np.savetxt('recon.csv', plot_vals[50], fmt='%.4f', delimiter=',')

    #plot(t, smooth(quantiles[50]), color='black', label='Smoothed Estimate')
    #plot(t, smooth(quantiles[2.5]), color='grey', linestyle='dashed', label='Smoothed 90% Quantiles')
    #plot(t, smooth(quantiles[97.5]), color='grey', linestyle='dashed')
    #plot(t, quantiles[50], color='blue', alpha=.5, linewidth=3, label='Unsmoothed Estimate')
    #plot(t, data.y, color='red', label='HADCRU NH Data')
    #axis([1845, 1981, -1, 1])
    #legend(loc='upper left')

    # scatter residuals
    #figure()
    #scatter(quantiles[50], quantiles[50]-data.y)
    #xlabel('temp_t^predicted')
    #ylabel('residual_t')


# def precipProxy_var(scores, flag):
#     """Compute...
    
#     Notes: *scores* is assumed to be a pandas Series.
#     """

#     scores_late = scores.ix[1901:1981]

#     if flag == 'pdsi':
#       pdsi = pandas.read_csv(base_path + 'csv/jjPdsi.csv', index_col=[0])
#       climVar = pdsi['p'].ix[1901:1981].values
#     else:
#       precip = pandas.read_csv(base_path + 'csv/mjPrecip.csv', index_col=[0])
#       ref_mean = np.mean(precip['precip'].ix[1961:1990].values)
#       climVar = precip['precip'].ix[1901:1981].values
#       climVar_anom = climVar - ref_mean*np.ones(np.shape(climVar)[0])
      
   
          
#     years = range(1901,1981+1)
#     idx   = np.logical_and(1901<=scores.index, scores.index<=1981)
    
#     #variance adjusted chronology
#     stdev  = scores.values*std(climVar_anom)/std(scores.values)
#     scaled = stdev - (stdev.mean()- climVar_anom.mean())
    
#     ################################################################################
#     #linear regression, no rescaling
#     print(climVar.mean())
#     print(ref_mean)
    
#     #center the climate variable
#     climVar_cent = climVar - climVar.mean()
    
#     slope, intercept, r_value, p_value, std_err = stats.linregress(scores_late, climVar_cent)
#     ypreds = intercept + slope*scores.values
#     #    ypreds_p = ypreds + ref_mean*np.ones(np.shape(climVar)[0])
    
#     #rescale variance
#     ypreds_s = ypreds/ypreds.std()*climVar_anom.std()
   
#     #    reconnb = pandas.read_csv(base_path + 'csv/comparePrecipRecons.csv', index_col=[0])
#     #    reconnb = reconnb['non_bayes'].ix[1750:1981].values
    
#     ################################################################################
#     #Valerie
    
#     recon = stdev - (stdev.mean() - climVar_anom.mean())*np.ones(len(stdev)) #+ ref_mean
    
#     ################################################################################
#     #regression with scaling
    
#     slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(scaled[idx], climVar_anom)
#     ypreds1 = intercept1 + slope1*scaled #+ ref_mean
    
    
#     ####################################################################################
#     #plots
        
#     #    plot(ypreds, '-k')
#     #    plot(ypreds1, '-r')
#     #    plot(reconnb, '-b')
    
#     #plot(ypreds1, '-k')
    
#     plot(ypreds_s, '-k')
#     plot(recon, '-r')
#     show()

#     ################################################################################
#     #MCMC
    
#     # define priors
#     beta  = Normal('beta', mu=zeros(2), tau=.001, value=zeros(2))
#     sigma = Uniform('sigma', lower=0., upper=100., value=1.)

#     # define predictions
#     @deterministic
#     def mu(beta=beta, chron=scores_late):
#         return beta[0] + beta[1]*chron

#     @deterministic
#     def predicted(mu=mu, sigma=sigma):
#         return rnormal(mu, sigma**-2.)

#     # define likelihood
#     @observed
#     # def y(value=climVar_cent, mu=mu, sigma=sigma):
#     #     return normal_like(value, mu, sigma**-2.)

#     def y(value=climVar_cent, mu=mu, sigma=sigma):
#         return normal_like(value, mu, sigma**-2.)

#     # generate MCMC samples
#     vars = [beta, sigma, mu, predicted, y]
#     mc = MCMC(vars)
#     mc.use_step_method(Metropolis, beta)
#     mc.sample(iter=20000, thin=10, burn=10000, verbose=1)

#     betas  = beta.trace.gettrace()
#     sigmas = sigma.trace.gettrace()
#     chron  = scores.values
#     pred   = zeros((betas.shape[0], chron.shape[0]))

#     for i in range(betas.shape[0]):
#         pred[i, :] = predicted._eval_fun(mu=mu._eval_fun(beta=betas[i], chron=chron), sigma=sigmas[i])       
#         #try to rescale to make it match with recon
#         # pred[i, :] = pred[i, :] / pred[i,:].std() * climVar_anom.std() # 
#         # print("pred i")
#         # print(pred[i,:])

#         # print("ypreds.std")
#         # print(ypreds.std())
#         #pred[i, :] = pred[i, :]/ypreds.std()*climVar_anom.std()

#         #pred[i, :] = pred[i, :]/asarray(pred[i,:]).std()*climVar_anom.std()
#     ################################################################################

#     # plotting setup
#     #t = range(1845, 1981+1)
#     #t = range(1750, 1981+1)
#     t = scores.index

#     plot_vals = quantiles(pred, (5, 50, 95))
    
#     for i in range(betas.shape[0]):
#       pred[i, :] = pred[i, :]/plot_vals[50].std()*climVar_anom.std()

#     plot_vals = quantiles(pred, (5, 50, 95))

#     def smooth(x):
#         from pymc import gp
#         M = gp.Mean(lambda x: zeros(len(x)))
#         C = gp.Covariance(gp.matern.euclidean, amp=1, scale=15, diff_degree=2)
#         gp.observe(M, C, range(len(x)), x, .5)
#         return M(range(len(x)))

#     #plot data and fitted line
#     idx = np.argsort(chron)
#     lw=2.5
 
#     idx = np.logical_and(1901 <= t, t <= 1981)

     
#     figure(1)
        
#     locator_params(axis='y', nbins=4)
#     top = plot_vals[95]
#     bot = plot_vals[5]
#     mean = plot_vals[50]


#     fill_between(t, bot, top, color='0.8')    
#     plot(t, mean, '-k', lw=lw)
#     #plot(t, mean_adj - ref_mean, '-k', lw=lw)
#     xlim([1745, 1985])
#     #ylabel('Average MJ Precipitation (mm)')
#     #xlabel('Year')
    
#     plot(range(1901,1982), climVar_cent, '-r', lw=lw/2) 
#     plot(t, recon, '-b', lw=lw/2)     
    
#     xlim([1745, 1985])
#     ylabel('Average MJ Precipitation (mm)')
#     xlabel('Year')
    
#     show()

#     # plot(mean, '-k')
#     # plot(recon, '-r')

#     # show()
        
#     return recon, betas, sigmas
    



