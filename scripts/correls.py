"""Compute and plot precip, pdsi, and temp correlations."""

import pandas

import utils.pcaFuns as pcaFuns
import utils.corrFuns as corrFuns
import utils.chronologies

from utils.config import base_path


## config
months = ['January', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


## read in gridded data
pdsiGDD   = pandas.read_csv(base_path + 'csv/pdsiGDD.csv',   index_col=[0])
precipGDD = pandas.read_csv(base_path + 'csv/precipGDD.csv', index_col=[0])
#tempGDD   = pandas.read_csv(base_path + 'csv/tempGDD.csv',   index_col=[0], na_values=['-999.900'])
tempGDD   = pandas.read_csv(base_path + 'csv/tempGDD_v2.csv',   index_col=[0], na_values=['-999.900'])

## read in blacksburg data
pdsiBB   = pandas.read_csv(base_path + 'csv/pdsiBB.csv', index_col=[0], na_values=[''])
precipBB = pandas.read_csv(base_path + 'csv/precipBB.csv', index_col=[0], na_values=[''])
tempBB   = pandas.read_fwf(base_path + 'csv/tempBB.txt', index_col=[0], 
                           na_values=['-999.9'], widths=[5,7,7,7,7,7,7,7,7,7,7,7,7],  header=None, names=months)

## read published chronologies
chrons = utils.chronologies.published_chronologies()

## compute and save pca from published chronologies
S, T, E, cols = pcaFuns.pca(chrons, 1845, 1981)

## compute correlations between nc and all other chrons
print '===> correlation of nc with published chronologies'
corrs = corrFuns.corrvals(chrons['nc'], chrons)
print corrs

for series in [ 'nc', 'lh', 'wd', 'cc', 'oc']:
  print '==> correlation of %s with precipGDD' % series
  corr, rcrit = corrFuns.corrvals(chrons[series], precipGDD)
  print corr.xs('mj').values
  
  print '==> correlation of %s with pdsiGDD' % series
  corr, rcrit = corrFuns.corrvals(chrons[series], pdsiGDD)
  print corr.xs('jj').values

  print '==> correlation of %s with tempGDD' % series
  corr, rcrit = corrFuns.corrvals(chrons[series], tempGDD)
  print corr.xs('July').values

corr, rcrit = corrFuns.corrvals(S, precipGDD)
print 'corr between PC1 and precip', corr.xs('mj').values
corr, rcrit = corrFuns.corrvals(S, pdsiGDD)
print 'corr between PC1 and pdsi', corr.xs('jj').values
corr, rcrit = corrFuns.corrvals(S, tempGDD)
print 'corr between PC1 and July', corr.xs('July').values

## make some plots
for df, kind, subplot in [ (precipGDD, 'precip', 311), (pdsiGDD, 'pdsi', 312), (tempGDD, 'temp', 313) ]:

   corrNC, rcrit  = corrFuns.corrvals(chrons['nc'], df)
   corrPCA, rcrit = corrFuns.corrvals(S, df)
  
   corrFuns.corrPlot(corrNC, corrPCA, kind, rcrit, subplot)


#for df, kind, subplot in [ (precipGDD, 'precip', 111) ]:

#  corrNC, rcrit  = corrFuns.corrvals(chrons['nc'], df)
#  corrPCA, rcrit = corrFuns.corrvals(S, df)
#  
#  corrFuns.corrPlot(corrNC, corrPCA, kind, rcrit, subplot)


