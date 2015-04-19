"""Compare to climate indices (ENSO and NAO)"""

import matplotlib.pylab as plt
import matplotlib.ticker
import numpy as np
import pandas

np.set_printoptions(linewidth=120)

from scipy.stats import pearsonr
from utils.compareFuns import moving_correlation, get_window

from utils.config import base_path

from pydendro.normalize import spline

nino = pandas.read_csv(base_path + 'otherRecons/nino_cook.csv', index_col=[0],  names=['Nino1p2', 'Nino3', 'Nino3p4', 'Nino4' ], header=None)
nao  = pandas.read_csv(base_path + 'otherRecons/nao_trouet.csv', index_col=[0])

# my recons
recon_precip = pandas.read_csv(base_path + 'csv/reconPrecip.csv', index_col=[0])
recon_pdsi   = pandas.read_csv(base_path + 'csv/reconPdsi.csv', index_col=[0])

start = 1750
end   = 1981

nino_1p2 = nino['Nino1p2'].ix[start:end].values.flatten()
nino_3   = nino['Nino3'].ix[start:end].values.flatten()
nino_3p4 = nino['Nino3p4'].ix[start:end].values.flatten()
nino_4   = nino['Nino4'].ix[start:end].values.flatten()
nao      = nao.ix[start:end].values.flatten()

recon_precip = recon_precip.ix[start:end].values.flatten()
recon_pdsi   = recon_pdsi.ix[start:end].values.flatten()

mat = np.vstack([recon_precip, nino_1p2, nino_3, nino_3p4, nino_4, nao])

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
