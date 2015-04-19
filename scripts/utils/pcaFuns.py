
import numpy as np

from pca_module import PCA_svd as PCA
from scipy.stats import pearsonr

import pandas


def GLK(x, y):
  """GLK between two NumPy arrays."""
  dx = x[1:] - x[:-1]
  dy = y[1:] - y[:-1]

  Gx = np.zeros(dx.shape)
  Gx[dx<0.0] = -0.5
  Gx[dx>0.0] = 0.5

  Gy = np.zeros(dy.shape)
  Gy[dy<0.0] = -0.5
  Gy[dy>0.0] = 0.5
  
  return 1.0/(len(x) - 1.0) * np.sum(np.abs(Gx + Gy))


def pca(df, start, end, return_raw_scores=False):
  """Compute PCA of df from start to end."""

  chrons = df.ix[start:end]

  for series in chrons:
    if any(chrons[series].isnull()):
      del chrons[series]

  x = chrons.values

  # for i in range(1, 4):
  #   corr, pval = pearsonr(x[:, 0], x[:, i])
  #   print corr, pval

  S, T, V, E = PCA(x, standardize=False)
  
  if start==1750:
    S=-S

  if return_raw_scores:
    return S, T, V, E, chrons.columns
    
  S = pandas.Series(S[:, 0], index=chrons.index)
  return S, T, V, E, chrons.columns


def build_recon(pAnom, pca, start=1901, end=1981):
  """Build scaled reconstruction."""

  pAnom  = pAnom.ix[start:end]

  mpca   = float(pca.ix[start:end].mean())
  mpAnom = float(pAnom.mean())

  spca   = float(pca.ix[start:end].std())
  spAnom = float(pAnom.std())

  stdev  = pca.ix[1901:1981] * spAnom / spca
  scaled = stdev - (float(stdev.ix[start:end].mean()) - mpAnom)

  return scaled, stdev

def test_recon(pAnom, scaled, start1, end1, start2, end2):
  """Compute reconstruction statistics."""

  pAnom1 = pAnom.ix[start1:end1].values.flatten()
  pAnom2 = pAnom.ix[start2:end2].values.flatten()

  scaled1 = scaled.ix[start1:end1].values.flatten()
  scaled2 = scaled.ix[start2:end2].values.flatten()

  mpAnom1 = pAnom1.mean()
  mpAnom2 = pAnom2.mean()

  pMinusScaled = pAnom2 - scaled2
  pMinusCal    = pAnom2 - mpAnom1
  pMinusVer    = pAnom2 - mpAnom2

  SSD  = (pMinusScaled*pMinusScaled).sum()
  SSMa = (pMinusCal*pMinusCal).sum()
  SSMb = (pMinusVer*pMinusVer).sum()

  MSE = SSD
  RE  = 1 - SSD/SSMa
  CE  = 1 - SSD/SSMb

  glk = GLK(pAnom2, scaled2)

  rcal, p = pearsonr(pAnom1, scaled1)
  rver, p = pearsonr(pAnom2, scaled2)

  return { 'mse': MSE, 're': RE, 'ce': CE, 'rcal': rcal, 'rver': rver, 'glk': glk }


def test_chron(df, kind, start=1845, end=1981, splice=False):
  """Note: kind is either 'mjPrecip' or 'jjPdsi'."""
  
  S1, T1, E1, cols = pca(df, start, end)

  if splice:
    S2, T2, E2, cols = pca(df, 1845, 1981)
    S = S1.ix[1750:1844].append(S2.ix[1845:1981])
  else:
    S = S1

  #precip = pandas.read_csv('csv/mjPrecip.csv', index_col=[0])
  #precip = pandas.read_csv('csv/jjPdsi.csv', index_col=[0])
  precip = pandas.read_csv('csv/'+kind+'.csv', index_col=[0])
  pAnom  = precip - precip.ix[1961:1990].mean()

  scaled0, stdev0 = build_recon(pAnom, S, 1901, 1981)
  scaled1, stdev1 = build_recon(pAnom, S, 1901, 1940)
  scaled2, stdev2 = build_recon(pAnom, S, 1941, 1981)

  stats1 = test_recon(pAnom, scaled1, 1901, 1940, 1941, 1981)
  stats2 = test_recon(pAnom, scaled2, 1941, 1981, 1901, 1940)

  return stats1, stats2


def test_scores(scores):

  #precip = pandas.read_csv('csv/mjPrecip.csv', index_col=[0])
  precip = pandas.read_csv('csv/jjPdsi.csv', index_col=[0])
  pAnom  = precip - precip.ix[1961:1990].mean()

  scaled0, stdev0 = build_recon(pAnom, scores, 1901, 1981)
  scaled1, stdev1 = build_recon(pAnom, scores, 1901, 1940)
  scaled2, stdev2 = build_recon(pAnom, scores, 1941, 1981)

  stats1 = test_recon(pAnom, scaled1, 1901, 1940, 1941, 1981)
  stats2 = test_recon(pAnom, scaled2, 1941, 1981, 1901, 1940)

  return stats1, stats2


def build_recon_vt(df, kind, start=1901, end=1981):
  """Build scaled reconstruction."""
  
  S1, T1, E1, cols = pca(df, start, end)

  #precip = pandas.read_csv('csv/mjPrecip.csv', index_col=[0])
  #precip = pandas.read_csv('csv/jjPdsi.csv', index_col=[0])
  precip = pandas.read_csv('csv/'+kind+'.csv', index_col=[0])
  pAnom  = precip - precip.ix[1961:1990].mean()
  
  pAnom  = pAnom.ix[1901:1981]

  mpca   = float(S1.ix[start:end].mean())
  mpAnom = float(pAnom.mean())

  spca   = float(S1.ix[start:end].std())
  spAnom = float(pAnom.std())

  stdev  = S1.ix[start:end] * spAnom / spca
  scaled = stdev - (float(stdev.ix[start:end].mean()) - mpAnom)
  
  recon = pandas.DataFrame(scaled, columns=['recon'])
  recon.to_csv('csv/reconNoBayes.csv')

  return scaled, stdev

