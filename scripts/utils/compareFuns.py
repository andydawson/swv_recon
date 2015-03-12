import numpy as np

from scipy.stats import pearsonr
from scipy.signal import get_window

def moving_correlation(x, y, window):
  
  w = len(window)/2
  
  imin = 0
  imax = x.shape[0] - 1
  
  corr = np.zeros(x.shape[0])
  pval = np.zeros(x.shape[0])
  for i in range(imin + w, imax - w + 1):#range(imax+1):
    i1 = i - w #max(i-w, imin)
    i2 = i + w #min(i+w, imax)
    
    #w1 = w - (i - i1)
    #w2 = w + (i2 - i)
    
    a1 = x[i1:i2+1] * window#[w1:w2+1]
    a2 = y[i1:i2+1] * window#[w1:w2+1]
    
    corr[i], pval[i] = pearsonr(a1, a2)
    
  return corr[w:-w], pval[w:-w]
