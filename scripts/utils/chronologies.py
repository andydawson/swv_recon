
import pandas
from config import base_path

pub_base_names  = [ 'nc', 'lh', 'wd', 'cc', 'oc']
eps_start_years = { 'nc': 1845, 'lh': 1750, 'wd': 1735, 'cc': 1800, 'oc': 1745 }

def published_chronologies(cut_eps=False):

  ## read published chronologies
  chrons = {}
  for base in pub_base_names:
    #chron = pandas.read_fwf('data/chronsPub/' + base,
    #                        widths=[8,8], index_col=[0], header=None, names=['chron'])
    chron = pandas.read_fwf('data/chronsGen/' + base,
                            widths=[8,8], index_col=[0], header=None, names=['chron'])
    if cut_eps:
      chrons[base] = chron['chron'].ix[eps_start_years[base]:]  
    else:
      chrons[base] = chron['chron']  

  return pandas.DataFrame(chrons)
