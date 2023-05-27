import argparse 
import numpy as np 
import sys 
sys.path.insert(0, '..')
import pcabm.sc as sc 
import pcabm.commFunc as cf 
import pcabm.plem as plem 
import pcabm.dcbm as dc 
import pcabm.pcabm as pcabm 
import pandas as pd 
import time
from joblib import Parallel, delayed 
from scipy.optimize import minimize 
from sklearn.metrics.cluster import adjusted_rand_score 
from problem import *

################################
## Parameters
################################

parser = argparse.ArgumentParser(description= 'Setting the Parameters.')
parser.add_argument('-n', type=int, default=500, help= 'number of nodes')
parser.add_argument('-g', type=float, default=2, help='multiplier of gamma')
parser.add_argument('-r', type=float, default=4, help='multiplier of rho')
parser.add_argument('-seed', type=int, default=1, help='random seed')
args = parser.parse_args()
param_t=pd.DataFrame({'n': 10*[args.n], 'rho': 10*[args.r], 'gamma': 10*[args.g],
    'corr':list(np.arange(10)/10), 'seed': 10*[args. seed]})

results = Parallel(n_jobs=10) (delayed(solve_cov) (row) for index, row in param_t.iterrows())

L= len(results)
for l in range(L) :
    for elem in results[l] :
        print(elem, end = ' ')
    print('')