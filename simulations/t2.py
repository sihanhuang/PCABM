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
parser.add_argument('-n', type=int, default=1000, help= 'number of nodes')
parser.add_argument('-g', type=float, default=1, help='multiplier of gamma')
parser.add_argument('-r', type=float, default=6, help='multiplier of rho')
parser.add_argument('-seed', type=int, default=1, help='random seed')
args = parser.parse_args()
param_t=pd.DataFrame({'n': 3*[args.n], 'rho': 3*[args.r], 'gamma': 3*[args.g],
    'k':[2,3,4], 'seed': 3*[args. seed]})

results = Parallel(n_jobs=3) (delayed(solve_K) (row) for index, row in param_t.iterrows())
results = np.array(results)

for j in range(results.shape[1]) :
    for i in range(results. shape[0]) :
        print(results[i,j], end = ' ')
    print('')