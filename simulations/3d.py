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
parser.add_argument('-n', type=int, default=200, help= 'number of nodes')
parser.add_argument('-g', type=float, default=1.5, help='multiplier of gamma')
parser.add_argument('-r', type=float, default=2, help='multiplier of rho')
parser.add_argument('-seed', type=int, default=1, help='random seed')
args = parser.parse_args()
param_t=pd.DataFrame({'n': 6*[args.n], 'rho': 6*[args.r],
    'gamma' : 6*[args.g], 'shuffle':[.5,.6,.7,.8,.9,1.0], 'seed': 6*[args. seed]})

results = Parallel(n_jobs=6) (delayed(solve_init) (row) for index, row in param_t.iterrows())
results = np.array(results)

for j in range(results.shape[1]) :
    for i in range(results. shape[0]) :
        print(results[i,j], end = ' ')
    print('')